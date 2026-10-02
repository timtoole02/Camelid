//! Watched folders: a collection can watch a folder on this computer and holds
//! every document in it, kept current. A scan reads only files whose size or
//! modification time changed since the last one, re-indexes a file whose
//! content changed, and removes the documents of files that are gone. Scans
//! run when a folder is added, when asked, and every `POLL_INTERVAL` while the
//! server runs.
//!
//! A folder is read from this computer's disk, so every folder route needs the
//! local web UI on a loopback listener, the same rule as choosing a workspace.
//! Hidden entries (a leading `.`) and symbolic links are skipped, and only the
//! library's document types are read.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use super::citations::{sha256_hex, ChunkSpan};
use super::document_collections::{self, CollectionError};
use super::document_vectors;
use super::documents::{self, db_lock, open_connection, NewDocument, StoreError};
use super::{api_error, AppState};

/// Documents one folder may hold; a larger folder is refused, not truncated.
pub(crate) const MAX_FOLDER_DOCUMENTS: usize = 10_000;
/// Entries of any kind a scan visits before it gives up on the folder.
const MAX_FOLDER_ENTRIES: usize = 200_000;
/// Folder nesting a scan descends; deeper folders are not read.
const MAX_FOLDER_DEPTH: usize = 32;
/// A larger file is listed as skipped instead of read.
pub(crate) const MAX_DOCUMENT_BYTES: u64 = 64 * 1024 * 1024;
const POLL_INTERVAL: Duration = Duration::from_secs(30);
const LISTED_SKIPS: usize = 50;

const INDEXED: &str = "indexed";
const TOO_LARGE: &str = "too_large";
const UNREADABLE: &str = "unreadable";
const NO_TEXT: &str = "no_text";
const EXTRACT_FAILED: &str = "extract_failed";

pub(crate) fn init_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS watched_folders (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            collection_id TEXT NOT NULL REFERENCES document_collections(id) ON DELETE CASCADE,
            created_at INTEGER NOT NULL,
            last_scan_at INTEGER,
            last_error TEXT,
            last_changes TEXT
        );
        CREATE TABLE IF NOT EXISTS watched_files (
            folder_id TEXT NOT NULL REFERENCES watched_folders(id) ON DELETE CASCADE,
            rel_path TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            modified_ns INTEGER NOT NULL,
            source_sha256 TEXT,
            status TEXT NOT NULL,
            PRIMARY KEY (folder_id, rel_path)
        );",
    )
}

/// What one scan changed in the library.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanChanges {
    /// Documents added for files new to the library.
    pub added: usize,
    /// Documents re-indexed because their file's content changed.
    pub updated: usize,
    /// Documents removed because their file is gone or can no longer be read.
    pub removed: usize,
    /// Files whose document was already current.
    pub unchanged: usize,
    /// Files that hold no document (see `SkippedFile::reason`).
    pub skipped: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SkippedFile {
    pub path: String,
    /// `too_large`, `unreadable`, `no_text`, or `extract_failed`.
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScanProgress {
    pub done: usize,
    pub total: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FolderView {
    pub id: String,
    pub path: String,
    pub collection_id: String,
    pub created_at: i64,
    pub last_scan_at: Option<i64>,
    pub last_error: Option<String>,
    pub last_changes: Option<ScanChanges>,
    /// Files whose document is in the library.
    pub documents: usize,
    pub skipped_count: usize,
    /// The first skipped files by path.
    pub skipped: Vec<SkippedFile>,
    pub scanning: bool,
    pub queued: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<ScanProgress>,
}

#[derive(Debug, Deserialize)]
pub struct WatchFolderRequest {
    pub path: String,
    pub collection_id: String,
}

#[derive(Debug)]
pub(crate) enum FolderError {
    Forbidden,
    InvalidPath(&'static str),
    Overlaps(String),
    FolderNotFound(String),
    Collection(CollectionError),
    Database(rusqlite::Error),
}

impl From<rusqlite::Error> for FolderError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Database(error)
    }
}

impl FolderError {
    fn into_response(self) -> Response {
        let (status, code, message, param) = match self {
            Self::Forbidden => (
                StatusCode::FORBIDDEN,
                "local_management_forbidden",
                "Watching a folder needs Camelid's web UI on this computer.".to_string(),
                None,
            ),
            Self::InvalidPath(reason) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_folder",
                reason.to_string(),
                Some("path"),
            ),
            Self::Overlaps(path) => (
                StatusCode::CONFLICT,
                "folder_overlaps",
                format!("{path} is already watched, or is inside or around a watched folder."),
                Some("path"),
            ),
            Self::FolderNotFound(id) => (
                StatusCode::NOT_FOUND,
                "folder_not_found",
                format!("No watched folder has the id {id:?}."),
                None,
            ),
            Self::Collection(error) => return error.into_response(),
            Self::Database(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "folder_database_error",
                error.to_string(),
                None,
            ),
        };
        api_error(status, code, message, param)
    }
}

/// Folder routes read this computer's disk, so they answer only the local web
/// UI (or a same-origin request) on a loopback listener.
fn local_access(serve_addr: SocketAddr, headers: &HeaderMap) -> bool {
    serve_addr.ip().is_loopback() && super::workspace::local_management_request_allowed(headers)
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// The folder `raw` names, resolved: absolute, existing, a directory, not a
/// filesystem root, and Unicode.
pub(crate) fn canonical_folder(raw: &str) -> Result<PathBuf, FolderError> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(FolderError::InvalidPath("Choose a folder to watch."));
    }
    let path = Path::new(raw);
    if !path.is_absolute() {
        return Err(FolderError::InvalidPath("Give the folder's full path."));
    }
    let canonical = std::fs::canonicalize(path)
        .map_err(|_| FolderError::InvalidPath("That folder does not exist or cannot be opened."))?;
    if !canonical.is_dir() {
        return Err(FolderError::InvalidPath("That path is not a folder."));
    }
    if canonical.parent().is_none() {
        return Err(FolderError::InvalidPath(
            "Watch a folder, not a whole drive.",
        ));
    }
    if canonical.to_str().is_none() {
        return Err(FolderError::InvalidPath(
            "Camelid can watch only folders whose path is valid Unicode.",
        ));
    }
    Ok(canonical)
}

/// Starts watching `canonical` (from `canonical_folder`) into a collection.
/// A folder inside, around, or equal to a watched one would hold the same files
/// twice, so it is refused.
pub(crate) fn add_folder(
    conn: &Connection,
    canonical: &Path,
    collection_id: &str,
) -> Result<String, FolderError> {
    document_collections::require_collections(conn, &[collection_id.to_string()])
        .map_err(FolderError::Collection)?;
    let mut statement = conn.prepare("SELECT path FROM watched_folders")?;
    let watched = statement
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    let display = super::workspace::simplify_path(canonical);
    if watched.iter().any(|other| {
        let other = Path::new(other);
        canonical.starts_with(other) || other.starts_with(canonical)
    }) {
        return Err(FolderError::Overlaps(display));
    }
    let id = uuid::Uuid::new_v4().to_string();
    conn.execute(
        "INSERT INTO watched_folders (id, path, collection_id, created_at) VALUES (?1, ?2, ?3, ?4)",
        params![id, canonical.to_str(), collection_id, now_secs()],
    )?;
    Ok(id)
}

/// Stops watching a folder and removes its documents from the library. The
/// files on disk are not touched.
pub(crate) fn remove_folder(conn: &mut Connection, id: &str) -> Result<usize, FolderError> {
    let tx = conn.transaction()?;
    let doc_ids = {
        let mut statement = tx.prepare("SELECT doc_id FROM watched_files WHERE folder_id = ?1")?;
        let rows = statement.query_map(params![id], |row| row.get::<_, String>(0))?;
        rows.collect::<Result<Vec<_>, _>>()?
    };
    if tx.execute("DELETE FROM watched_folders WHERE id = ?1", params![id])? == 0 {
        return Err(FolderError::FolderNotFound(id.to_string()));
    }
    let mut removed = 0;
    for doc_id in doc_ids {
        removed += usize::from(documents::remove_document(&tx, &doc_id)?);
    }
    tx.commit()?;
    Ok(removed)
}

/// Every watched folder, oldest first, with what the scanner is doing now.
pub(crate) fn list_folders(conn: &Connection) -> Result<Vec<FolderView>, rusqlite::Error> {
    let mut statement = conn.prepare(
        "SELECT f.id, f.path, f.collection_id, f.created_at, f.last_scan_at, f.last_error, f.last_changes,
                (SELECT COUNT(*) FROM watched_files AS w WHERE w.folder_id = f.id AND w.status = 'indexed'),
                (SELECT COUNT(*) FROM watched_files AS w WHERE w.folder_id = f.id AND w.status != 'indexed')
         FROM watched_folders AS f ORDER BY f.created_at, f.id",
    )?;
    let folders = statement
        .query_map([], |row| {
            Ok(FolderView {
                id: row.get(0)?,
                path: super::workspace::simplify_path(Path::new(&row.get::<_, String>(1)?)),
                collection_id: row.get(2)?,
                created_at: row.get(3)?,
                last_scan_at: row.get(4)?,
                last_error: row.get(5)?,
                last_changes: row
                    .get::<_, Option<String>>(6)?
                    .and_then(|json| serde_json::from_str(&json).ok()),
                documents: row.get::<_, i64>(7)? as usize,
                skipped_count: row.get::<_, i64>(8)? as usize,
                skipped: Vec::new(),
                scanning: false,
                queued: false,
                progress: None,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let mut skipped = conn.prepare(
        "SELECT rel_path, status FROM watched_files
         WHERE folder_id = ?1 AND status != 'indexed' ORDER BY rel_path LIMIT ?2",
    )?;
    let scans = scans().lock().unwrap();
    folders
        .into_iter()
        .map(|mut folder| {
            folder.skipped = skipped
                .query_map(params![folder.id, LISTED_SKIPS as i64], |row| {
                    Ok(SkippedFile {
                        path: row.get(0)?,
                        reason: row.get(1)?,
                    })
                })?
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(current) = scans
                .current
                .as_ref()
                .filter(|current| current.0 == folder.id)
            {
                folder.scanning = true;
                folder.progress = Some(current.1);
            }
            folder.queued = scans.queue.contains(&folder.id);
            Ok(folder)
        })
        .collect()
}

fn folder_view(conn: &Connection, id: &str) -> Result<FolderView, FolderError> {
    list_folders(conn)?
        .into_iter()
        .find(|folder| folder.id == id)
        .ok_or_else(|| FolderError::FolderNotFound(id.to_string()))
}

struct FoundFile {
    rel_path: String,
    path: PathBuf,
    byte_size: u64,
    modified_ns: i64,
}

#[derive(Default)]
struct Walk {
    files: Vec<FoundFile>,
    /// Folders (relative paths, `""` for the root) that could not be listed in
    /// full. Documents of files under them are kept rather than removed.
    unreadable: Vec<String>,
}

fn modified_ns(metadata: &std::fs::Metadata) -> i64 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_nanos() as i64)
}

/// Lists the documents under `root` without following links or entering
/// hidden entries. `Err` means the folder cannot be scanned at all.
fn walk(root: &Path) -> Result<Walk, String> {
    let mut walk = Walk::default();
    let mut visited = 0;
    let mut pending = vec![(root.to_path_buf(), String::new(), 0)];
    while let Some((dir, prefix, depth)) = pending.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) if prefix.is_empty() => {
                return Err(
                    "The folder cannot be opened. It may have been moved, renamed or disconnected."
                        .to_string(),
                );
            }
            Err(_) => {
                walk.unreadable.push(prefix);
                continue;
            }
        };
        for entry in entries {
            let Ok(entry) = entry else {
                walk.unreadable.push(prefix.clone());
                continue;
            };
            visited += 1;
            if visited > MAX_FOLDER_ENTRIES {
                return Err(format!(
                    "The folder holds more than {MAX_FOLDER_ENTRIES} files and folders. Watch a smaller folder."
                ));
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            // `file_type` describes a link itself, so links are never followed.
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            let rel_path = if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}/{name}")
            };
            if kind.is_dir() {
                if depth + 1 < MAX_FOLDER_DEPTH {
                    pending.push((entry.path(), rel_path, depth + 1));
                }
                continue;
            }
            if !kind.is_file() || !documents::has_document_extension(name) {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                walk.unreadable.push(rel_path);
                continue;
            };
            walk.files.push(FoundFile {
                rel_path,
                path: entry.path(),
                byte_size: metadata.len(),
                modified_ns: modified_ns(&metadata),
            });
            if walk.files.len() > MAX_FOLDER_DOCUMENTS {
                return Err(format!(
                    "The folder holds more than {MAX_FOLDER_DOCUMENTS} documents. Watch a smaller folder."
                ));
            }
        }
    }
    walk.files
        .sort_by(|left, right| left.rel_path.cmp(&right.rel_path));
    walk.unreadable.sort();
    walk.unreadable.dedup();
    Ok(walk)
}

/// A file as the last scan left it.
struct Known {
    doc_id: String,
    byte_size: i64,
    modified_ns: i64,
    source_sha256: Option<String>,
    status: String,
    has_document: bool,
}

fn known_files(
    conn: &Connection,
    folder_id: &str,
) -> Result<HashMap<String, Known>, rusqlite::Error> {
    let mut statement = conn.prepare(
        "SELECT w.rel_path, w.doc_id, w.byte_size, w.modified_ns, w.source_sha256, w.status,
                d.id IS NOT NULL
         FROM watched_files AS w LEFT JOIN documents AS d ON d.id = w.doc_id
         WHERE w.folder_id = ?1",
    )?;
    let rows = statement.query_map(params![folder_id], |row| {
        let status: String = row.get(5)?;
        let exists: bool = row.get(6)?;
        Ok((
            row.get::<_, String>(0)?,
            Known {
                doc_id: row.get(1)?,
                byte_size: row.get(2)?,
                modified_ns: row.get(3)?,
                source_sha256: row.get(4)?,
                has_document: exists && status == INDEXED,
                status,
            },
        ))
    })?;
    rows.collect()
}

enum Read {
    /// The bytes hash as before: only the file's size or time moved.
    Same(String),
    Document {
        text: String,
        sha256: String,
        chunks: Vec<ChunkSpan>,
    },
    Skip(&'static str, Option<String>),
}

fn read_document(file: &FoundFile, known: Option<&Known>) -> Read {
    if file.byte_size > MAX_DOCUMENT_BYTES {
        return Read::Skip(TOO_LARGE, None);
    }
    let Ok(bytes) = std::fs::read(&file.path) else {
        return Read::Skip(UNREADABLE, None);
    };
    if bytes.len() as u64 > MAX_DOCUMENT_BYTES {
        return Read::Skip(TOO_LARGE, None);
    }
    let sha256 = sha256_hex(&bytes);
    if known
        .is_some_and(|known| known.has_document && known.source_sha256.as_deref() == Some(&sha256))
    {
        return Read::Same(sha256);
    }
    // A malformed file must not end the scan for every file after it.
    let Ok(text) = std::panic::catch_unwind(AssertUnwindSafe(|| {
        documents::extract_text_from_bytes(&file.rel_path, &bytes)
    })) else {
        return Read::Skip(EXTRACT_FAILED, Some(sha256));
    };
    let chunks = documents::chunk_document(&text);
    if chunks.is_empty() {
        return Read::Skip(NO_TEXT, Some(sha256));
    }
    Read::Document {
        text,
        sha256,
        chunks,
    }
}

fn record_file(
    conn: &Connection,
    folder_id: &str,
    file: &FoundFile,
    doc_id: &str,
    sha256: Option<&str>,
    status: &str,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "INSERT INTO watched_files (folder_id, rel_path, doc_id, byte_size, modified_ns, source_sha256, status)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(folder_id, rel_path) DO UPDATE SET
           doc_id = excluded.doc_id, byte_size = excluded.byte_size, modified_ns = excluded.modified_ns,
           source_sha256 = excluded.source_sha256, status = excluded.status",
        params![
            folder_id,
            file.rel_path,
            doc_id,
            file.byte_size as i64,
            file.modified_ns,
            sha256,
            status
        ],
    )?;
    Ok(())
}

fn folder_exists(conn: &Connection, folder_id: &str) -> Result<bool, rusqlite::Error> {
    conn.prepare_cached("SELECT 1 FROM watched_folders WHERE id = ?1")?
        .exists(params![folder_id])
}

fn record_scan(
    conn: &Connection,
    folder_id: &str,
    error: Option<&str>,
    changes: Option<ScanChanges>,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "UPDATE watched_folders SET last_scan_at = ?2, last_error = ?3,
           last_changes = COALESCE(?4, last_changes)
         WHERE id = ?1",
        params![
            folder_id,
            now_secs(),
            error,
            changes.and_then(|changes| serde_json::to_string(&changes).ok())
        ],
    )?;
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ScanOutcome {
    /// The folder stopped being watched before or during the scan.
    Gone,
    /// The folder could not be scanned; nothing in the library changed.
    Failed(String),
    Done(ScanChanges),
}

/// Brings a folder's documents up to date with its files. The database lock is
/// held only while writing, never while a file is read or its text extracted.
pub(crate) fn scan_folder(
    conn: &mut Connection,
    folder_id: &str,
    progress: &mut dyn FnMut(ScanProgress),
) -> Result<ScanOutcome, rusqlite::Error> {
    let folder = {
        let _lock = db_lock().lock().unwrap();
        conn.query_row(
            "SELECT path, collection_id FROM watched_folders WHERE id = ?1",
            params![folder_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()?
    };
    let Some((path, collection_id)) = folder else {
        return Ok(ScanOutcome::Gone);
    };
    let walk = match walk(Path::new(&path)) {
        Ok(walk) => walk,
        Err(message) => {
            let _lock = db_lock().lock().unwrap();
            record_scan(conn, folder_id, Some(&message), None)?;
            return Ok(ScanOutcome::Failed(message));
        }
    };
    let mut known = {
        let _lock = db_lock().lock().unwrap();
        known_files(conn, folder_id)?
    };
    let collection = [collection_id];
    let total = walk.files.len();
    let mut changes = ScanChanges::default();
    for (done, file) in walk.files.iter().enumerate() {
        progress(ScanProgress { done, total });
        let previous = known.remove(&file.rel_path);
        if let Some(previous) = &previous {
            let same_stat = previous.byte_size == file.byte_size as i64
                && previous.modified_ns == file.modified_ns;
            if same_stat && previous.has_document {
                changes.unchanged += 1;
                continue;
            }
            if same_stat && previous.status != INDEXED {
                changes.skipped += 1;
                continue;
            }
        }
        let had_document = previous
            .as_ref()
            .is_some_and(|previous| previous.has_document);
        let doc_id = previous
            .as_ref()
            .map(|previous| previous.doc_id.clone())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let read = read_document(file, previous.as_ref());

        let _lock = db_lock().lock().unwrap();
        let tx = conn.transaction()?;
        if !folder_exists(&tx, folder_id)? {
            return Ok(ScanOutcome::Gone);
        }
        match read {
            Read::Same(sha256) => {
                record_file(&tx, folder_id, file, &doc_id, Some(&sha256), INDEXED)?;
                changes.unchanged += 1;
            }
            Read::Document {
                text,
                sha256,
                chunks,
            } => {
                let stored = documents::store_document(
                    &tx,
                    &NewDocument {
                        doc_id: &doc_id,
                        filename: &file.rel_path,
                        text: &text,
                        source_sha256: &sha256,
                        collection_ids: &collection,
                    },
                    &chunks,
                );
                match stored {
                    Ok(()) => {}
                    Err(StoreError::Database(_, error))
                    | Err(StoreError::Collection(CollectionError::Database(error))) => {
                        return Err(error)
                    }
                    // The collection was deleted, and the folder with it.
                    Err(_) => return Ok(ScanOutcome::Gone),
                }
                record_file(&tx, folder_id, file, &doc_id, Some(&sha256), INDEXED)?;
                if had_document {
                    changes.updated += 1;
                } else {
                    changes.added += 1;
                }
            }
            Read::Skip(reason, sha256) => {
                if had_document && documents::remove_document(&tx, &doc_id)? {
                    changes.removed += 1;
                }
                record_file(&tx, folder_id, file, &doc_id, sha256.as_deref(), reason)?;
                changes.skipped += 1;
            }
        }
        tx.commit()?;
    }
    progress(ScanProgress { done: total, total });

    let kept = |rel_path: &str| {
        walk.unreadable.iter().any(|dir| {
            dir.is_empty()
                || rel_path == dir
                || rel_path
                    .strip_prefix(dir.as_str())
                    .is_some_and(|rest| rest.starts_with('/'))
        })
    };
    let note =
        (!walk.unreadable.is_empty()).then(|| {
            let shown = walk
                .unreadable
                .iter()
                .take(5)
                .map(|dir| {
                    if dir.is_empty() {
                        "(the folder itself)"
                    } else {
                        dir.as_str()
                    }
                })
                .collect::<Vec<_>>();
            format!(
            "Some of the folder could not be read, so its documents were kept as they were: {}{}",
            shown.join(", "),
            if walk.unreadable.len() > shown.len() { ", …" } else { "" }
        )
        });
    let _lock = db_lock().lock().unwrap();
    let tx = conn.transaction()?;
    if !folder_exists(&tx, folder_id)? {
        return Ok(ScanOutcome::Gone);
    }
    for (rel_path, gone) in known {
        if kept(&rel_path) {
            continue;
        }
        if documents::remove_document(&tx, &gone.doc_id)? && gone.has_document {
            changes.removed += 1;
        }
        tx.execute(
            "DELETE FROM watched_files WHERE folder_id = ?1 AND rel_path = ?2",
            params![folder_id, rel_path],
        )?;
    }
    record_scan(&tx, folder_id, note.as_deref(), Some(changes))?;
    tx.commit()?;
    Ok(ScanOutcome::Done(changes))
}

#[derive(Default)]
struct Scans {
    queue: Vec<String>,
    current: Option<(String, ScanProgress)>,
}

fn scans() -> &'static Mutex<Scans> {
    static SCANS: OnceLock<Mutex<Scans>> = OnceLock::new();
    SCANS.get_or_init(Mutex::default)
}

static SCANNING: AtomicBool = AtomicBool::new(false);

/// Queues folders for a scan, behind any already queued, and starts the scanner
/// if it is idle. A folder already queued keeps its place.
pub(crate) fn schedule_scans(models_dir: PathBuf, folder_ids: Vec<String>) {
    {
        let mut scans = scans().lock().unwrap();
        for id in folder_ids {
            if !scans.queue.contains(&id) {
                scans.queue.push(id);
            }
        }
    }
    if SCANNING.swap(true, Ordering::AcqRel) {
        return;
    }
    tokio::spawn(async move {
        loop {
            loop {
                let next = {
                    let mut scans = scans().lock().unwrap();
                    if scans.queue.is_empty() {
                        None
                    } else {
                        let id = scans.queue.remove(0);
                        scans.current = Some((id.clone(), ScanProgress { done: 0, total: 0 }));
                        Some(id)
                    }
                };
                let Some(id) = next else {
                    break;
                };
                let scanned = tokio::task::spawn_blocking(move || scan_by_id(&id)).await;
                scans().lock().unwrap().current = None;
                match scanned {
                    Ok(Ok(ScanOutcome::Done(changes))) if changes.added + changes.updated > 0 => {
                        document_vectors::schedule_indexing(models_dir.clone(), true);
                    }
                    Ok(Ok(_)) => {}
                    Ok(Err(error)) => tracing::warn!(%error, "watched folder scan stopped"),
                    Err(error) => tracing::warn!(%error, "watched folder scan task failed"),
                }
            }
            SCANNING.store(false, Ordering::Release);
            // A folder queued after the last check must not wait for the next trigger.
            if scans().lock().unwrap().queue.is_empty() || SCANNING.swap(true, Ordering::AcqRel) {
                return;
            }
        }
    });
}

fn scan_by_id(folder_id: &str) -> Result<ScanOutcome, rusqlite::Error> {
    let mut conn = {
        let _lock = db_lock().lock().unwrap();
        open_connection()?
    };
    let id = folder_id.to_string();
    let outcome = scan_folder(&mut conn, folder_id, &mut |progress| {
        if let Some(current) = scans()
            .lock()
            .unwrap()
            .current
            .as_mut()
            .filter(|current| current.0 == id)
        {
            current.1 = progress;
        }
    });
    if let Err(error) = &outcome {
        let _lock = db_lock().lock().unwrap();
        let _ = record_scan(
            &conn,
            folder_id,
            Some(&format!("The scan stopped: {error}")),
            None,
        );
    }
    outcome
}

/// Re-scans every watched folder every `POLL_INTERVAL`, starting now, so a
/// change made while the server was stopped is picked up when it starts.
pub(crate) fn start_polling(models_dir: PathBuf) {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(POLL_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            let ids = tokio::task::spawn_blocking(|| {
                let _lock = db_lock().lock().unwrap();
                let conn = open_connection()?;
                let mut statement =
                    conn.prepare("SELECT id FROM watched_folders ORDER BY created_at, id")?;
                let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
                rows.collect::<Result<Vec<_>, rusqlite::Error>>()
            })
            .await;
            let ids = match ids {
                Ok(Ok(ids)) => ids,
                Ok(Err(error)) => {
                    tracing::warn!(%error, "could not list watched folders");
                    continue;
                }
                Err(_) => continue,
            };
            // The folder being scanned now would only be scanned again at once.
            let ids = {
                let scans = scans().lock().unwrap();
                ids.into_iter()
                    .filter(|id| {
                        scans
                            .current
                            .as_ref()
                            .is_none_or(|current| &current.0 != id)
                    })
                    .collect::<Vec<_>>()
            };
            if !ids.is_empty() {
                schedule_scans(models_dir.clone(), ids);
            }
        }
    });
}

fn with_connection<T>(
    work: impl FnOnce(&mut Connection) -> Result<T, FolderError>,
) -> Result<T, FolderError> {
    let _lock = db_lock().lock().unwrap();
    let mut conn = open_connection()?;
    work(&mut conn)
}

/// Endpoint: `GET /api/folders`
pub async fn list_watched_folders(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<FolderView>>, Response> {
    if !local_access(state.serve_addr, &headers) {
        return Err(FolderError::Forbidden.into_response());
    }
    with_connection(|conn| Ok(list_folders(conn)?))
        .map(Json)
        .map_err(FolderError::into_response)
}

/// Endpoint: `POST /api/folders`
pub async fn watch_folder(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<WatchFolderRequest>,
) -> Result<impl IntoResponse, Response> {
    if !local_access(state.serve_addr, &headers) {
        return Err(FolderError::Forbidden.into_response());
    }
    let canonical = canonical_folder(&payload.path).map_err(FolderError::into_response)?;
    let id = with_connection(|conn| add_folder(conn, &canonical, &payload.collection_id))
        .map_err(FolderError::into_response)?;
    schedule_scans(state.models_dir.clone(), vec![id.clone()]);
    with_connection(|conn| folder_view(conn, &id))
        .map(|folder| (StatusCode::CREATED, Json(folder)))
        .map_err(FolderError::into_response)
}

/// Endpoint: `POST /api/folders/:id/scan`
pub async fn scan_watched_folder(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<impl IntoResponse, Response> {
    if !local_access(state.serve_addr, &headers) {
        return Err(FolderError::Forbidden.into_response());
    }
    with_connection(|conn| folder_view(conn, &id)).map_err(FolderError::into_response)?;
    schedule_scans(state.models_dir.clone(), vec![id.clone()]);
    with_connection(|conn| folder_view(conn, &id))
        .map(|folder| (StatusCode::ACCEPTED, Json(folder)))
        .map_err(FolderError::into_response)
}

/// Endpoint: `DELETE /api/folders/:id`
pub async fn unwatch_folder(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, Response> {
    if !local_access(state.serve_addr, &headers) {
        return Err(FolderError::Forbidden.into_response());
    }
    scans().lock().unwrap().queue.retain(|queued| queued != &id);
    with_connection(|conn| remove_folder(conn, &id))
        .map(|_| StatusCode::NO_CONTENT)
        .map_err(FolderError::into_response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::documents::init_db;

    struct Library {
        conn: Connection,
        collection: String,
        dir: tempfile::TempDir,
    }

    impl Library {
        fn new() -> Self {
            let conn = Connection::open_in_memory().unwrap();
            conn.execute("PRAGMA foreign_keys = ON;", []).unwrap();
            init_db(&conn).unwrap();
            let collection = document_collections::create(&conn, "Watched").unwrap().id;
            Self {
                conn,
                collection,
                dir: tempfile::tempdir().unwrap(),
            }
        }

        fn root(&self) -> PathBuf {
            std::fs::canonicalize(self.dir.path()).unwrap()
        }

        fn write(&self, rel_path: &str, text: &str) {
            let path = self.dir.path().join(rel_path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }

        fn watch(&self) -> String {
            add_folder(&self.conn, &self.root(), &self.collection).unwrap()
        }

        fn scan(&mut self, id: &str) -> ScanOutcome {
            scan_folder(&mut self.conn, id, &mut |_| {}).unwrap()
        }

        /// `(filename, doc id, source text)` of every document, by filename.
        fn documents(&self) -> Vec<(String, String, String)> {
            let mut statement = self
                .conn
                .prepare("SELECT filename, id, source_text FROM documents ORDER BY filename")
                .unwrap();
            let rows = statement
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
                .unwrap();
            rows.collect::<Result<Vec<_>, _>>().unwrap()
        }

        fn filenames(&self) -> Vec<String> {
            self.documents()
                .into_iter()
                .map(|(name, _, _)| name)
                .collect()
        }

        fn members(&self) -> usize {
            document_collections::get(&self.conn, &self.collection)
                .unwrap()
                .doc_ids
                .len()
        }

        fn view(&self, id: &str) -> FolderView {
            folder_view(&self.conn, id).unwrap()
        }
    }

    fn done(outcome: ScanOutcome) -> ScanChanges {
        match outcome {
            ScanOutcome::Done(changes) => changes,
            other => panic!("expected a finished scan, got {other:?}"),
        }
    }

    #[test]
    fn a_scan_reads_documents_and_skips_hidden_entries_and_other_types() {
        let mut library = Library::new();
        library.write("guide.md", "# Guide\n\nRefunds take five business days.");
        library.write("notes/today.txt", "The courier leaves at dawn.");
        library.write(".draft.md", "A hidden draft.");
        library.write(".git/HEAD.md", "Inside a hidden folder.");
        library.write("photo.png", "not a document type");
        library.write("empty.txt", "   ");
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            library.dir.path().join("guide.md"),
            library.dir.path().join("linked.md"),
        )
        .unwrap();
        let id = library.watch();

        let changes = done(library.scan(&id));

        assert_eq!(
            changes,
            ScanChanges {
                added: 2,
                skipped: 1,
                ..ScanChanges::default()
            }
        );
        assert_eq!(library.filenames(), ["guide.md", "notes/today.txt"]);
        assert_eq!(library.members(), 2, "both documents join the collection");
        let view = library.view(&id);
        assert_eq!((view.documents, view.skipped_count), (2, 1));
        assert_eq!(
            view.skipped,
            [SkippedFile {
                path: "empty.txt".to_string(),
                reason: NO_TEXT.to_string()
            }]
        );
        assert_eq!(view.last_changes, Some(changes));
        assert!(view.last_scan_at.is_some() && view.last_error.is_none());

        assert_eq!(
            done(library.scan(&id)),
            ScanChanges {
                unchanged: 2,
                skipped: 1,
                ..ScanChanges::default()
            },
            "every file found is counted once"
        );
    }

    #[test]
    fn a_rescan_updates_changed_files_in_place_and_removes_deleted_ones() {
        let mut library = Library::new();
        library.write("policy.md", "Refunds take five business days.");
        library.write("old.txt", "This file will be deleted.");
        library.write("same.txt", "This file does not change.");
        let id = library.watch();
        done(library.scan(&id));
        let before = library.documents();

        library.write("policy.md", "Refunds now take ten business days, not five.");
        std::fs::remove_file(library.dir.path().join("old.txt")).unwrap();
        library.write("new.md", "A file added after the first scan.");
        let changes = done(library.scan(&id));

        assert_eq!(
            changes,
            ScanChanges {
                added: 1,
                updated: 1,
                removed: 1,
                unchanged: 1,
                skipped: 0,
            }
        );
        let after = library.documents();
        assert_eq!(
            after.iter().map(|doc| doc.0.as_str()).collect::<Vec<_>>(),
            ["new.md", "policy.md", "same.txt"]
        );
        let policy = |docs: &[(String, String, String)]| {
            docs.iter()
                .find(|doc| doc.0 == "policy.md")
                .unwrap()
                .clone()
        };
        assert_eq!(
            policy(&after).1,
            policy(&before).1,
            "re-indexed under the same id"
        );
        assert!(policy(&after).2.contains("ten business days"));
        assert_eq!(library.members(), 3);

        assert_eq!(
            done(library.scan(&id)),
            ScanChanges {
                unchanged: 3,
                ..ScanChanges::default()
            }
        );
    }

    #[test]
    fn a_new_time_on_unchanged_content_reindexes_nothing() {
        let mut library = Library::new();
        library.write("policy.md", "Refunds take five business days.");
        let id = library.watch();
        done(library.scan(&id));
        let path = library.dir.path().join("policy.md");
        let later = std::fs::metadata(&path).unwrap().modified().unwrap() + Duration::from_secs(60);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(later)
            .unwrap();

        assert_eq!(
            done(library.scan(&id)),
            ScanChanges {
                unchanged: 1,
                ..ScanChanges::default()
            }
        );
        let recorded: i64 = library
            .conn
            .query_row("SELECT modified_ns FROM watched_files", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(recorded, modified_ns(&std::fs::metadata(&path).unwrap()));
    }

    #[test]
    fn a_document_deleted_from_the_library_returns_on_the_next_scan() {
        let mut library = Library::new();
        library.write("policy.md", "Refunds take five business days.");
        let id = library.watch();
        done(library.scan(&id));
        let doc_id = library.documents()[0].1.clone();
        assert!(documents::remove_document(&library.conn, &doc_id).unwrap());

        assert_eq!(
            done(library.scan(&id)),
            ScanChanges {
                added: 1,
                ..ScanChanges::default()
            }
        );
        assert_eq!(library.documents()[0].1, doc_id);
        assert_eq!(library.members(), 1);
    }

    #[test]
    fn oversized_and_unparseable_files_are_skipped_without_ending_the_scan() {
        let mut library = Library::new();
        library.write("a-broken.pdf", "%PDF-1.7 this is not really a PDF");
        let big = std::fs::File::create(library.dir.path().join("b-big.txt")).unwrap();
        big.set_len(MAX_DOCUMENT_BYTES + 1).unwrap();
        library.write("c-fine.md", "Read after the two bad files.");
        let id = library.watch();

        let changes = done(library.scan(&id));

        assert_eq!((changes.added, changes.skipped), (1, 2));
        assert_eq!(library.filenames(), ["c-fine.md"]);
        let reasons = library
            .view(&id)
            .skipped
            .into_iter()
            .map(|skip| (skip.path, skip.reason))
            .collect::<Vec<_>>();
        assert_eq!(reasons[0].0, "a-broken.pdf");
        assert!(
            [NO_TEXT, EXTRACT_FAILED].contains(&reasons[0].1.as_str()),
            "{reasons:?}"
        );
        assert_eq!(reasons[1], ("b-big.txt".to_string(), TOO_LARGE.to_string()));
    }

    #[test]
    fn a_file_that_turns_unreadable_loses_its_stale_document() {
        let mut library = Library::new();
        library.write("notes.txt", "Readable at first.");
        let id = library.watch();
        done(library.scan(&id));
        library.write("notes.txt", "  ");

        assert_eq!(
            done(library.scan(&id)),
            ScanChanges {
                removed: 1,
                skipped: 1,
                ..ScanChanges::default()
            }
        );
        assert!(library.documents().is_empty());
    }

    #[test]
    fn a_folder_that_cannot_be_opened_changes_nothing() {
        let mut library = Library::new();
        library.write("policy.md", "Refunds take five business days.");
        let id = library.watch();
        done(library.scan(&id));
        let moved = library.dir.path().with_extension("moved");
        std::fs::rename(library.dir.path(), &moved).unwrap();

        let outcome = library.scan(&id);
        std::fs::rename(&moved, library.dir.path()).unwrap();

        assert!(matches!(outcome, ScanOutcome::Failed(_)), "{outcome:?}");
        assert_eq!(library.filenames(), ["policy.md"]);
        assert!(library
            .view(&id)
            .last_error
            .unwrap()
            .contains("cannot be opened"));
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_subfolder_keeps_its_documents() {
        use std::os::unix::fs::PermissionsExt;

        let mut library = Library::new();
        library.write("top.md", "At the top.");
        library.write(
            "locked/inner.md",
            "Inside a folder that becomes unreadable.",
        );
        let id = library.watch();
        done(library.scan(&id));
        let locked = library.dir.path().join("locked");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        if std::fs::read_dir(&locked).is_ok() {
            // Running as root: permissions cannot make the folder unreadable.
            std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
            return;
        }

        let changes = done(library.scan(&id));
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();

        assert_eq!(
            changes,
            ScanChanges {
                unchanged: 1,
                ..ScanChanges::default()
            }
        );
        assert_eq!(library.filenames(), ["locked/inner.md", "top.md"]);
        assert!(library.view(&id).last_error.unwrap().contains("locked"));
    }

    #[test]
    fn folders_inside_around_or_equal_to_a_watched_one_are_refused() {
        let library = Library::new();
        library.write("watched/inner/a.md", "a");
        library.write("sibling/b.md", "b");
        let outer = library.root();
        let watched = outer.join("watched");
        add_folder(&library.conn, &watched, &library.collection).unwrap();

        for path in [watched.clone(), watched.join("inner"), outer.clone()] {
            let refused = add_folder(&library.conn, &path, &library.collection);
            assert!(
                matches!(refused, Err(FolderError::Overlaps(_))),
                "{path:?}: {refused:?}"
            );
        }
        add_folder(&library.conn, &outer.join("sibling"), &library.collection).unwrap();
        assert!(matches!(
            add_folder(
                &library.conn,
                &outer.join("elsewhere"),
                "no-such-collection"
            ),
            Err(FolderError::Collection(
                CollectionError::CollectionNotFound(_)
            ))
        ));
    }

    #[test]
    fn a_watched_folder_is_an_existing_absolute_directory_below_the_root() {
        let library = Library::new();
        library.write("file.md", "a file, not a folder");
        let invalid = |raw: &str| matches!(canonical_folder(raw), Err(FolderError::InvalidPath(_)));

        assert!(invalid(""));
        assert!(invalid("relative/folder"));
        assert!(invalid(
            library.dir.path().join("missing").to_str().unwrap()
        ));
        assert!(invalid(
            library.dir.path().join("file.md").to_str().unwrap()
        ));
        #[cfg(unix)]
        assert!(invalid("/"));
        let padded = format!("  {}  ", library.dir.path().display());
        assert_eq!(canonical_folder(&padded).unwrap(), library.root());
    }

    #[test]
    fn removing_a_folder_removes_its_documents_and_leaves_the_files() {
        let mut library = Library::new();
        library.write("policy.md", "Refunds take five business days.");
        let id = library.watch();
        done(library.scan(&id));
        let tx = library.conn.transaction().unwrap();
        documents::store_document(
            &tx,
            &NewDocument {
                doc_id: "uploaded",
                filename: "uploaded.md",
                text: "Uploaded by hand.",
                source_sha256: "0",
                collection_ids: &[library.collection.clone()],
            },
            &documents::chunk_document("Uploaded by hand."),
        )
        .unwrap();
        tx.commit().unwrap();

        assert_eq!(remove_folder(&mut library.conn, &id).unwrap(), 1);
        assert_eq!(library.filenames(), ["uploaded.md"]);
        assert!(library.dir.path().join("policy.md").exists());
        assert!(matches!(library.scan(&id), ScanOutcome::Gone));
        assert!(matches!(
            remove_folder(&mut library.conn, &id),
            Err(FolderError::FolderNotFound(_))
        ));
    }

    #[test]
    fn deleting_the_collection_stops_the_watch_and_keeps_the_documents() {
        let mut library = Library::new();
        library.write("policy.md", "Refunds take five business days.");
        let id = library.watch();
        done(library.scan(&id));

        document_collections::delete(&library.conn, &library.collection).unwrap();

        assert!(list_folders(&library.conn).unwrap().is_empty());
        let files: i64 = library
            .conn
            .query_row("SELECT COUNT(*) FROM watched_files", [], |row| row.get(0))
            .unwrap();
        assert_eq!(files, 0);
        assert_eq!(library.filenames(), ["policy.md"]);
        assert!(matches!(library.scan(&id), ScanOutcome::Gone));
    }

    #[test]
    fn folder_routes_answer_only_the_local_web_ui_on_loopback() {
        let loopback = SocketAddr::from(([127, 0, 0, 1], 8181));
        let headers = |pairs: &[(&'static str, &str)]| {
            let mut map = HeaderMap::new();
            for (name, value) in pairs {
                map.insert(*name, value.parse().unwrap());
            }
            map
        };
        let web_ui = headers(&[
            ("host", "127.0.0.1:8181"),
            ("origin", "http://127.0.0.1:8181"),
        ]);

        assert!(local_access(loopback, &web_ui));
        assert!(local_access(
            loopback,
            &headers(&[
                ("host", "localhost:8181"),
                ("sec-fetch-site", "same-origin")
            ])
        ));
        assert!(!local_access(
            loopback,
            &headers(&[("host", "127.0.0.1:8181")])
        ));
        assert!(!local_access(
            loopback,
            &headers(&[
                ("host", "127.0.0.1:8181"),
                ("origin", "http://evil.example")
            ])
        ));
        assert!(!local_access(
            loopback,
            &headers(&[("host", "evil.example"), ("origin", "http://evil.example")])
        ));
        assert!(!local_access(
            SocketAddr::from(([0, 0, 0, 0], 8181)),
            &web_ui
        ));
    }
}
