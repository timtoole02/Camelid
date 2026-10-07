//! Local Document Drag-and-Drop RAG (Feature A)
//!
//! Provides document ingestion, exact-span chunking, SQLite FTS5 keyword
//! indexing, and hybrid retrieval: BM25 and cosine similarity over stored chunk
//! vectors (see `document_vectors`), fused by reciprocal rank. A search covers
//! the whole library, named documents, or named collections
//! (see `document_collections`).

mod storage;

use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Read};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rusqlite::{params, params_from_iter, types::Value, Connection};
use serde::{Deserialize, Serialize};

use super::citations::{chunk_text_with_spans, retain_verifiable, sha256_hex, ChunkSpan};
use super::document_collections::{self, CollectionError};
use super::document_folders;
use super::document_vectors::{self, Coverage, Unavailable};
use super::{api_error, AppState};

const DEFAULT_CHUNK_CHARS: usize = 512;
const DEFAULT_CHUNK_OVERLAP: usize = 64;
/// Candidates each ranker contributes before fusion and citation checks.
const CANDIDATE_POOL: usize = 50;
/// The usual reciprocal-rank-fusion constant; it damps the weight of the very top ranks.
const RRF_K: f64 = 60.0;

static DB_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) fn db_lock() -> &'static Mutex<()> {
    DB_MUTEX.get_or_init(|| Mutex::new(()))
}

pub(crate) fn init_db(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
            content,
            tokenize='porter unicode61'
        );",
    )?;
    ensure_citation_columns(conn)?;
    document_vectors::init_schema(conn)?;
    document_collections::init_schema(conn)?;
    document_folders::init_schema(conn)?;
    Ok(())
}

/// Adds the citation-binding columns to databases created before F2a. The table
/// and column names are compile-time constants, never caller input.
fn ensure_citation_columns(conn: &Connection) -> Result<(), rusqlite::Error> {
    for (table, column, decl) in [
        ("documents", "source_sha256", "TEXT"),
        ("documents", "text_sha256", "TEXT"),
        ("documents", "source_text", "TEXT"),
        ("document_chunks", "byte_start", "INTEGER"),
        ("document_chunks", "byte_end", "INTEGER"),
        ("document_chunks", "chunk_sha256", "TEXT"),
    ] {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let existing: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>, _>>()?;
        if !existing.iter().any(|name| name == column) {
            conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl};"))?;
        }
    }
    Ok(())
}

pub(super) fn library_exists() -> bool {
    storage::library_exists()
}

pub(crate) fn open_connection() -> Result<Connection, rusqlite::Error> {
    let path = storage::prepare_path()?;
    let conn = Connection::open(&path)?;
    conn.execute("PRAGMA foreign_keys = ON;", [])?;
    init_db(&conn)?;
    Ok(conn)
}

/// The file types `extract_text_from_bytes` reads, and the only ones a watched
/// folder picks up. An upload of any other type is read as lossy UTF-8.
pub(crate) const DOCUMENT_EXTENSIONS: &[&str] =
    &["txt", "md", "csv", "json", "rs", "py", "js", "docx", "pdf"];

pub(crate) fn has_document_extension(filename: &str) -> bool {
    filename.rsplit_once('.').is_some_and(|(_, extension)| {
        DOCUMENT_EXTENSIONS
            .iter()
            .any(|known| extension.eq_ignore_ascii_case(known))
    })
}

/// Extract clean textual tokens from supported document formats.
pub fn extract_text_from_bytes(filename: &str, raw_bytes: &[u8]) -> String {
    let lower = filename.to_lowercase();
    if lower.ends_with(".txt")
        || lower.ends_with(".md")
        || lower.ends_with(".csv")
        || lower.ends_with(".json")
        || lower.ends_with(".rs")
        || lower.ends_with(".py")
        || lower.ends_with(".js")
    {
        return String::from_utf8_lossy(raw_bytes).to_string();
    }

    // DOCX is a ZIP container. Read the actual XML entry so deflated documents
    // work as well as the uncommon uncompressed form.
    if lower.ends_with(".docx") {
        if let Ok(archive) = zip_extract_text(raw_bytes) {
            if !archive.is_empty() {
                return archive;
            }
        }
        return String::new();
    }

    // Let a PDF parser handle compressed streams, encodings, and font maps.
    if lower.ends_with(".pdf") {
        let extracted = pdf_extract_text(raw_bytes);
        if !extracted.is_empty() {
            return extracted;
        }
        return String::new();
    }

    // Fallback: extract readable UTF-8 strings
    String::from_utf8_lossy(raw_bytes).to_string()
}

fn zip_extract_text(bytes: &[u8]) -> Result<String, ()> {
    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|_| ())?;
    let mut document = archive.by_name("word/document.xml").map_err(|_| ())?;
    let mut xml = String::new();
    document.read_to_string(&mut xml).map_err(|_| ())?;
    Ok(strip_xml_tags(xml.as_bytes()))
}

fn strip_xml_tags(bytes: &[u8]) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    let s = String::from_utf8_lossy(bytes);
    for ch in s.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
            out.push(' ');
        } else if !in_tag {
            out.push(ch);
        }
    }
    decode_xml_entities(&out)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn decode_xml_entities(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn pdf_extract_text(bytes: &[u8]) -> String {
    pdf_extract::extract_text_from_mem(bytes)
        .unwrap_or_default()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Debug, Deserialize)]
pub struct IngestDocumentRequest {
    pub doc_id: Option<String>,
    pub filename: String,
    pub content: String,
    #[serde(default)]
    pub is_base64: bool,
    /// Collections the document joins; an unknown id ingests nothing.
    #[serde(default)]
    pub collection_ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct IngestDocumentResponse {
    pub doc_id: String,
    pub filename: String,
    pub chunk_count: usize,
    pub byte_size: usize,
}

#[derive(Debug, Deserialize)]
pub struct SearchDocumentsRequest {
    pub query: String,
    #[serde(default)]
    pub doc_ids: Option<Vec<String>>,
    /// Searches every member of these collections as well as `doc_ids`.
    #[serde(default)]
    pub collection_ids: Option<Vec<String>>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub mode: SearchMode,
    /// Search the whole library. Passages from outside `doc_ids` and
    /// `collection_ids` count only when similar enough to the query, so this
    /// needs the encoder.
    #[serde(default)]
    pub library: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Hybrid when the encoder is available, keyword alone otherwise.
    #[default]
    Auto,
    Keyword,
    /// Vector similarity alone. Refused when the encoder is unavailable.
    Semantic,
    /// Refused when the encoder is unavailable, instead of silently degrading.
    Hybrid,
}

fn default_top_k() -> usize {
    5
}

#[derive(Debug, Serialize, Clone)]
pub struct DocumentSearchResult {
    pub doc_id: String,
    pub filename: String,
    pub chunk_index: usize,
    pub excerpt: String,
    /// Comparable only within one response: normalized BM25 for `keyword`,
    /// cosine for `semantic`, and the fused reciprocal-rank score otherwise.
    pub score: f32,
    /// Byte range of `excerpt` within the document's canonical text plus the
    /// hashes that bind it there. `None` for rows ingested before F2a.
    pub byte_start: Option<usize>,
    pub byte_end: Option<usize>,
    pub chunk_sha256: Option<String>,
    pub doc_sha256: Option<String>,
    /// `keyword` for a BM25 hit only, `semantic` for a vector hit only,
    /// `hybrid` when both rankers found it, and `attached` when an explicitly
    /// attached document is supplied as context because nothing matched.
    pub retrieval: &'static str,
    /// Cosine similarity to the query, reported by library-wide searches.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct SemanticSummary {
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    /// Index coverage of the searched scope, so a partly indexed library
    /// shows. Present only when the search ranked by meaning.
    #[serde(flatten)]
    pub coverage: Option<Coverage>,
}

#[derive(Debug, Serialize)]
pub struct RetrievalSummary {
    /// What ranked the results: `hybrid`, `keyword`, `semantic`, `attached`,
    /// or `none` when nothing was searched.
    pub mode: &'static str,
    pub semantic: SemanticSummary,
    /// The similarity floor a library-wide search held passages from outside
    /// its documents and collections to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relevance_floor: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct SearchDocumentsResponse {
    pub results: Vec<DocumentSearchResult>,
    pub retrieval: RetrievalSummary,
}

#[derive(Debug, Serialize)]
pub struct DocumentMetaView {
    pub id: String,
    pub filename: String,
    pub file_type: String,
    pub byte_size: i64,
    pub chunk_count: i64,
    pub created_at: i64,
}

/// Endpoint: `POST /api/documents/ingest`
pub async fn ingest_document(
    State(state): State<AppState>,
    Json(payload): Json<IngestDocumentRequest>,
) -> Result<Json<IngestDocumentResponse>, Response> {
    let doc_id = payload
        .doc_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let filename = payload.filename.trim().to_string();

    let (text_content, source_sha256) = if payload.is_base64 {
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(payload.content.trim())
            .map_err(|e| {
                api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_base64",
                    format!("Failed to decode base64 document content: {e}"),
                    None,
                )
            })?;
        let digest = sha256_hex(&decoded);
        (extract_text_from_bytes(&filename, &decoded), digest)
    } else {
        let digest = sha256_hex(payload.content.as_bytes());
        (payload.content, digest)
    };

    let chunks = chunk_document(&text_content);
    if chunks.is_empty() {
        return Err(StoreError::Empty.into_response());
    }
    let byte_size = text_content.len();
    let chunk_count = chunks.len();

    let _lock = db_lock().lock().unwrap();
    let mut conn = open_connection().map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sqlite_open_error",
            format!("Could not open RAG database: {e}"),
            None,
        )
    })?;

    let tx = conn.transaction().map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sqlite_tx_error",
            format!("Database transaction failed: {e}"),
            None,
        )
    })?;
    store_document(
        &tx,
        &NewDocument {
            doc_id: &doc_id,
            filename: &filename,
            text: &text_content,
            source_sha256: &source_sha256,
            collection_ids: &payload.collection_ids,
        },
        &chunks,
    )
    .map_err(StoreError::into_response)?;
    tx.commit().map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "commit_error",
            e.to_string(),
            None,
        )
    })?;
    document_vectors::schedule_indexing(state.models_dir.clone(), true);

    Ok(Json(IngestDocumentResponse {
        doc_id,
        filename,
        chunk_count,
        byte_size,
    }))
}

/// A document ready to store: its canonical text and the hash of the bytes it
/// was read from.
pub(crate) struct NewDocument<'a> {
    pub doc_id: &'a str,
    pub filename: &'a str,
    pub text: &'a str,
    pub source_sha256: &'a str,
    /// Collections the document joins; they must exist.
    pub collection_ids: &'a [String],
}

/// Why a document was not stored. A database error carries the API code it is
/// reported with.
#[derive(Debug)]
pub(crate) enum StoreError {
    Empty,
    Collection(CollectionError),
    Database(&'static str, rusqlite::Error),
}

impl StoreError {
    pub(crate) fn into_response(self) -> Response {
        match self {
            Self::Empty => api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "empty_document",
                "The document did not contain any readable text.".to_string(),
                None,
            ),
            Self::Collection(error) => error.into_response(),
            Self::Database(code, error) => api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                code,
                error.to_string(),
                None,
            ),
        }
    }
}

/// The library's chunks of a document's text.
pub(crate) fn chunk_document(text: &str) -> Vec<ChunkSpan> {
    chunk_text_with_spans(text, DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP)
}

/// Stores `doc` as `chunks` within the caller's transaction. An earlier version
/// under the same id is replaced in place, so its collection memberships
/// survive.
pub(crate) fn store_document(
    conn: &Connection,
    doc: &NewDocument<'_>,
    chunks: &[ChunkSpan],
) -> Result<(), StoreError> {
    if chunks.is_empty() {
        return Err(StoreError::Empty);
    }
    let database = |code: &'static str| move |error| StoreError::Database(code, error);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    document_collections::require_collections(conn, doc.collection_ids)
        .map_err(StoreError::Collection)?;

    let ext = doc
        .filename
        .split('.')
        .next_back()
        .unwrap_or("txt")
        .to_lowercase();

    // Remove any previous chunks and their external-content FTS rows for doc_id.
    conn.execute(
        "DELETE FROM document_chunks_fts WHERE rowid IN (SELECT id FROM document_chunks WHERE doc_id = ?1)",
        params![doc.doc_id],
    )
    .map_err(database("delete_fts_error"))?;
    conn.execute(
        "DELETE FROM document_chunks WHERE doc_id = ?1",
        params![doc.doc_id],
    )
    .map_err(database("delete_chunk_error"))?;

    // Update an existing row in place: INSERT OR REPLACE deletes it first, and
    // that delete would cascade to the document's collection memberships.
    conn.execute(
        "INSERT INTO documents
         (id, filename, file_type, byte_size, chunk_count, created_at, source_sha256, text_sha256, source_text)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
         ON CONFLICT(id) DO UPDATE SET
           filename = excluded.filename, file_type = excluded.file_type,
           byte_size = excluded.byte_size, chunk_count = excluded.chunk_count,
           created_at = excluded.created_at, source_sha256 = excluded.source_sha256,
           text_sha256 = excluded.text_sha256, source_text = excluded.source_text",
        params![
            doc.doc_id,
            doc.filename,
            ext,
            doc.text.len() as i64,
            chunks.len() as i64,
            now,
            doc.source_sha256,
            sha256_hex(doc.text.as_bytes()),
            doc.text
        ],
    )
    .map_err(database("insert_doc_error"))?;
    document_collections::add_memberships(conn, doc.collection_ids, doc.doc_id)
        .map_err(database("collection_membership_error"))?;

    // Insert chunks and index in FTS5
    for (idx, chunk) in chunks.iter().enumerate() {
        conn.execute(
            "INSERT INTO document_chunks
             (doc_id, chunk_index, content, byte_start, byte_end, chunk_sha256)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                doc.doc_id,
                idx as i64,
                chunk.text,
                chunk.start as i64,
                chunk.end as i64,
                chunk.sha256()
            ],
        )
        .map_err(database("insert_chunk_error"))?;

        let rowid = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO document_chunks_fts (rowid, content) VALUES (?1, ?2)",
            params![rowid, chunk.text],
        )
        .map_err(database("insert_fts_error"))?;
    }
    Ok(())
}

/// Deletes a document with its chunks and their keyword-index rows; its vectors
/// and collection memberships go with it by cascade. Returns whether it existed.
pub(crate) fn remove_document(conn: &Connection, doc_id: &str) -> Result<bool, rusqlite::Error> {
    conn.execute(
        "DELETE FROM document_chunks_fts WHERE rowid IN (SELECT id FROM document_chunks WHERE doc_id = ?1)",
        params![doc_id],
    )?;
    Ok(conn.execute("DELETE FROM documents WHERE id = ?1", params![doc_id])? > 0)
}

/// Sanitizes a text query into FTS5-safe query string.
fn sanitize_fts5_query(query: &str) -> String {
    let tokens: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty() && w.len() > 1)
        .map(|w| format!("\"{}\"", w))
        .collect();

    if tokens.is_empty() {
        return "\"\"".to_string();
    }

    tokens.join(" OR ")
}

/// When the user explicitly attached documents, a zero-hit lexical search must
/// not silently turn into a plain model request. Return leading chunks in
/// attachment order, round-robin across files, so every attached document gets
/// a chance to contribute before later chunks consume the small context limit.
fn attached_document_context(
    conn: &Connection,
    doc_ids: &[String],
    top_k: usize,
) -> Result<Vec<DocumentSearchResult>, rusqlite::Error> {
    let mut statement = conn.prepare(
        "SELECT c.doc_id, d.filename, c.chunk_index, c.content,
                c.byte_start, c.byte_end, c.chunk_sha256, d.text_sha256
         FROM document_chunks AS c
         JOIN documents AS d ON d.id = c.doc_id
         WHERE c.doc_id = ?1
         ORDER BY c.chunk_index ASC
         LIMIT ?2",
    )?;
    let mut chunks_by_document = Vec::new();

    // At most `top_k` documents can contribute to a `top_k` response. This
    // also bounds work for a malformed request containing thousands of ids.
    for doc_id in doc_ids.iter().take(top_k) {
        let chunks = statement
            .query_map(params![doc_id, top_k as i64], |row| {
                Ok(DocumentSearchResult {
                    doc_id: row.get(0)?,
                    filename: row.get(1)?,
                    chunk_index: row.get::<_, i64>(2)? as usize,
                    excerpt: row.get(3)?,
                    score: 0.0,
                    byte_start: row.get::<_, Option<i64>>(4)?.map(|value| value as usize),
                    byte_end: row.get::<_, Option<i64>>(5)?.map(|value| value as usize),
                    chunk_sha256: row.get(6)?,
                    doc_sha256: row.get(7)?,
                    retrieval: "attached",
                    similarity: None,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        chunks_by_document.push(chunks);
    }

    let mut results = Vec::new();
    let mut chunk_index = 0;
    while results.len() < top_k {
        let mut added = false;
        for chunks in &chunks_by_document {
            if let Some(chunk) = chunks.get(chunk_index) {
                results.push(chunk.clone());
                added = true;
                if results.len() == top_k {
                    break;
                }
            }
        }
        if !added {
            break;
        }
        chunk_index += 1;
    }

    Ok(results)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Found {
    Keyword,
    Semantic,
    Both,
}

impl Found {
    fn label(self) -> &'static str {
        match self {
            Self::Keyword => "keyword",
            Self::Semantic => "semantic",
            Self::Both => "hybrid",
        }
    }
}

/// Reciprocal rank fusion: each ranking contributes `1 / (RRF_K + rank)` for
/// the chunks it holds. Ties break on the better single-ranking position, then
/// chunk id, so the order is deterministic.
fn reciprocal_rank_fusion(keyword: &[i64], semantic: &[i64]) -> Vec<(i64, f32, Found)> {
    let mut fused: HashMap<i64, (f64, usize, bool, bool)> = HashMap::new();
    for (list, ids) in [(0, keyword), (1, semantic)] {
        for (rank, id) in ids.iter().enumerate() {
            let entry = fused.entry(*id).or_insert((0.0, usize::MAX, false, false));
            entry.0 += 1.0 / (RRF_K + (rank + 1) as f64);
            entry.1 = entry.1.min(rank);
            if list == 0 {
                entry.2 = true;
            } else {
                entry.3 = true;
            }
        }
    }
    let mut ranked = fused
        .into_iter()
        .map(|(id, (score, best, in_keyword, in_semantic))| {
            let found = match (in_keyword, in_semantic) {
                (true, true) => Found::Both,
                (true, false) => Found::Keyword,
                _ => Found::Semantic,
            };
            (id, score, best, found)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then(left.2.cmp(&right.2))
            .then(left.0.cmp(&right.0))
    });
    ranked
        .into_iter()
        .map(|(id, score, _, found)| (id, score as f32, found))
        .collect()
}

/// Chunks in scope ranked by BM25, best first, as `(chunk id, raw bm25)`.
/// The scope is applied before the limit so an attached document is never
/// crowded out by better matches elsewhere in the library.
fn keyword_ranked(
    conn: &Connection,
    query: &str,
    doc_ids: Option<&[String]>,
    limit: usize,
) -> Result<Vec<(i64, f64)>, rusqlite::Error> {
    let mut sql = String::from(
        "SELECT c.id, bm25(document_chunks_fts) AS score
         FROM document_chunks_fts AS f
         JOIN document_chunks AS c ON c.id = f.rowid
         WHERE document_chunks_fts MATCH ?1",
    );
    let mut bind_values = vec![Value::Text(sanitize_fts5_query(query))];
    if let Some(allowed_ids) = doc_ids {
        let placeholders = allowed_ids
            .iter()
            .map(|doc_id| {
                bind_values.push(Value::Text(doc_id.clone()));
                format!("?{}", bind_values.len())
            })
            .collect::<Vec<_>>()
            .join(", ");
        sql.push_str(&format!(" AND c.doc_id IN ({placeholders})"));
    }
    bind_values.push(Value::Integer(limit as i64));
    sql.push_str(&format!(
        " ORDER BY score ASC, c.id ASC LIMIT ?{}",
        bind_values.len()
    ));
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(bind_values.iter()), |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
    })?;
    rows.collect()
}

/// Loads the ranked chunks in rank order. A chunk whose document row is gone
/// drops out here.
fn fetch_results(
    conn: &Connection,
    ranked: &[(i64, f32, &'static str)],
) -> Result<Vec<DocumentSearchResult>, rusqlite::Error> {
    if ranked.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = (1..=ranked.len())
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT c.id, c.doc_id, d.filename, c.chunk_index, c.content,
                c.byte_start, c.byte_end, c.chunk_sha256, d.text_sha256
         FROM document_chunks AS c
         JOIN documents AS d ON d.id = c.doc_id
         WHERE c.id IN ({placeholders})"
    );
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map(
        params_from_iter(ranked.iter().map(|(id, _, _)| id)),
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                DocumentSearchResult {
                    doc_id: row.get(1)?,
                    filename: row.get(2)?,
                    chunk_index: row.get::<_, i64>(3)? as usize,
                    excerpt: row.get(4)?,
                    score: 0.0,
                    byte_start: row.get::<_, Option<i64>>(5)?.map(|value| value as usize),
                    byte_end: row.get::<_, Option<i64>>(6)?.map(|value| value as usize),
                    chunk_sha256: row.get(7)?,
                    doc_sha256: row.get(8)?,
                    retrieval: "keyword",
                    similarity: None,
                },
            ))
        },
    )?;
    let mut by_id = rows.collect::<Result<HashMap<_, _>, _>>()?;
    Ok(ranked
        .iter()
        .filter_map(|(id, score, retrieval)| {
            by_id.remove(id).map(|mut result| {
                result.score = *score;
                result.retrieval = retrieval;
                result
            })
        })
        .collect())
}

pub(crate) struct SearchPlan<'a> {
    pub query: &'a str,
    pub doc_ids: Option<&'a [String]>,
    pub top_k: usize,
    pub keyword: bool,
    pub query_vector: Option<&'a [f32]>,
    pub library: Option<LibraryPlan<'a>>,
}

/// A whole-library search, in which a passage from outside `pinned` counts
/// only when its similarity to the query is at least `floor`.
pub(crate) struct LibraryPlan<'a> {
    pub pinned: &'a [String],
    pub floor: f32,
}

/// Two rankings merged best first, without repeats; ties go to the lower id.
fn merge_ranked<S: Copy>(
    mut ranked: Vec<(i64, S)>,
    more: Vec<(i64, S)>,
    better: impl Fn(&S, &S) -> std::cmp::Ordering,
) -> Vec<(i64, S)> {
    let seen = ranked.iter().map(|(id, _)| *id).collect::<HashSet<_>>();
    ranked.extend(more.into_iter().filter(|(id, _)| !seen.contains(id)));
    ranked.sort_by(|left, right| better(&left.1, &right.1).then(left.0.cmp(&right.0)));
    ranked
}

/// Ranks, fuses and citation-checks one search. Returns the verified results
/// and the mode that actually ranked them: a search that asked for vectors
/// but found none indexed in scope reports `keyword`.
pub(crate) fn run_search(
    conn: &Connection,
    plan: &SearchPlan<'_>,
) -> Result<(Vec<DocumentSearchResult>, &'static str), rusqlite::Error> {
    let mut keyword = if plan.keyword {
        keyword_ranked(conn, plan.query, plan.doc_ids, CANDIDATE_POOL)?
    } else {
        Vec::new()
    };
    let mut semantic = match plan.query_vector {
        Some(vector) => {
            document_vectors::semantic_ranked(conn, vector, plan.doc_ids, CANDIDATE_POOL)?
        }
        None => Vec::new(),
    };
    let mut similarities = HashMap::new();
    let mut pinned_chunks = HashSet::new();
    if let (Some(library), Some(vector)) = (&plan.library, plan.query_vector) {
        if !library.pinned.is_empty() {
            // Pinned documents get pools of their own, so the rest of the
            // library cannot crowd them out before the floor is applied.
            if plan.keyword {
                let pinned =
                    keyword_ranked(conn, plan.query, Some(library.pinned), CANDIDATE_POOL)?;
                keyword = merge_ranked(keyword, pinned, |left, right| left.total_cmp(right));
            }
            let pinned = document_vectors::semantic_ranked(
                conn,
                vector,
                Some(library.pinned),
                CANDIDATE_POOL,
            )?;
            semantic = merge_ranked(semantic, pinned, |left, right| right.total_cmp(left));
        }
        let candidates = keyword
            .iter()
            .map(|(id, _)| *id)
            .chain(semantic.iter().map(|(id, _)| *id))
            .collect::<Vec<_>>();
        let relevance = document_vectors::chunk_relevance(conn, vector, &candidates)?;
        let pinned = library
            .pinned
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let eligible = |id: &i64| {
            relevance.get(id).is_some_and(|chunk| {
                pinned.contains(chunk.doc_id.as_str())
                    || chunk
                        .similarity
                        .is_some_and(|similarity| similarity >= library.floor)
            })
        };
        keyword.retain(|(id, _)| eligible(id));
        semantic.retain(|(id, _)| eligible(id));
        pinned_chunks = relevance
            .iter()
            .filter(|(_, chunk)| pinned.contains(chunk.doc_id.as_str()))
            .map(|(id, _)| *id)
            .collect();
        similarities = relevance
            .into_values()
            .filter_map(|chunk| {
                chunk
                    .similarity
                    .map(|similarity| ((chunk.doc_id, chunk.chunk_index), similarity))
            })
            .collect();
    }
    let mut results = if pinned_chunks.is_empty() {
        let (mode, ranked) = fuse_ranked(plan, keyword, semantic);
        let results = first_verified(conn, fetch_results(conn, &ranked)?, plan.top_k)?;
        (results, mode)
    } else {
        // Pinned documents and the rest of the library are ranked apart and
        // share the slots, so passages from elsewhere in the library cannot
        // push an attached document out of the answer.
        let (pinned_keyword, keyword): (Vec<_>, Vec<_>) = keyword
            .into_iter()
            .partition(|(id, _)| pinned_chunks.contains(id));
        let (pinned_semantic, semantic): (Vec<_>, Vec<_>) = semantic
            .into_iter()
            .partition(|(id, _)| pinned_chunks.contains(id));
        let (mode, pinned_ranked) = fuse_ranked(plan, pinned_keyword, pinned_semantic);
        let (_, library_ranked) = fuse_ranked(plan, keyword, semantic);
        let pinned = first_verified(conn, fetch_results(conn, &pinned_ranked)?, plan.top_k)?;
        let library = first_verified(conn, fetch_results(conn, &library_ranked)?, plan.top_k)?;
        (share_slots(pinned, library, plan.top_k), mode)
    };
    for result in &mut results.0 {
        result.similarity = similarities
            .get(&(result.doc_id.clone(), result.chunk_index))
            .copied();
    }
    Ok(results)
}

/// Fills `top_k` slots from two verified rankings: `first` keeps at least half
/// of them when it has that many passages, and either side takes the slots the
/// other cannot fill. `first`'s passages come first.
fn share_slots(
    first: Vec<DocumentSearchResult>,
    second: Vec<DocumentSearchResult>,
    top_k: usize,
) -> Vec<DocumentSearchResult> {
    let second_take = second.len().min(top_k - first.len().min(top_k.div_ceil(2)));
    let first_take = first.len().min(top_k - second_take);
    first
        .into_iter()
        .take(first_take)
        .chain(second.into_iter().take(second_take))
        .collect()
}

/// Orders one search's keyword and semantic candidates into a single ranking,
/// with the mode that ranked them.
fn fuse_ranked(
    plan: &SearchPlan<'_>,
    keyword: Vec<(i64, f64)>,
    semantic: Vec<(i64, f32)>,
) -> (&'static str, Vec<(i64, f32, &'static str)>) {
    if plan.query_vector.is_some() && !plan.keyword {
        (
            "semantic",
            semantic
                .into_iter()
                .map(|(id, score)| (id, score, "semantic"))
                .collect(),
        )
    } else if !semantic.is_empty() || (plan.library.is_some() && plan.query_vector.is_some()) {
        let keyword_ids = keyword.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        let semantic_ids = semantic.iter().map(|(id, _)| *id).collect::<Vec<_>>();
        (
            "hybrid",
            reciprocal_rank_fusion(&keyword_ids, &semantic_ids)
                .into_iter()
                .map(|(id, score, found)| (id, score, found.label()))
                .collect(),
        )
    } else {
        (
            "keyword",
            keyword
                .into_iter()
                // BM25 in SQLite is negative, more negative for a better match; map
                // it into [0, 1) so a better match also scores higher.
                .map(|(id, bm25)| {
                    let strength = bm25.abs();
                    (id, (strength / (1.0 + strength)) as f32, "keyword")
                })
                .collect(),
        )
    }
}

/// The first `top_k` candidates, in rank order, that pass the citation check.
/// Candidates are checked a window at a time, so a search that needs no
/// replacements checks `top_k` of them rather than the whole candidate pool.
fn first_verified(
    conn: &Connection,
    candidates: Vec<DocumentSearchResult>,
    top_k: usize,
) -> Result<Vec<DocumentSearchResult>, rusqlite::Error> {
    let mut verified = Vec::with_capacity(top_k);
    let mut candidates = candidates.into_iter();
    while verified.len() < top_k {
        let window = candidates
            .by_ref()
            .take(top_k - verified.len())
            .collect::<Vec<_>>();
        if window.is_empty() {
            break;
        }
        verified.extend(retain_verifiable(conn, window)?);
    }
    Ok(verified)
}

fn internal_error(code: &'static str) -> impl Fn(rusqlite::Error) -> Response {
    move |error| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            code,
            error.to_string(),
            None,
        )
    }
}

struct SearchOutcome {
    results: Vec<DocumentSearchResult>,
    ranked_by: &'static str,
    coverage: Option<Coverage>,
}

/// A search's work under the database lock. Only `attached`, the explicitly
/// attached documents, fall back to their opening passages when nothing
/// matches; a collection is too broad for that to mean anything. Errors carry
/// the API code they are reported with.
fn search_locked(
    conn: &Connection,
    plan: &SearchPlan<'_>,
    attached: Option<&[String]>,
) -> Result<SearchOutcome, (&'static str, rusqlite::Error)> {
    if plan.query_vector.is_some() {
        // Otherwise a restored chunk would stay skipped for a client that only searches.
        document_vectors::clear_recovered_skips(conn, plan.doc_ids)
            .map_err(|error| ("index_coverage_error", error))?;
    }
    let (mut results, mut ranked_by) =
        run_search(conn, plan).map_err(|error| ("search_error", error))?;
    // In a library-wide search the named documents fall back to their opening
    // passages when none of theirs matched, as they would without the library.
    let named_matched = match &plan.library {
        Some(library) if !library.pinned.is_empty() => results
            .iter()
            .any(|result| library.pinned.contains(&result.doc_id)),
        _ => !results.is_empty(),
    };
    if !named_matched {
        if let Some(attached_ids) = attached.filter(|ids| !ids.is_empty()) {
            let fallback = attached_document_context(conn, attached_ids, plan.top_k)
                .and_then(|results| retain_verifiable(conn, results))
                .map_err(|error| ("attached_document_fallback_error", error))?;
            if !fallback.is_empty() {
                if results.is_empty() {
                    ranked_by = "attached";
                }
                results = share_slots(fallback, results, plan.top_k);
            }
        }
    }
    // Counting coverage scans every chunk in scope, so a search that did not
    // rank by meaning skips it.
    let coverage = if plan.query_vector.is_some() {
        Some(
            document_vectors::coverage(conn, plan.doc_ids)
                .map_err(|error| ("index_coverage_error", error))?,
        )
    } else {
        None
    };
    Ok(SearchOutcome {
        results,
        ranked_by,
        coverage,
    })
}

/// Endpoint: `POST /api/documents/search`
pub async fn search_documents(
    State(state): State<AppState>,
    Json(payload): Json<SearchDocumentsRequest>,
) -> Result<Json<SearchDocumentsResponse>, Response> {
    let query = payload.query.trim().to_string();
    let nothing_to_search = || {
        Json(SearchDocumentsResponse {
            results: Vec::new(),
            retrieval: RetrievalSummary {
                mode: "none",
                semantic: SemanticSummary {
                    available: false,
                    reason: None,
                    coverage: None,
                },
                relevance_floor: None,
            },
        })
    };
    if query.is_empty() {
        return Ok(nothing_to_search());
    }
    let mut scope = {
        let _lock = db_lock().lock().unwrap();
        let conn = open_connection().map_err(internal_error("sqlite_open_error"))?;
        document_collections::resolve_scope(
            &conn,
            payload.doc_ids.as_deref(),
            payload.collection_ids.as_deref(),
        )
        .map_err(CollectionError::into_response)?
    };
    // A library-wide search covers everything; its named documents and
    // collections only decide which passages skip the similarity floor.
    let pinned = if payload.library {
        scope.take().unwrap_or_default()
    } else {
        Vec::new()
    };
    if scope.as_ref().is_some_and(Vec::is_empty) {
        return Ok(nothing_to_search());
    }
    let top_k = payload.top_k.clamp(1, 20);
    let mode = payload.mode;
    if payload.library && mode == SearchMode::Keyword {
        return Err(api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "library_search_needs_meaning",
            "Library-wide search keeps only passages close in meaning to the question, so it cannot run in keyword mode.".to_string(),
            Some("mode"),
        ));
    }
    let refuse_param = if payload.library { "library" } else { "mode" };

    let mut unavailable = None;
    let mut query_vector = None;
    if mode == SearchMode::Keyword {
        unavailable = Some("keyword_mode");
    } else {
        match document_vectors::encoder(&state.models_dir).await {
            Ok(encoder) => {
                let text = query.clone();
                match tokio::task::spawn_blocking(move || encoder.embed_query(&text)).await {
                    Ok(Ok(vector)) => query_vector = Some(vector),
                    Ok(Err(error)) => {
                        tracing::warn!(%error, "document query embedding failed");
                        unavailable = Some(Unavailable::QueryFailed.code());
                    }
                    Err(error) => {
                        tracing::warn!(%error, "document query embedding task failed");
                        unavailable = Some(Unavailable::QueryFailed.code());
                    }
                }
            }
            Err(reason) => {
                if mode != SearchMode::Auto || payload.library {
                    return Err(api_error(
                        StatusCode::CONFLICT,
                        reason.code(),
                        reason.message().to_string(),
                        Some(refuse_param),
                    ));
                }
                unavailable = Some(reason.code());
            }
        }
        if query_vector.is_none() && (mode != SearchMode::Auto || payload.library) {
            return Err(api_error(
                StatusCode::CONFLICT,
                Unavailable::QueryFailed.code(),
                Unavailable::QueryFailed.message().to_string(),
                Some(refuse_param),
            ));
        }
    }

    let (
        SearchOutcome {
            results,
            ranked_by,
            coverage,
        },
        relevance_floor,
    ) = {
        let _lock = db_lock().lock().unwrap();
        let conn = open_connection().map_err(internal_error("sqlite_open_error"))?;
        let relevance_floor = match (payload.library, query_vector.as_deref()) {
            (true, Some(vector)) => Some(document_vectors::library_relevance_floor(
                document_vectors::indexed_chunk_count(&conn, vector.len())
                    .map_err(internal_error("search_error"))?,
            )),
            _ => None,
        };
        let outcome = search_locked(
            &conn,
            &SearchPlan {
                query: &query,
                doc_ids: scope.as_deref(),
                top_k,
                keyword: mode != SearchMode::Semantic,
                query_vector: query_vector.as_deref(),
                library: relevance_floor.map(|floor| LibraryPlan {
                    pinned: &pinned,
                    floor,
                }),
            },
            payload.doc_ids.as_deref(),
        )
        .map_err(|(code, error)| internal_error(code)(error))?;
        (outcome, relevance_floor)
    };
    if coverage
        .as_ref()
        .is_some_and(|coverage| coverage.pending() > 0)
    {
        document_vectors::schedule_indexing(state.models_dir.clone(), false);
    }

    Ok(Json(SearchDocumentsResponse {
        results,
        retrieval: RetrievalSummary {
            mode: ranked_by,
            semantic: SemanticSummary {
                available: query_vector.is_some(),
                reason: unavailable,
                coverage,
            },
            relevance_floor,
        },
    }))
}

/// Endpoint: `GET /api/documents`
pub async fn list_documents(
    State(_state): State<AppState>,
) -> Result<Json<Vec<DocumentMetaView>>, Response> {
    let _lock = db_lock().lock().unwrap();
    let conn = open_connection().map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sqlite_error",
            e.to_string(),
            None,
        )
    })?;

    let mut stmt = conn.prepare(
        "SELECT id, filename, file_type, byte_size, chunk_count, created_at FROM documents ORDER BY created_at DESC"
    ).map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, "prepare_error", e.to_string(), None))?;

    let rows = stmt
        .query_map([], |row| {
            Ok(DocumentMetaView {
                id: row.get(0)?,
                filename: row.get(1)?,
                file_type: row.get(2)?,
                byte_size: row.get(3)?,
                chunk_count: row.get(4)?,
                created_at: row.get(5)?,
            })
        })
        .map_err(|e| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "query_error",
                e.to_string(),
                None,
            )
        })?;

    let mut docs = Vec::new();
    for doc in rows.flatten() {
        docs.push(doc);
    }

    Ok(Json(docs))
}

/// Endpoint: `DELETE /api/documents/:id`
pub async fn delete_document(
    State(_state): State<AppState>,
    AxumPath(doc_id): AxumPath<String>,
) -> Result<impl IntoResponse, Response> {
    let _lock = db_lock().lock().unwrap();
    let mut conn = open_connection().map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sqlite_error",
            e.to_string(),
            None,
        )
    })?;

    let tx = conn.transaction().map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sqlite_tx_error",
            e.to_string(),
            None,
        )
    })?;
    remove_document(&tx, &doc_id).map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "delete_error",
            e.to_string(),
            None,
        )
    })?;
    tx.commit().map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "commit_error",
            e.to_string(),
            None,
        )
    })?;

    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;

    #[test]
    fn test_chunk_spans_are_exact_slices() {
        let text = "Hello world! This is a test paragraph.\n\nSecond paragraph has more content for testing.";
        let chunks = chunk_text_with_spans(text, 50, 10);
        assert!(!chunks.is_empty());
        assert!(chunks[0].text.contains("Hello world!"));
        for chunk in &chunks {
            assert_eq!(chunk.text, text[chunk.start..chunk.end]);
        }
    }

    #[test]
    fn test_sanitize_fts5_query() {
        assert_eq!(sanitize_fts5_query("hello world"), "\"hello\" OR \"world\"");
        assert_eq!(sanitize_fts5_query("   "), "\"\"");
    }

    #[test]
    fn test_extracts_text_from_deflated_docx_document_xml() {
        let cursor = Cursor::new(Vec::new());
        let mut writer = zip::ZipWriter::new(cursor);
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        writer.start_file("word/document.xml", options).unwrap();
        writer
            .write_all(
                br#"<?xml version="1.0"?><w:document><w:body><w:p><w:r><w:t>One &amp; two</w:t></w:r></w:p><w:p><w:r><w:t>three</w:t></w:r></w:p></w:body></w:document>"#,
            )
            .unwrap();
        let bytes = writer.finish().unwrap().into_inner();

        assert_eq!(
            extract_text_from_bytes("NOTES.DOCX", &bytes),
            "One & two three"
        );
    }

    #[test]
    fn attached_documents_supply_context_when_keywords_do_not_match() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        for (doc_id, filename, chunks) in [
            ("doc-a", "alpha.txt", vec!["File1\nFile2", "Alpha tail"]),
            ("doc-b", "beta.txt", vec!["Beta lead", "Beta tail"]),
        ] {
            conn.execute(
                "INSERT INTO documents (id, filename, file_type, byte_size, chunk_count, created_at)
                 VALUES (?1, ?2, 'txt', 1, ?3, 1)",
                params![doc_id, filename, chunks.len() as i64],
            )
            .unwrap();
            for (index, content) in chunks.into_iter().enumerate() {
                conn.execute(
                    "INSERT INTO document_chunks (doc_id, chunk_index, content) VALUES (?1, ?2, ?3)",
                    params![doc_id, index as i64, content],
                )
                .unwrap();
            }
        }

        let doc_ids = vec!["doc-a".to_string(), "doc-b".to_string()];
        let results = attached_document_context(&conn, &doc_ids, 3).unwrap();
        assert_eq!(
            results
                .iter()
                .map(|result| (result.doc_id.as_str(), result.chunk_index))
                .collect::<Vec<_>>(),
            vec![("doc-a", 0), ("doc-b", 0), ("doc-a", 1)]
        );
        assert_eq!(results[0].excerpt, "File1\nFile2");
        assert!(results.iter().all(|result| result.retrieval == "attached"));
    }

    use crate::api::document_vectors::tests::{put_skip, put_vector, seed_passages, unit};

    const POLICY: [&str; 3] = [
        "Enterprise customers may request a refund within 60 days of the invoice date.",
        "Support is staffed from 08:00 to 20:00 UTC on weekdays and weekends.",
        "Exports are fulfilled within thirty days as a signed archive.",
    ];

    fn library() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("PRAGMA foreign_keys = ON;", []).unwrap();
        init_db(&conn).unwrap();
        conn
    }

    /// Seeds the policy and gives its refund, support and export passages
    /// orthogonal vectors. Chunk indexes are 0, 1 and 2 in that order.
    fn index_policy(conn: &Connection, doc_id: &str) -> Vec<i64> {
        let ids = seed_passages(conn, doc_id, &POLICY);
        for (axis, id) in ids.iter().enumerate() {
            let mut vector = [0.0_f32; 3];
            vector[axis] = 1.0;
            put_vector(conn, *id, &vector);
        }
        ids
    }

    fn plan<'a>(query: &'a str, vector: Option<&'a [f32]>) -> SearchPlan<'a> {
        SearchPlan {
            query,
            doc_ids: None,
            top_k: 5,
            keyword: true,
            query_vector: vector,
            library: None,
        }
    }

    fn library_plan<'a>(
        query: &'a str,
        vector: &'a [f32],
        pinned: &'a [String],
        floor: f32,
    ) -> SearchPlan<'a> {
        SearchPlan {
            library: Some(LibraryPlan { pinned, floor }),
            ..plan(query, Some(vector))
        }
    }

    #[test]
    fn a_library_search_keeps_only_passages_that_clear_the_floor() {
        let conn = library();
        index_policy(&conn, "policy");
        // Cosine 0.8 to the refund passage, 0.6 to support, 0 to exports.
        let query = unit(&[0.8, 0.6, 0.0]);
        let (results, mode) = run_search(&conn, &library_plan("refund", &query, &[], 0.5)).unwrap();
        assert_eq!(mode, "hybrid");
        assert_eq!(found(&results), [(0, "hybrid"), (1, "semantic")]);
        let similarities = results
            .iter()
            .map(|result| result.similarity)
            .collect::<Vec<_>>();
        assert!((similarities[0].unwrap() - 0.8).abs() < 1e-6);
        assert!((similarities[1].unwrap() - 0.6).abs() < 1e-6);

        let (results, mode) = run_search(&conn, &library_plan("refund", &query, &[], 0.9)).unwrap();
        assert_eq!(
            mode, "hybrid",
            "the search ranked by meaning even though nothing cleared the floor"
        );
        assert!(
            results.is_empty(),
            "a keyword match below the floor is left out too"
        );
    }

    #[test]
    fn pinned_documents_skip_the_floor_and_the_rest_of_the_library_does_not() {
        let conn = library();
        index_policy(&conn, "attached");
        index_policy(&conn, "elsewhere");
        let pinned = vec!["attached".to_string()];
        let query = unit(&[0.0, 1.0, 0.0]);
        let (results, _) =
            run_search(&conn, &library_plan("refund", &query, &pinned, 0.9)).unwrap();
        let mut served = results
            .iter()
            .map(|result| (result.doc_id.as_str(), result.chunk_index))
            .collect::<Vec<_>>();
        served.sort();
        assert_eq!(
            served,
            [
                ("attached", 0),
                ("attached", 1),
                ("attached", 2),
                ("elsewhere", 1)
            ],
            "every attached passage counts; elsewhere only the support passage clears 0.9"
        );
    }

    #[test]
    fn closer_library_passages_do_not_crowd_out_an_attached_document() {
        let conn = library();
        for doc_id in ["first", "second", "third"] {
            index_policy(&conn, doc_id);
        }
        // Every attached passage is orthogonal to the query.
        for id in seed_passages(&conn, "attached", &POLICY) {
            put_vector(&conn, id, &[0.0, 0.0, 1.0]);
        }
        let pinned = vec!["attached".to_string()];
        let query = unit(&[1.0, 0.0, 0.0]);
        let plan = SearchPlan {
            top_k: 2,
            ..library_plan("unmatched", &query, &pinned, 0.5)
        };
        let (results, _) = run_search(&conn, &plan).unwrap();
        let docs = results
            .iter()
            .map(|result| result.doc_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0], "attached", "the attached document keeps a slot");
        assert_ne!(docs[1], "attached", "the library keeps the other");
    }

    #[test]
    fn an_unmatched_attached_document_still_falls_back_beside_library_passages() {
        let conn = library();
        index_policy(&conn, "elsewhere");
        seed_passages(&conn, "attached", &POLICY);
        let pinned = vec!["attached".to_string()];
        let query = unit(&[1.0, 0.0, 0.0]);
        let plan = library_plan("unmatched", &query, &pinned, 0.5);
        let outcome = search_locked(&conn, &plan, Some(&pinned)).unwrap();
        assert!(outcome
            .results
            .iter()
            .any(|result| result.doc_id == "attached" && result.chunk_index == 0));
        assert!(outcome
            .results
            .iter()
            .any(|result| result.doc_id == "elsewhere" && result.chunk_index == 0));
    }

    #[test]
    fn shared_slots_give_each_side_what_the_other_cannot_use() {
        let conn = library();
        index_policy(&conn, "a");
        index_policy(&conn, "b");
        let query = unit(&[1.0, 1.0, 1.0]);
        let take = |doc_id: &str| {
            let doc_ids = vec![doc_id.to_string()];
            let plan = SearchPlan {
                doc_ids: Some(&doc_ids),
                ..plan("unmatched", Some(&query))
            };
            run_search(&conn, &plan).unwrap().0
        };
        let count = |results: &[DocumentSearchResult], doc_id: &str| {
            results.iter().filter(|r| r.doc_id == doc_id).count()
        };
        let shared = share_slots(take("a"), take("b"), 4);
        assert_eq!((count(&shared, "a"), count(&shared, "b")), (2, 2));
        let shared = share_slots(take("a"), Vec::new(), 4);
        assert_eq!(count(&shared, "a"), 3);
        let shared = share_slots(take("a").into_iter().take(1).collect(), take("b"), 4);
        assert_eq!((count(&shared, "a"), count(&shared, "b")), (1, 3));
        let shared = share_slots(take("a"), take("b"), 1);
        assert_eq!((count(&shared, "a"), count(&shared, "b")), (1, 0));
    }

    #[test]
    fn a_chunk_without_a_current_vector_counts_only_when_pinned() {
        let conn = library();
        index_policy(&conn, "indexed");
        seed_passages(&conn, "plain", &POLICY);
        let query = unit(&[1.0, 0.0, 0.0]);
        let (results, _) = run_search(&conn, &library_plan("refund", &query, &[], 0.5)).unwrap();
        assert!(results.iter().all(|result| result.doc_id == "indexed"));
        assert!(!results.is_empty());

        let pinned = vec!["plain".to_string()];
        let (results, _) =
            run_search(&conn, &library_plan("refund", &query, &pinned, 0.5)).unwrap();
        assert!(results.iter().any(|result| result.doc_id == "plain"
            && result.chunk_index == 0
            && result.similarity.is_none()));
    }

    #[test]
    fn the_library_floor_rises_with_its_indexed_chunks() {
        let conn = library();
        index_policy(&conn, "policy");
        seed_passages(&conn, "plain", &POLICY);
        assert_eq!(
            document_vectors::indexed_chunk_count(&conn, 3).unwrap(),
            3,
            "only chunks with a current vector count"
        );
        assert_eq!(
            document_vectors::indexed_chunk_count(&conn, 768).unwrap(),
            0
        );
        let floor = document_vectors::library_relevance_floor;
        assert!((floor(0) - 0.6408).abs() < 1e-6 && (floor(1) - 0.6408).abs() < 1e-6);
        assert!((floor(4406) - 0.6895).abs() < 1e-4);
        assert!(floor(22) < floor(1_000) && floor(1_000) < floor(100_000));
    }

    #[test]
    fn merged_rankings_keep_each_chunk_once_in_score_order() {
        assert_eq!(
            merge_ranked(
                vec![(3, -2.0), (9, -1.0)],
                vec![(9, -1.0), (4, -3.0), (5, -1.0)],
                |l: &f64, r: &f64| l.total_cmp(r)
            ),
            vec![(4, -3.0), (3, -2.0), (5, -1.0), (9, -1.0)]
        );
    }

    fn found(results: &[DocumentSearchResult]) -> Vec<(usize, &'static str)> {
        results
            .iter()
            .map(|result| (result.chunk_index, result.retrieval))
            .collect()
    }

    #[test]
    fn fusion_rewards_agreement_and_breaks_ties_deterministically() {
        let fused = reciprocal_rank_fusion(&[10, 20, 30], &[30, 40]);
        assert_eq!(
            fused
                .iter()
                .map(|(id, _, found)| (*id, *found))
                .collect::<Vec<_>>(),
            vec![
                (30, Found::Both),
                (10, Found::Keyword),
                (20, Found::Keyword),
                (40, Found::Semantic)
            ],
            "found by both outranks either list's top hit alone; equal ranks tie on chunk id"
        );
        let expected = (1.0 / 63.0 + 1.0 / 61.0) as f32;
        assert!((fused[0].1 - expected).abs() < 1e-7);

        let tied = reciprocal_rank_fusion(&[7], &[3]);
        assert_eq!(
            tied.iter().map(|(id, _, _)| *id).collect::<Vec<_>>(),
            vec![3, 7]
        );
        assert_eq!(
            tied[0].1, tied[1].1,
            "equal ranks score equally; the lower id goes first"
        );
        assert!(reciprocal_rank_fusion(&[], &[]).is_empty());
    }

    #[test]
    fn without_vectors_search_is_the_plain_keyword_ranking() {
        let conn = library();
        seed_passages(&conn, "policy", &POLICY);
        let (results, mode) = run_search(&conn, &plan("refund invoice", None)).unwrap();
        assert_eq!(mode, "keyword");
        assert_eq!(found(&results), vec![(0, "keyword")]);
        assert!(results[0].score > 0.0 && results[0].score < 1.0);

        let (results, _) = run_search(&conn, &plan("within days", None)).unwrap();
        assert!(results.len() >= 2);
        assert!(
            results
                .windows(2)
                .all(|pair| pair[0].score >= pair[1].score),
            "a better keyword match scores higher: {:?}",
            results
                .iter()
                .map(|result| result.score)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_paraphrase_with_no_shared_words_is_found_by_meaning() {
        let conn = library();
        index_policy(&conn, "policy");
        let query = unit(&[0.9, 0.1, 0.0]);

        let (keyword_only, _) = run_search(&conn, &plan("reimbursement deadline", None)).unwrap();
        assert!(
            keyword_only.is_empty(),
            "no word of the question is in the policy"
        );

        let (results, mode) =
            run_search(&conn, &plan("reimbursement deadline", Some(&query))).unwrap();
        assert_eq!(mode, "hybrid");
        assert_eq!(found(&results)[0], (0, "semantic"));
    }

    #[test]
    fn a_chunk_both_rankers_found_leads_the_fused_results() {
        let conn = library();
        index_policy(&conn, "policy");
        // Meaning leans toward exports; the keyword names the refund passage.
        let query = unit(&[0.6, 0.0, 0.8]);
        let (results, mode) = run_search(&conn, &plan("refund", Some(&query))).unwrap();
        assert_eq!(mode, "hybrid");
        assert_eq!(found(&results)[..2], [(0, "hybrid"), (2, "semantic")]);
    }

    #[test]
    fn semantic_mode_ranks_by_similarity_alone() {
        let conn = library();
        index_policy(&conn, "policy");
        let query = unit(&[0.0, 1.0, 0.1]);
        let (results, mode) = run_search(
            &conn,
            &SearchPlan {
                keyword: false,
                ..plan("refund", Some(&query))
            },
        )
        .unwrap();
        assert_eq!(mode, "semantic");
        assert_eq!(
            found(&results)[0],
            (1, "semantic"),
            "the keyword is ignored"
        );
        assert!(results.iter().all(|result| result.retrieval == "semantic"));
    }

    #[test]
    fn a_scope_with_nothing_indexed_reports_keyword_ranking() {
        let conn = library();
        index_policy(&conn, "indexed");
        seed_passages(&conn, "plain", &POLICY);
        let query = unit(&[1.0, 0.0, 0.0]);
        let scope = vec!["plain".to_string()];
        let (results, mode) = run_search(
            &conn,
            &SearchPlan {
                doc_ids: Some(&scope),
                ..plan("refund", Some(&query))
            },
        )
        .unwrap();
        assert_eq!(
            mode, "keyword",
            "vectors outside the scope do not make it hybrid"
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "plain");
    }

    #[test]
    fn a_collection_scope_ranks_only_its_members() {
        let mut conn = library();
        index_policy(&conn, "member");
        index_policy(&conn, "outsider");
        let finance = document_collections::create(&conn, "Finance").unwrap();
        document_collections::add_documents(&mut conn, &finance.id, &["member".to_string()])
            .unwrap();
        let scope = document_collections::resolve_scope(&conn, None, Some(&[finance.id])).unwrap();
        let query = unit(&[1.0, 0.0, 0.0]);
        let (results, mode) = run_search(
            &conn,
            &SearchPlan {
                doc_ids: scope.as_deref(),
                ..plan("refund", Some(&query))
            },
        )
        .unwrap();
        assert_eq!(mode, "hybrid");
        assert_eq!(
            results
                .iter()
                .map(|result| result.doc_id.as_str())
                .collect::<Vec<_>>(),
            ["member", "member", "member"],
            "the outsider holds identical passages and still never appears"
        );
    }

    #[test]
    fn a_meaning_match_from_a_corrupted_document_is_withheld_and_backfilled() {
        let conn = library();
        index_policy(&conn, "corrupted");
        index_policy(&conn, "healthy");
        conn.execute(
            "UPDATE documents SET source_text = REPLACE(source_text, '60 days', '90 days') WHERE id = 'corrupted'",
            [],
        )
        .unwrap();
        let query = unit(&[1.0, 0.0, 0.0]);
        let (results, _) = run_search(
            &conn,
            &SearchPlan {
                top_k: 2,
                ..plan("reimbursement", Some(&query))
            },
        )
        .unwrap();
        assert_eq!(
            results.len(),
            2,
            "withheld candidates are replaced, not just dropped"
        );
        assert!(results.iter().all(|result| result.doc_id == "healthy"));
        assert_eq!(results[0].chunk_index, 0);
        assert!(results
            .iter()
            .all(|result| !result.excerpt.contains("90 days")));
    }

    #[test]
    fn verification_keeps_rank_order_and_backfills_across_windows() {
        let conn = library();
        let corrupted = seed_passages(&conn, "corrupted", &POLICY);
        let healthy = seed_passages(&conn, "healthy", &POLICY);
        conn.execute(
            "UPDATE documents SET source_text = REPLACE(source_text, '60 days', '90 days') WHERE id = 'corrupted'",
            [],
        )
        .unwrap();
        // Two withheld candidates lead, so the first window keeps nothing.
        let order = [
            corrupted[0],
            corrupted[1],
            healthy[2],
            corrupted[2],
            healthy[0],
            healthy[1],
        ];
        let ranked = order
            .iter()
            .map(|id| (*id, 0.0_f32, "keyword"))
            .collect::<Vec<_>>();
        let pick = |top_k| {
            first_verified(&conn, fetch_results(&conn, &ranked).unwrap(), top_k)
                .unwrap()
                .iter()
                .map(|result| (result.doc_id.clone(), result.chunk_index))
                .collect::<Vec<_>>()
        };
        let healthy_at = |index: usize| ("healthy".to_string(), index);
        assert_eq!(pick(2), vec![healthy_at(2), healthy_at(0)]);
        assert_eq!(
            pick(10),
            vec![healthy_at(2), healthy_at(0), healthy_at(1)],
            "a pool that runs out returns what verified, in rank order"
        );
    }

    #[test]
    fn a_search_by_meaning_returns_a_restored_skipped_chunk_to_pending() {
        let conn = library();
        let ids = seed_passages(&conn, "policy", &POLICY[..1]);
        // Skipped while its text failed its hash; the text has since been restored.
        put_skip(&conn, ids[0]);
        let stuck = document_vectors::coverage(&conn, None).unwrap();
        assert_eq!(
            (
                stuck.indexable_chunks,
                stuck.indexed_chunks,
                stuck.skipped_chunks,
                stuck.pending()
            ),
            (1, 0, 1, 0),
            "nothing would start the indexer for it"
        );

        let keyword_only = search_locked(&conn, &plan("refund", None), None).unwrap();
        assert_eq!(keyword_only.coverage, None);
        assert_eq!(
            document_vectors::coverage(&conn, None)
                .unwrap()
                .skipped_chunks,
            1,
            "a keyword search does not touch the index"
        );

        let query = unit(&[1.0, 0.0, 0.0]);
        let searched = search_locked(&conn, &plan("refund", Some(&query)), None).unwrap();
        let coverage = searched.coverage.unwrap();
        assert_eq!(
            (
                coverage.indexed_chunks,
                coverage.skipped_chunks,
                coverage.pending()
            ),
            (0, 0, 1),
            "the search itself reports it pending, which schedules indexing"
        );
        assert_eq!(searched.results.len(), 1, "keyword ranking still finds it");
    }

    #[test]
    fn a_search_by_meaning_leaves_a_still_tampered_chunk_skipped() {
        let conn = library();
        let ids = seed_passages(&conn, "policy", &POLICY);
        conn.execute(
            "UPDATE document_chunks SET content = 'tampered' WHERE id = ?1",
            params![ids[0]],
        )
        .unwrap();
        put_skip(&conn, ids[0]);
        put_vector(&conn, ids[1], &[0.0, 1.0, 0.0]);
        put_vector(&conn, ids[2], &[0.0, 0.0, 1.0]);
        let query = unit(&[0.0, 1.0, 0.0]);
        let coverage = search_locked(&conn, &plan("support", Some(&query)), None)
            .unwrap()
            .coverage
            .unwrap();
        assert_eq!(
            (
                coverage.indexed_chunks,
                coverage.skipped_chunks,
                coverage.pending()
            ),
            (2, 1, 0)
        );
    }
}
