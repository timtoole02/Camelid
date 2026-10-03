//! Semantic half of Knowledge Library retrieval.
//!
//! Chunk vectors come from one pinned encoder and are stored beside the chunks
//! they were computed from. Each vector is bound to the hash of the chunk text
//! it was computed from, so a vector is only scored while its chunk still
//! hashes to what was embedded. Retrieval scores with the same cosine function
//! as `/v1/rerank`, which is why a separate rerank stage would add nothing.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use axum::Json;
use rusqlite::{params, params_from_iter, types::Value, Connection};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::embedding::{cosine_similarity, EmbeddingRuntime};

use super::citations::sha256_hex;
use super::documents::{db_lock, open_connection};
use super::{api_error, AppState};

pub(crate) const ENCODER_FILENAME: &str = "nomic-embed-text-v1.5.Q8_0.gguf";
pub(crate) const ENCODER_SHA256: &str =
    "3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7";
const ENCODER_BYTES: u64 = 146_146_432;
const DISABLE_ENV: &str = "CAMELID_DOCUMENT_SEMANTIC";
/// Chunks per encoder call. Small batches keep the database lock free for
/// searches between batches.
const INDEX_BATCH: usize = 16;

/// The cosine similarity below which a library-wide search leaves a passage
/// out. It rises with the library's indexed chunks, because the best chance
/// match an unrelated message finds does; fitted for the pinned encoder, see
/// docs/architecture/EMBEDDINGS.md.
pub(crate) fn library_relevance_floor(indexed_chunks: usize) -> f32 {
    const AT_ONE_CHUNK: f64 = 0.6408;
    const PER_LN_CHUNK: f64 = 0.0058;
    (AT_ONE_CHUNK + PER_LN_CHUNK * (indexed_chunks.max(1) as f64).ln()).clamp(0.3, 0.9) as f32
}

/// Chunks across the library with a current vector from the pinned encoder.
pub(crate) fn indexed_chunk_count(
    conn: &Connection,
    dims: usize,
) -> Result<usize, rusqlite::Error> {
    conn.query_row(
        "SELECT COUNT(*) FROM document_chunk_vectors AS v
         JOIN document_chunks AS c ON c.id = v.chunk_id
         WHERE v.vector IS NOT NULL AND v.encoder_sha256 = ?1 AND v.dims = ?2
           AND v.chunk_sha256 = c.chunk_sha256",
        params![ENCODER_SHA256, dims as i64],
        |row| row.get::<_, i64>(0),
    )
    .map(|count| count.max(0) as usize)
}

pub(crate) fn init_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
    // `vector IS NULL` records a chunk that was skipped because its stored
    // text no longer matched its recorded hash.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS document_chunk_vectors (
            chunk_id INTEGER PRIMARY KEY REFERENCES document_chunks(id) ON DELETE CASCADE,
            chunk_sha256 TEXT NOT NULL,
            encoder_sha256 TEXT NOT NULL,
            dims INTEGER NOT NULL,
            vector BLOB
        );",
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unavailable {
    Disabled,
    NotInstalled,
    Mismatch,
    LoadFailed,
    QueryFailed,
}

impl Unavailable {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::Disabled => "semantic_disabled",
            Self::NotInstalled => "encoder_not_installed",
            Self::Mismatch => "encoder_mismatch",
            Self::LoadFailed => "encoder_load_failed",
            Self::QueryFailed => "query_embedding_failed",
        }
    }

    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::Disabled => "Semantic document search is turned off by CAMELID_DOCUMENT_SEMANTIC.",
            Self::NotInstalled => {
                "Semantic document search needs nomic-embed-text-v1.5.Q8_0.gguf in the models directory."
            }
            Self::Mismatch => {
                "The models directory holds a nomic-embed-text-v1.5.Q8_0.gguf that is not the pinned artifact, so it is not used."
            }
            Self::LoadFailed => "The document encoder could not be loaded.",
            Self::QueryFailed => "The query could not be embedded.",
        }
    }
}

pub(crate) struct Encoder {
    runtime: EmbeddingRuntime,
    dims: usize,
}

impl Encoder {
    pub(crate) fn embed_query(&self, text: &str) -> crate::Result<Vec<f32>> {
        self.runtime
            .embed(&self.runtime.prepare_retrieval_query(text), None)
    }

    fn embed_documents(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        let inputs = texts
            .iter()
            .map(|text| self.runtime.prepare_retrieval_document(text))
            .collect::<Vec<_>>();
        self.runtime.embed_batch(&inputs, None)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    len: u64,
    modified: Option<SystemTime>,
}

enum Slot {
    Empty,
    Ready(Arc<Encoder>),
    Failed {
        reason: Unavailable,
        path: PathBuf,
        stamp: FileStamp,
    },
}

fn slot() -> &'static tokio::sync::Mutex<Slot> {
    static SLOT: OnceLock<tokio::sync::Mutex<Slot>> = OnceLock::new();
    SLOT.get_or_init(|| tokio::sync::Mutex::new(Slot::Empty))
}

fn disabled_by(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        )
    })
}

/// The pinned encoder, loaded once per process. A missing file is re-checked
/// on every call so a later download is picked up; a file that failed
/// verification is not re-hashed until it changes.
pub(crate) async fn encoder(models_dir: &Path) -> Result<Arc<Encoder>, Unavailable> {
    if disabled_by(std::env::var(DISABLE_ENV).ok().as_deref()) {
        return Err(Unavailable::Disabled);
    }
    let mut slot = slot().lock().await;
    if let Slot::Ready(encoder) = &*slot {
        return Ok(Arc::clone(encoder));
    }
    let path = models_dir.join(ENCODER_FILENAME);
    let metadata = match std::fs::metadata(&path) {
        Ok(metadata) if metadata.is_file() => metadata,
        _ => return Err(Unavailable::NotInstalled),
    };
    let stamp = FileStamp {
        len: metadata.len(),
        modified: metadata.modified().ok(),
    };
    if let Slot::Failed {
        reason,
        path: seen_path,
        stamp: seen_stamp,
    } = &*slot
    {
        if *seen_path == path && *seen_stamp == stamp {
            return Err(*reason);
        }
    }
    let load_path = path.clone();
    let loaded = tokio::task::spawn_blocking(move || load_verified(&load_path))
        .await
        .unwrap_or(Err(Unavailable::LoadFailed));
    match loaded {
        Ok(encoder) => {
            let encoder = Arc::new(encoder);
            *slot = Slot::Ready(Arc::clone(&encoder));
            Ok(encoder)
        }
        Err(reason) => {
            *slot = Slot::Failed {
                reason,
                path,
                stamp,
            };
            Err(reason)
        }
    }
}

fn verify_file(path: &Path, expected_len: u64, expected_sha256: &str) -> Result<(), Unavailable> {
    let mut file = File::open(path).map_err(|_| Unavailable::NotInstalled)?;
    let len = file.metadata().map_err(|_| Unavailable::LoadFailed)?.len();
    if len != expected_len {
        return Err(Unavailable::Mismatch);
    }
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|_| Unavailable::LoadFailed)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    if format!("{:x}", hasher.finalize()) != expected_sha256 {
        return Err(Unavailable::Mismatch);
    }
    Ok(())
}

fn load_verified(path: &Path) -> Result<Encoder, Unavailable> {
    verify_file(path, ENCODER_BYTES, ENCODER_SHA256)?;
    let runtime = EmbeddingRuntime::load(path).map_err(|error| {
        tracing::warn!(%error, "document encoder failed to load");
        Unavailable::LoadFailed
    })?;
    let dims = runtime
        .embed(&runtime.prepare_retrieval_document("probe"), None)
        .map_err(|error| {
            tracing::warn!(%error, "document encoder failed its probe embedding");
            Unavailable::LoadFailed
        })?
        .len();
    Ok(Encoder { runtime, dims })
}

fn to_blob(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn from_blob(blob: &[u8], dims: usize) -> Option<Vec<f32>> {
    if dims == 0 || blob.len() != dims * 4 {
        return None;
    }
    Some(
        blob.chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect(),
    )
}

fn scope_clause(doc_ids: Option<&[String]>, binds: &mut Vec<Value>) -> String {
    let Some(doc_ids) = doc_ids else {
        return String::new();
    };
    let placeholders = doc_ids
        .iter()
        .map(|doc_id| {
            binds.push(Value::Text(doc_id.clone()));
            format!("?{}", binds.len())
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(" AND c.doc_id IN ({placeholders})")
}

/// Chunks in scope ranked by cosine similarity to `query`, best first. Only
/// vectors from the pinned encoder whose chunk still carries the hash that was
/// embedded take part.
pub(crate) fn semantic_ranked(
    conn: &Connection,
    query: &[f32],
    doc_ids: Option<&[String]>,
    limit: usize,
) -> Result<Vec<(i64, f32)>, rusqlite::Error> {
    let mut binds = vec![
        Value::Text(ENCODER_SHA256.to_string()),
        Value::Integer(query.len() as i64),
    ];
    let sql = format!(
        "SELECT v.chunk_id, v.vector
         FROM document_chunk_vectors AS v
         JOIN document_chunks AS c ON c.id = v.chunk_id
         WHERE v.vector IS NOT NULL AND v.encoder_sha256 = ?1 AND v.dims = ?2
           AND v.chunk_sha256 = c.chunk_sha256{}",
        scope_clause(doc_ids, &mut binds)
    );
    let mut statement = conn.prepare(&sql)?;
    let mut ranked = Vec::new();
    let rows = statement.query_map(params_from_iter(binds.iter()), |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
    })?;
    for row in rows {
        let (chunk_id, blob) = row?;
        let Some(vector) = from_blob(&blob, query.len()) else {
            continue;
        };
        if let Ok(score) = cosine_similarity(query, &vector) {
            ranked.push((chunk_id, score));
        }
    }
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    ranked.truncate(limit);
    Ok(ranked)
}

/// Where a candidate chunk sits, and its cosine similarity to the query when
/// it has a current vector from the pinned encoder.
pub(crate) struct ChunkRelevance {
    pub doc_id: String,
    pub chunk_index: usize,
    pub similarity: Option<f32>,
}

pub(crate) fn chunk_relevance(
    conn: &Connection,
    query: &[f32],
    chunk_ids: &[i64],
) -> Result<HashMap<i64, ChunkRelevance>, rusqlite::Error> {
    if chunk_ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut binds = vec![
        Value::Text(ENCODER_SHA256.to_string()),
        Value::Integer(query.len() as i64),
    ];
    let placeholders = chunk_ids
        .iter()
        .map(|id| {
            binds.push(Value::Integer(*id));
            format!("?{}", binds.len())
        })
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT c.id, c.doc_id, c.chunk_index, v.vector
         FROM document_chunks AS c
         LEFT JOIN document_chunk_vectors AS v
           ON v.chunk_id = c.id AND v.vector IS NOT NULL AND v.encoder_sha256 = ?1
          AND v.dims = ?2 AND v.chunk_sha256 = c.chunk_sha256
         WHERE c.id IN ({placeholders})"
    );
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(binds.iter()), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)? as usize,
            row.get::<_, Option<Vec<u8>>>(3)?,
        ))
    })?;
    let mut relevance = HashMap::new();
    for row in rows {
        let (id, doc_id, chunk_index, blob) = row?;
        let similarity = blob
            .and_then(|blob| from_blob(&blob, query.len()))
            .and_then(|vector| cosine_similarity(query, &vector).ok());
        relevance.insert(
            id,
            ChunkRelevance {
                doc_id,
                chunk_index,
                similarity,
            },
        );
    }
    Ok(relevance)
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Coverage {
    /// Chunks that carry a citation binding and can therefore be embedded.
    pub indexable_chunks: u64,
    pub indexed_chunks: u64,
    /// Chunks whose stored text no longer matched its hash when the indexer
    /// reached them. Citation checks withhold these anyway.
    pub skipped_chunks: u64,
}

impl Coverage {
    pub(crate) fn pending(&self) -> u64 {
        self.indexable_chunks
            .saturating_sub(self.indexed_chunks + self.skipped_chunks)
    }
}

const COVERAGE_SELECT: &str = "SUM(c.chunk_sha256 IS NOT NULL),
        SUM(v.vector IS NOT NULL),
        SUM(v.chunk_id IS NOT NULL AND v.vector IS NULL)
     FROM document_chunks AS c
     LEFT JOIN document_chunk_vectors AS v
       ON v.chunk_id = c.id AND v.encoder_sha256 = ?1 AND v.chunk_sha256 = c.chunk_sha256";

fn coverage_from_row(row: &rusqlite::Row<'_>, offset: usize) -> rusqlite::Result<Coverage> {
    let count = |index: usize| -> rusqlite::Result<u64> {
        Ok(row
            .get::<_, Option<i64>>(offset + index)?
            .unwrap_or(0)
            .max(0) as u64)
    };
    Ok(Coverage {
        indexable_chunks: count(0)?,
        indexed_chunks: count(1)?,
        skipped_chunks: count(2)?,
    })
}

pub(crate) fn coverage(
    conn: &Connection,
    doc_ids: Option<&[String]>,
) -> Result<Coverage, rusqlite::Error> {
    let mut binds = vec![Value::Text(ENCODER_SHA256.to_string())];
    let scope = scope_clause(doc_ids, &mut binds);
    let sql = format!("SELECT {COVERAGE_SELECT} WHERE 1 = 1{scope}");
    conn.query_row(&sql, params_from_iter(binds.iter()), |row| {
        coverage_from_row(row, 0)
    })
}

fn coverage_by_document(conn: &Connection) -> Result<Vec<(String, Coverage)>, rusqlite::Error> {
    let sql = format!("SELECT c.doc_id, {COVERAGE_SELECT} GROUP BY c.doc_id ORDER BY c.doc_id");
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map(params![ENCODER_SHA256], |row| {
        Ok((row.get::<_, String>(0)?, coverage_from_row(row, 1)?))
    })?;
    rows.collect()
}

struct PendingChunk {
    id: i64,
    content: String,
    chunk_sha256: String,
}

/// Up to `limit` chunks with no vector or skip record for the pinned encoder
/// and their current hash, newest first, strictly below `before_id`.
fn pending_batch(
    conn: &Connection,
    before_id: i64,
    limit: usize,
) -> Result<Vec<PendingChunk>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT c.id, c.content, c.chunk_sha256
         FROM document_chunks AS c
         LEFT JOIN document_chunk_vectors AS v
           ON v.chunk_id = c.id AND v.encoder_sha256 = ?1 AND v.chunk_sha256 = c.chunk_sha256
         WHERE v.chunk_id IS NULL AND c.chunk_sha256 IS NOT NULL AND c.id < ?2
         ORDER BY c.id DESC
         LIMIT ?3",
    )?;
    let rows = statement.query_map(params![ENCODER_SHA256, before_id, limit as i64], |row| {
        Ok(PendingChunk {
            id: row.get(0)?,
            content: row.get(1)?,
            chunk_sha256: row.get(2)?,
        })
    })?;
    rows.collect()
}

/// Whether any chunk above `id` is waiting for a vector: an upload that
/// arrived after the current indexing pass started.
fn pending_above(conn: &Connection, id: i64) -> Result<bool, rusqlite::Error> {
    conn.query_row(
        "SELECT EXISTS (
             SELECT 1
             FROM document_chunks AS c
             LEFT JOIN document_chunk_vectors AS v
               ON v.chunk_id = c.id AND v.encoder_sha256 = ?1 AND v.chunk_sha256 = c.chunk_sha256
             WHERE v.chunk_id IS NULL AND c.chunk_sha256 IS NOT NULL AND c.id > ?2
         )",
        params![ENCODER_SHA256, id],
        |row| row.get(0),
    )
}

/// Writes one vector (or a skip record when `vector` is `None`), but only if
/// the chunk still exists with the hash that was embedded: a document
/// re-ingested mid-batch gets new chunk ids and must not inherit these rows.
fn store(
    conn: &Connection,
    chunk_id: i64,
    chunk_sha256: &str,
    dims: usize,
    vector: Option<&[f32]>,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "INSERT OR REPLACE INTO document_chunk_vectors
             (chunk_id, chunk_sha256, encoder_sha256, dims, vector)
         SELECT ?1, ?2, ?3, ?4, ?5
         WHERE EXISTS (SELECT 1 FROM document_chunks WHERE id = ?1 AND chunk_sha256 = ?2)",
        params![
            chunk_id,
            chunk_sha256,
            ENCODER_SHA256,
            dims as i64,
            vector.map(to_blob)
        ],
    )?;
    Ok(())
}

/// A skipped chunk in scope whose text matches its hash again (for example a
/// restored file) becomes pending once more.
pub(crate) fn clear_recovered_skips(
    conn: &Connection,
    doc_ids: Option<&[String]>,
) -> Result<(), rusqlite::Error> {
    let recovered = {
        let mut binds = Vec::new();
        let sql = format!(
            "SELECT c.id, c.content, c.chunk_sha256
             FROM document_chunk_vectors AS v
             JOIN document_chunks AS c ON c.id = v.chunk_id
             WHERE v.vector IS NULL{}",
            scope_clause(doc_ids, &mut binds)
        );
        let mut statement = conn.prepare(&sql)?;
        let rows = statement.query_map(params_from_iter(binds.iter()), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })?;
        rows.filter_map(Result::ok)
            .filter(|(_, content, hash)| {
                hash.as_deref() == Some(sha256_hex(content.as_bytes()).as_str())
            })
            .map(|(id, _, _)| id)
            .collect::<Vec<_>>()
    };
    for chunk_id in recovered {
        conn.execute(
            "DELETE FROM document_chunk_vectors WHERE chunk_id = ?1 AND vector IS NULL",
            params![chunk_id],
        )?;
    }
    Ok(())
}

fn locked<T>(
    conn: &Connection,
    work: impl FnOnce(&Connection) -> Result<T, rusqlite::Error>,
) -> Result<T, String> {
    let _lock = db_lock()
        .lock()
        .map_err(|_| "document database lock poisoned".to_string())?;
    work(conn).map_err(|error| error.to_string())
}

/// Embeds pending chunks, newest first, and returns how many were stored. A
/// pass stops early when a newer upload arrives, so the scheduler starts over
/// from the newest chunk and a small upload does not wait behind a large one.
fn index_pending(encoder: &Encoder) -> Result<usize, String> {
    let conn = {
        let _lock = db_lock()
            .lock()
            .map_err(|_| "document database lock poisoned".to_string())?;
        open_connection().map_err(|error| error.to_string())?
    };
    locked(&conn, |conn| clear_recovered_skips(conn, None))?;
    let mut before_id = i64::MAX;
    let mut pass_top = None;
    let mut stored = 0;
    loop {
        let batch = locked(&conn, |conn| pending_batch(conn, before_id, INDEX_BATCH))?;
        let Some(last) = batch.last() else {
            break;
        };
        before_id = last.id;
        let top = *pass_top.get_or_insert(batch[0].id);
        let (intact, tampered): (Vec<_>, Vec<_>) = batch
            .into_iter()
            .partition(|chunk| sha256_hex(chunk.content.as_bytes()) == chunk.chunk_sha256);
        let texts = intact
            .iter()
            .map(|chunk| chunk.content.clone())
            .collect::<Vec<_>>();
        let vectors = if texts.is_empty() {
            Vec::new()
        } else {
            encoder
                .embed_documents(&texts)
                .map_err(|error| format!("chunk embedding failed: {error}"))?
        };
        if vectors.len() != intact.len()
            || vectors.iter().any(|vector| vector.len() != encoder.dims)
        {
            return Err("the encoder returned an unexpected number or size of vectors".to_string());
        }
        locked(&conn, |conn| {
            for (chunk, vector) in intact.iter().zip(&vectors) {
                store(
                    conn,
                    chunk.id,
                    &chunk.chunk_sha256,
                    encoder.dims,
                    Some(vector),
                )?;
            }
            for chunk in &tampered {
                store(conn, chunk.id, &chunk.chunk_sha256, encoder.dims, None)?;
            }
            Ok(())
        })?;
        stored += intact.len();
        if locked(&conn, |conn| pending_above(conn, top))? {
            break;
        }
    }
    Ok(stored)
}

fn has_pending() -> bool {
    let Ok(_lock) = db_lock().lock() else {
        return false;
    };
    open_connection()
        .and_then(|conn| coverage(&conn, None))
        .is_ok_and(|coverage| coverage.pending() > 0)
}

static INDEXING: AtomicBool = AtomicBool::new(false);

fn last_failure() -> &'static Mutex<Option<String>> {
    static LAST_FAILURE: OnceLock<Mutex<Option<String>>> = OnceLock::new();
    LAST_FAILURE.get_or_init(|| Mutex::new(None))
}

fn recorded_failure() -> Option<String> {
    last_failure()
        .lock()
        .ok()
        .and_then(|failure| failure.clone())
}

fn record_failure(failure: Option<String>) {
    if let Ok(mut slot) = last_failure().lock() {
        *slot = failure;
    }
}

/// Starts the background indexer unless it is already running. `after_new_content`
/// retries even after a failed run; status polls and searches do not, so a
/// persistent failure cannot turn polling into a retry loop.
pub(crate) fn schedule_indexing(models_dir: PathBuf, after_new_content: bool) {
    if !after_new_content && recorded_failure().is_some() {
        return;
    }
    if INDEXING.swap(true, Ordering::AcqRel) {
        return;
    }
    tokio::spawn(async move {
        loop {
            let Ok(encoder) = encoder(&models_dir).await else {
                INDEXING.store(false, Ordering::Release);
                return;
            };
            match tokio::task::spawn_blocking(move || index_pending(&encoder)).await {
                Ok(Ok(stored)) => {
                    record_failure(None);
                    tracing::debug!(stored, "document semantic index run finished");
                }
                Ok(Err(error)) => {
                    tracing::warn!(%error, "document semantic indexing stopped");
                    record_failure(Some(error));
                    INDEXING.store(false, Ordering::Release);
                    return;
                }
                Err(error) => {
                    tracing::warn!(%error, "document semantic indexing task failed");
                    record_failure(Some(error.to_string()));
                    INDEXING.store(false, Ordering::Release);
                    return;
                }
            }
            INDEXING.store(false, Ordering::Release);
            // Chunks committed after the last empty batch must not wait for the next trigger.
            let more = tokio::task::spawn_blocking(has_pending)
                .await
                .unwrap_or(false);
            if !more || INDEXING.swap(true, Ordering::AcqRel) {
                return;
            }
        }
    });
}

#[derive(Debug, Serialize)]
pub struct SemanticIndexStatus {
    pub available: bool,
    pub encoder: &'static str,
    pub indexing: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl SemanticIndexStatus {
    pub(crate) fn from_encoder(encoder: &Result<Arc<Encoder>, Unavailable>) -> Self {
        let reason = encoder.as_ref().err().copied();
        Self {
            available: encoder.is_ok(),
            encoder: ENCODER_FILENAME,
            indexing: INDEXING.load(Ordering::Acquire),
            reason: reason.map(Unavailable::code),
            message: reason.map(Unavailable::message),
            error: recorded_failure(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DocumentIndexStatus {
    pub id: String,
    #[serde(flatten)]
    pub coverage: Coverage,
}

#[derive(Debug, Serialize)]
pub struct IndexStatusResponse {
    pub semantic: SemanticIndexStatus,
    pub documents: Vec<DocumentIndexStatus>,
}

/// Endpoint: `GET /api/documents/index-status`
pub async fn index_status(
    State(state): State<AppState>,
) -> Result<Json<IndexStatusResponse>, Response> {
    let encoder = encoder(&state.models_dir).await;
    let documents = {
        let _lock = db_lock().lock().unwrap();
        let conn = open_connection().map_err(|e| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "sqlite_open_error",
                e.to_string(),
                None,
            )
        })?;
        // A skipped chunk whose text was restored must count as pending here,
        // or nothing would ever start the indexer for it.
        clear_recovered_skips(&conn, None)
            .and_then(|()| coverage_by_document(&conn))
            .map_err(|e| {
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "index_status_query_error",
                    e.to_string(),
                    None,
                )
            })?
    };
    if encoder.is_ok() && documents.iter().any(|(_, coverage)| coverage.pending() > 0) {
        schedule_indexing(state.models_dir.clone(), false);
    }
    Ok(Json(IndexStatusResponse {
        semantic: SemanticIndexStatus::from_encoder(&encoder),
        documents: documents
            .into_iter()
            .map(|(id, coverage)| DocumentIndexStatus { id, coverage })
            .collect(),
    }))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::api::citations::chunk_text_with_spans;
    use crate::api::documents::init_db;

    /// Seeds a document with exact spans and returns its chunk row ids in order.
    pub(crate) fn seed(conn: &Connection, doc_id: &str, text: &str) -> Vec<i64> {
        let spans = chunk_text_with_spans(text, 120, 24);
        conn.execute(
            "INSERT OR REPLACE INTO documents
             (id, filename, file_type, byte_size, chunk_count, created_at, source_sha256, text_sha256, source_text)
             VALUES (?1, ?2, 'txt', ?3, ?4, 1, ?5, ?5, ?6)",
            params![
                doc_id,
                format!("{doc_id}.txt"),
                text.len() as i64,
                spans.len() as i64,
                sha256_hex(text.as_bytes()),
                text
            ],
        )
        .unwrap();
        spans
            .iter()
            .enumerate()
            .map(|(index, span)| {
                conn.execute(
                    "INSERT INTO document_chunks
                     (doc_id, chunk_index, content, byte_start, byte_end, chunk_sha256)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        doc_id,
                        index as i64,
                        span.text,
                        span.start as i64,
                        span.end as i64,
                        span.sha256()
                    ],
                )
                .unwrap();
                let rowid = conn.last_insert_rowid();
                conn.execute(
                    "INSERT INTO document_chunks_fts (rowid, content) VALUES (?1, ?2)",
                    params![rowid, span.text],
                )
                .unwrap();
                rowid
            })
            .collect()
    }

    /// Seeds a document whose chunks are exactly `passages`, joined by blank
    /// lines, and returns their chunk row ids in order.
    pub(crate) fn seed_passages(conn: &Connection, doc_id: &str, passages: &[&str]) -> Vec<i64> {
        let text = passages.join("\n\n");
        conn.execute(
            "INSERT OR REPLACE INTO documents
             (id, filename, file_type, byte_size, chunk_count, created_at, source_sha256, text_sha256, source_text)
             VALUES (?1, ?2, 'txt', ?3, ?4, 1, ?5, ?5, ?6)",
            params![
                doc_id,
                format!("{doc_id}.txt"),
                text.len() as i64,
                passages.len() as i64,
                sha256_hex(text.as_bytes()),
                text
            ],
        )
        .unwrap();
        let mut start = 0;
        passages
            .iter()
            .enumerate()
            .map(|(index, passage)| {
                let end = start + passage.len();
                conn.execute(
                    "INSERT INTO document_chunks
                     (doc_id, chunk_index, content, byte_start, byte_end, chunk_sha256)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        doc_id,
                        index as i64,
                        passage,
                        start as i64,
                        end as i64,
                        sha256_hex(passage.as_bytes())
                    ],
                )
                .unwrap();
                let rowid = conn.last_insert_rowid();
                conn.execute(
                    "INSERT INTO document_chunks_fts (rowid, content) VALUES (?1, ?2)",
                    params![rowid, passage],
                )
                .unwrap();
                start = end + 2;
                rowid
            })
            .collect()
    }

    pub(crate) fn put_vector(conn: &Connection, chunk_id: i64, vector: &[f32]) {
        let hash: String = conn
            .query_row(
                "SELECT chunk_sha256 FROM document_chunks WHERE id = ?1",
                params![chunk_id],
                |row| row.get(0),
            )
            .unwrap();
        store(conn, chunk_id, &hash, vector.len(), Some(vector)).unwrap();
    }

    /// Records `chunk_id` as skipped, the way the indexer does when its text
    /// no longer matches its hash.
    pub(crate) fn put_skip(conn: &Connection, chunk_id: i64) {
        let hash: String = conn
            .query_row(
                "SELECT chunk_sha256 FROM document_chunks WHERE id = ?1",
                params![chunk_id],
                |row| row.get(0),
            )
            .unwrap();
        store(conn, chunk_id, &hash, 3, None).unwrap();
    }

    pub(crate) fn unit(values: &[f32]) -> Vec<f32> {
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        values.iter().map(|value| value / norm).collect()
    }

    fn memory() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("PRAGMA foreign_keys = ON;", []).unwrap();
        init_db(&conn).unwrap();
        conn
    }

    const TEXT: &str = "Enterprise customers may request a refund within 60 days of the invoice date. Refunds are issued to the original payment method. Trial accounts are not eligible for refunds under any circumstance. Exports are fulfilled within thirty days.";

    #[test]
    fn vectors_round_trip_through_their_blob() {
        let vector = vec![0.25, -1.5, 3.0e-7, f32::MIN_POSITIVE];
        assert_eq!(from_blob(&to_blob(&vector), 4), Some(vector));
        assert_eq!(from_blob(&[0; 12], 4), None, "a short blob is rejected");
        assert_eq!(from_blob(&[], 0), None);
    }

    #[test]
    fn only_explicit_off_values_disable_semantic_search() {
        for off in ["0", "false", "OFF", " no "] {
            assert!(disabled_by(Some(off)), "{off:?} disables");
        }
        for on in [None, Some("1"), Some("true"), Some(""), Some("yes")] {
            assert!(!disabled_by(on), "{on:?} leaves it on");
        }
    }

    #[test]
    fn the_encoder_file_must_match_its_pinned_size_and_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("encoder.gguf");
        std::fs::write(&path, b"abc").unwrap();
        let abc = sha256_hex(b"abc");
        assert_eq!(verify_file(&path, 3, &abc), Ok(()));
        assert_eq!(verify_file(&path, 4, &abc), Err(Unavailable::Mismatch));
        assert_eq!(
            verify_file(&path, 3, &sha256_hex(b"abd")),
            Err(Unavailable::Mismatch)
        );
        assert_eq!(
            verify_file(&dir.path().join("missing.gguf"), 3, &abc),
            Err(Unavailable::NotInstalled)
        );
    }

    #[tokio::test]
    async fn a_missing_or_foreign_encoder_is_reported_not_loaded() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            encoder(dir.path()).await,
            Err(Unavailable::NotInstalled)
        ));
        std::fs::write(dir.path().join(ENCODER_FILENAME), b"not the pinned encoder").unwrap();
        assert!(matches!(
            encoder(dir.path()).await,
            Err(Unavailable::Mismatch)
        ));
        assert!(
            matches!(encoder(dir.path()).await, Err(Unavailable::Mismatch)),
            "the cached verdict holds while the file is unchanged"
        );
    }

    #[test]
    fn semantic_ranking_scores_only_vectors_bound_to_the_current_chunk_text() {
        let conn = memory();
        let ids = seed(&conn, "doc", TEXT);
        assert!(ids.len() >= 2, "fixture spans several chunks");
        put_vector(&conn, ids[0], &unit(&[1.0, 0.0, 0.0]));
        put_vector(&conn, ids[1], &unit(&[0.6, 0.8, 0.0]));
        let query = unit(&[1.0, 0.1, 0.0]);

        let ranked = semantic_ranked(&conn, &query, None, 10).unwrap();
        assert_eq!(
            ranked.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![ids[0], ids[1]]
        );
        assert!(ranked[0].1 > ranked[1].1);

        conn.execute(
            "UPDATE document_chunks SET chunk_sha256 = 'rebound' WHERE id = ?1",
            params![ids[0]],
        )
        .unwrap();
        let ranked = semantic_ranked(&conn, &query, None, 10).unwrap();
        assert_eq!(
            ranked.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![ids[1]],
            "a vector computed from other text is never scored"
        );
    }

    #[test]
    fn semantic_ranking_ignores_other_encoders_other_sizes_and_other_scopes() {
        let conn = memory();
        let a = seed(&conn, "a", TEXT);
        let b = seed(&conn, "b", TEXT);
        put_vector(&conn, a[0], &unit(&[1.0, 0.0, 0.0]));
        put_vector(&conn, b[0], &unit(&[1.0, 0.0, 0.0]));
        put_vector(&conn, a[1], &unit(&[1.0, 0.0, 0.0, 0.0]));
        conn.execute(
            "UPDATE document_chunk_vectors SET encoder_sha256 = 'other' WHERE chunk_id = ?1",
            params![b[0]],
        )
        .unwrap();
        let query = unit(&[1.0, 0.0, 0.0]);

        assert_eq!(
            semantic_ranked(&conn, &query, None, 10).unwrap(),
            vec![(a[0], 1.0)],
            "another encoder's vector and a vector of another size are never scored"
        );
        put_vector(&conn, b[1], &unit(&[1.0, 0.0, 0.0]));
        let scoped = semantic_ranked(&conn, &query, Some(&["b".to_string()]), 10).unwrap();
        assert_eq!(
            scoped.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![b[1]]
        );
    }

    #[test]
    fn coverage_counts_indexed_skipped_and_pending_chunks() {
        let conn = memory();
        let ids = seed(&conn, "doc", TEXT);
        let total = ids.len() as u64;
        assert_eq!(
            coverage(&conn, None).unwrap(),
            Coverage {
                indexable_chunks: total,
                indexed_chunks: 0,
                skipped_chunks: 0
            }
        );
        put_vector(&conn, ids[0], &[1.0]);
        let hash: String = conn
            .query_row(
                "SELECT chunk_sha256 FROM document_chunks WHERE id = ?1",
                params![ids[1]],
                |row| row.get(0),
            )
            .unwrap();
        store(&conn, ids[1], &hash, 1, None).unwrap();
        let covered = coverage(&conn, Some(&["doc".to_string()])).unwrap();
        assert_eq!((covered.indexed_chunks, covered.skipped_chunks), (1, 1));
        assert_eq!(covered.pending(), total - 2);
        assert_eq!(
            coverage(&conn, Some(&["other".to_string()])).unwrap(),
            Coverage::default()
        );
        assert_eq!(
            coverage_by_document(&conn).unwrap(),
            vec![("doc".to_string(), covered)]
        );
    }

    #[test]
    fn pre_citation_chunks_are_not_indexable() {
        let conn = memory();
        conn.execute(
            "INSERT INTO documents (id, filename, file_type, byte_size, chunk_count, created_at)
             VALUES ('legacy', 'old.txt', 'txt', 5, 1, 1)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO document_chunks (doc_id, chunk_index, content) VALUES ('legacy', 0, 'hello')",
            [],
        )
        .unwrap();
        assert_eq!(coverage(&conn, None).unwrap(), Coverage::default());
        assert!(pending_batch(&conn, i64::MAX, 10).unwrap().is_empty());
    }

    #[test]
    fn pending_batches_walk_newest_first_and_stop_at_the_cursor() {
        let conn = memory();
        let ids = seed(&conn, "doc", TEXT);
        let first = pending_batch(&conn, i64::MAX, 2).unwrap();
        assert_eq!(
            first.iter().map(|chunk| chunk.id).collect::<Vec<_>>(),
            vec![ids[ids.len() - 1], ids[ids.len() - 2]]
        );
        let rest = pending_batch(&conn, first[1].id, 100).unwrap();
        assert_eq!(rest.len(), ids.len() - 2);
        put_vector(&conn, ids[0], &[1.0]);
        assert!(pending_batch(&conn, i64::MAX, 100)
            .unwrap()
            .iter()
            .all(|chunk| chunk.id != ids[0]));
    }

    #[test]
    fn an_upload_after_a_pass_started_is_seen_as_newer() {
        let conn = memory();
        let large = seed(&conn, "large", TEXT);
        let top = *large.last().unwrap();
        assert!(
            !pending_above(&conn, top).unwrap(),
            "nothing newer than the pass's first chunk yet"
        );
        let small = seed(
            &conn,
            "small",
            "A short note that arrives while the large file is indexing.",
        );
        assert!(pending_above(&conn, top).unwrap());
        for id in &small {
            put_vector(&conn, *id, &[1.0]);
        }
        assert!(
            !pending_above(&conn, top).unwrap(),
            "an indexed upload no longer interrupts the pass"
        );
    }

    #[test]
    fn a_vector_is_not_stored_for_a_chunk_that_changed_or_vanished() {
        let conn = memory();
        let ids = seed(&conn, "doc", TEXT);
        store(&conn, ids[0], "stale-hash", 1, Some(&[1.0])).unwrap();
        store(&conn, 999_999, "anything", 1, Some(&[1.0])).unwrap();
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM document_chunk_vectors", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(rows, 0);
    }

    #[test]
    fn deleting_or_reingesting_a_document_drops_its_vectors() {
        let conn = memory();
        let ids = seed(&conn, "doc", TEXT);
        for id in &ids {
            put_vector(&conn, *id, &[1.0]);
        }
        conn.execute("DELETE FROM document_chunks WHERE doc_id = 'doc'", [])
            .unwrap();
        let rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM document_chunk_vectors", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(rows, 0, "vectors cascade with their chunks");
    }

    #[test]
    fn a_restored_chunk_leaves_the_skip_list() {
        let conn = memory();
        let ids = seed(&conn, "doc", TEXT);
        let (content, hash): (String, String) = conn
            .query_row(
                "SELECT content, chunk_sha256 FROM document_chunks WHERE id = ?1",
                params![ids[0]],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        conn.execute(
            "UPDATE document_chunks SET content = 'tampered' WHERE id = ?1",
            params![ids[0]],
        )
        .unwrap();
        store(&conn, ids[0], &hash, 1, None).unwrap();
        clear_recovered_skips(&conn, None).unwrap();
        assert_eq!(
            coverage(&conn, None).unwrap().skipped_chunks,
            1,
            "still tampered"
        );

        conn.execute(
            "UPDATE document_chunks SET content = ?1 WHERE id = ?2",
            params![content, ids[0]],
        )
        .unwrap();
        clear_recovered_skips(&conn, Some(&["other".to_string()])).unwrap();
        assert_eq!(
            coverage(&conn, None).unwrap().skipped_chunks,
            1,
            "a check scoped to other documents leaves it alone"
        );
        clear_recovered_skips(&conn, Some(&["doc".to_string()])).unwrap();
        assert_eq!(coverage(&conn, None).unwrap().skipped_chunks, 0);
        assert!(pending_batch(&conn, i64::MAX, 100)
            .unwrap()
            .iter()
            .any(|chunk| chunk.id == ids[0]));
    }
}
