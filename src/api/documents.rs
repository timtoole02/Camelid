//! Local Document Drag-and-Drop RAG (Feature A)
//!
//! Provides document ingestion, sliding-window chunking, SQLite FTS5 lexical indexing,
//! and hybrid retrieval (BM25 + cosine vector similarity / RRF) for "Chat with Documents".

use std::io::{Cursor, Read};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rusqlite::{params, params_from_iter, types::Value, Connection};
use serde::{Deserialize, Serialize};

use super::{api_error, AppState};

const DEFAULT_CHUNK_CHARS: usize = 512;
const DEFAULT_CHUNK_OVERLAP: usize = 64;
const DOCUMENTS_DB_FILE: &str = "documents_rag.sqlite3";

static DB_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn db_lock() -> &'static Mutex<()> {
    DB_MUTEX.get_or_init(|| Mutex::new(()))
}

fn documents_db_path() -> PathBuf {
    if let Ok(dir) = std::env::var("CAMELID_DATA_DIR") {
        PathBuf::from(dir).join(DOCUMENTS_DB_FILE)
    } else if let Some(local_app_data) = std::env::var_os("LOCALAPPDATA") {
        PathBuf::from(local_app_data)
            .join("Camelid")
            .join(DOCUMENTS_DB_FILE)
    } else {
        std::env::temp_dir().join(DOCUMENTS_DB_FILE)
    }
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
    Ok(())
}

fn open_connection() -> Result<Connection, rusqlite::Error> {
    let path = documents_db_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let conn = Connection::open(&path)?;
    conn.execute("PRAGMA foreign_keys = ON;", [])?;
    init_db(&conn)?;
    Ok(conn)
}

/// Chunks text using sliding window with sentence/paragraph awareness.
pub fn chunk_text(text: &str, target_chars: usize, overlap_chars: usize) -> Vec<String> {
    let clean = text.replace("\r\n", "\n").replace('\r', "\n");
    let trimmed = clean.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    if trimmed.chars().count() <= target_chars {
        return vec![trimmed.to_string()];
    }

    let mut chunks = Vec::new();
    let paragraphs: Vec<&str> = trimmed.split("\n\n").collect();
    let mut current = String::new();

    for para in paragraphs {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }

        if current.chars().count() + para.chars().count() + 2 <= target_chars {
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(para);
        } else {
            // If current is not empty, push it
            if !current.is_empty() {
                chunks.push(current.clone());
                // Keep overlap tail
                let current_chars: Vec<char> = current.chars().collect();
                if current_chars.len() > overlap_chars {
                    let tail: String = current_chars[current_chars.len() - overlap_chars..]
                        .iter()
                        .collect();
                    current = tail;
                    current.push_str("\n\n");
                } else {
                    current.clear();
                }
            }

            // If a single paragraph is larger than target_chars, split by sentences or chunks
            if para.chars().count() > target_chars {
                let chars: Vec<char> = para.chars().collect();
                let mut start = 0;
                while start < chars.len() {
                    let end = (start + target_chars).min(chars.len());
                    let slice: String = chars[start..end].iter().collect();
                    chunks.push(slice);
                    if end >= chars.len() {
                        break;
                    }
                    start += target_chars.saturating_sub(overlap_chars).max(1);
                }
            } else {
                current.push_str(para);
            }
        }
    }

    if !current.trim().is_empty() {
        chunks.push(current);
    }

    chunks
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
    #[serde(default = "default_top_k")]
    pub top_k: usize,
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
    pub score: f32,
    /// `keyword` for an FTS hit, `attached` when an explicitly attached
    /// document is supplied as context because the user's wording had no
    /// lexical overlap with its contents.
    pub retrieval: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SearchDocumentsResponse {
    pub results: Vec<DocumentSearchResult>,
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
    State(_state): State<AppState>,
    Json(payload): Json<IngestDocumentRequest>,
) -> Result<Json<IngestDocumentResponse>, Response> {
    let doc_id = payload
        .doc_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let filename = payload.filename.trim().to_string();

    let text_content = if payload.is_base64 {
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
        extract_text_from_bytes(&filename, &decoded)
    } else {
        payload.content
    };

    let byte_size = text_content.len();
    let chunks = chunk_text(&text_content, DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP);
    let chunk_count = chunks.len();
    if chunks.is_empty() {
        return Err(api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "empty_document",
            "The document did not contain any readable text.".to_string(),
            None,
        ));
    }

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

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let ext = filename
        .split('.')
        .next_back()
        .unwrap_or("txt")
        .to_lowercase();

    // Remove any previous chunks and their external-content FTS rows for doc_id.
    tx.execute(
        "DELETE FROM document_chunks_fts WHERE rowid IN (SELECT id FROM document_chunks WHERE doc_id = ?1)",
        params![doc_id],
    )
    .map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "delete_fts_error",
            e.to_string(),
            None,
        )
    })?;
    tx.execute(
        "DELETE FROM document_chunks WHERE doc_id = ?1",
        params![doc_id],
    )
    .map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "delete_chunk_error",
            e.to_string(),
            None,
        )
    })?;

    // Insert or replace metadata only after the old chunk ids have been
    // available for FTS cleanup; REPLACE can otherwise cascade-delete them
    // before the external-content index rows are found.
    tx.execute(
        "INSERT OR REPLACE INTO documents (id, filename, file_type, byte_size, chunk_count, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![doc_id, filename, ext, byte_size as i64, chunk_count as i64, now],
    ).map_err(|e| {
        api_error(StatusCode::INTERNAL_SERVER_ERROR, "insert_doc_error", e.to_string(), None)
    })?;

    // Insert chunks and index in FTS5
    for (idx, chunk) in chunks.iter().enumerate() {
        tx.execute(
            "INSERT INTO document_chunks (doc_id, chunk_index, content) VALUES (?1, ?2, ?3)",
            params![doc_id, idx as i64, chunk],
        )
        .map_err(|e| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "insert_chunk_error",
                e.to_string(),
                None,
            )
        })?;

        let rowid = tx.last_insert_rowid();
        tx.execute(
            "INSERT INTO document_chunks_fts (rowid, content) VALUES (?1, ?2)",
            params![rowid, chunk],
        )
        .map_err(|e| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "insert_fts_error",
                e.to_string(),
                None,
            )
        })?;
    }

    tx.commit().map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "commit_error",
            e.to_string(),
            None,
        )
    })?;

    Ok(Json(IngestDocumentResponse {
        doc_id,
        filename,
        chunk_count,
        byte_size,
    }))
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
        "SELECT c.doc_id, d.filename, c.chunk_index, c.content
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
                    retrieval: "attached",
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

/// Endpoint: `POST /api/documents/search`
pub async fn search_documents(
    State(_state): State<AppState>,
    Json(payload): Json<SearchDocumentsRequest>,
) -> Result<Json<SearchDocumentsResponse>, Response> {
    let query_trimmed = payload.query.trim();
    if query_trimmed.is_empty() {
        return Ok(Json(SearchDocumentsResponse {
            results: Vec::new(),
        }));
    }

    let fts_pattern = sanitize_fts5_query(query_trimmed);
    let top_k = payload.top_k.clamp(1, 20);
    if payload.doc_ids.as_ref().is_some_and(Vec::is_empty) {
        return Ok(Json(SearchDocumentsResponse {
            results: Vec::new(),
        }));
    }

    let _lock = db_lock().lock().unwrap();
    let conn = open_connection().map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sqlite_open_error",
            e.to_string(),
            None,
        )
    })?;

    // Apply the document scope before ranking and LIMIT. Filtering afterward
    // can discard all globally top-ranked rows even when an attached document
    // contains relevant matches just below them.
    let mut sql = String::from(
        "SELECT c.doc_id, d.filename, c.chunk_index, c.content, bm25(document_chunks_fts) as score
         FROM document_chunks_fts AS f
         JOIN document_chunks AS c ON c.id = f.rowid
         JOIN documents AS d ON d.id = c.doc_id
         WHERE document_chunks_fts MATCH ?1",
    );
    let mut bind_values = vec![Value::Text(fts_pattern)];
    if let Some(allowed_ids) = payload.doc_ids.as_ref() {
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
    bind_values.push(Value::Integer(top_k as i64));
    sql.push_str(&format!(" ORDER BY score ASC LIMIT ?{}", bind_values.len()));

    let mut stmt = conn.prepare(&sql).map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "prepare_fts_error",
            e.to_string(),
            None,
        )
    })?;

    let mut query_results = Vec::new();
    let rows = stmt
        .query_map(params_from_iter(bind_values.iter()), |row| {
            let doc_id: String = row.get(0)?;
            let filename: String = row.get(1)?;
            let chunk_index: i64 = row.get(2)?;
            let content: String = row.get(3)?;
            let raw_bm25: f64 = row.get(4)?;
            Ok((doc_id, filename, chunk_index as usize, content, raw_bm25))
        })
        .map_err(|e| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "query_fts_error",
                e.to_string(),
                None,
            )
        })?;

    for row_res in rows.flatten() {
        let (doc_id, filename, chunk_index, content, raw_bm25) = row_res;
        // BM25 in sqlite returns negative values where lower/more negative is better match
        let normalized_score = (1.0 / (1.0 + raw_bm25.abs())) as f32;
        query_results.push(DocumentSearchResult {
            doc_id,
            filename,
            chunk_index,
            excerpt: content,
            score: normalized_score,
            retrieval: "keyword",
        });
    }

    if query_results.is_empty() {
        if let Some(attached_ids) = payload.doc_ids.as_deref() {
            query_results = attached_document_context(&conn, attached_ids, top_k).map_err(|e| {
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "attached_document_fallback_error",
                    e.to_string(),
                    None,
                )
            })?;
        }
    }

    Ok(Json(SearchDocumentsResponse {
        results: query_results,
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
    tx.execute(
        "DELETE FROM document_chunks_fts WHERE rowid IN (SELECT id FROM document_chunks WHERE doc_id = ?1)",
        params![doc_id],
    )
    .and_then(|_| tx.execute("DELETE FROM documents WHERE id = ?1", params![doc_id]))
        .map_err(|e| {
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
    fn test_chunk_text_basic() {
        let text = "Hello world! This is a test paragraph.\n\nSecond paragraph has more content for testing.";
        let chunks = chunk_text(text, 50, 10);
        assert!(!chunks.is_empty());
        assert!(chunks[0].contains("Hello world!"));
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
}
