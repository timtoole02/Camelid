//! Verifiable citations for the Knowledge Library (F2a).
//!
//! A citation binds `{document sha256, byte range, chunk sha256}`. Resolving one
//! re-derives the span from the stored canonical text and compares hashes. When
//! anything drifted the citation is *refused* rather than rendered, so a stale
//! or tampered source can never be quoted as if it were current.

use std::collections::HashMap;

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::Json;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::documents::{db_lock, open_connection, DocumentSearchResult};
use super::{api_error, AppState};

/// Lowercase hex SHA-256, the single hashing convention for citation binding.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// A chunk plus the byte range it occupies in the canonical document text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSpan {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

impl ChunkSpan {
    pub fn sha256(&self) -> String {
        sha256_hex(self.text.as_bytes())
    }
}

/// Splits `text` into overlapping windows that are **exact byte slices** of the
/// input: `span.text == text[span.start..span.end]` holds for every span. That
/// invariant is what makes a byte range citable; a chunker that reflows or
/// re-joins text produces ranges that cannot be pointed at.
///
/// A window also ends where a new section begins (see `section_starts`), once it
/// holds a quarter of `target_chars`, and the next window starts exactly there
/// with no overlap, so one chunk does not carry the tail of one topic into the
/// next and score below a question about either.
pub fn chunk_text_with_spans(
    text: &str,
    target_chars: usize,
    overlap_chars: usize,
) -> Vec<ChunkSpan> {
    let target = target_chars.max(1);
    let overlap = overlap_chars.min(target - 1);
    let min_section = (target / 4).max(1);

    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let total = chars.len();
    let mut spans: Vec<ChunkSpan> = Vec::new();
    if total == 0 {
        return spans;
    }
    let sections = section_starts(&chars);

    let byte_at = |index: usize| -> usize {
        if index >= total {
            text.len()
        } else {
            chars[index].0
        }
    };

    let mut start_ci = 0usize;
    while start_ci < total {
        // Never start on whitespace: the offset would point at padding.
        while start_ci < total && chars[start_ci].1.is_whitespace() {
            start_ci += 1;
        }
        if start_ci >= total {
            break;
        }

        let hard_end = (start_ci + target).min(total);
        let next_section = sections
            .iter()
            .copied()
            .find(|&section| section >= start_ci + min_section && section <= hard_end);
        let mut end_ci = hard_end;
        if let Some(section) = next_section {
            end_ci = section;
        } else if hard_end < total {
            if let Some(boundary) = preferred_break(&chars, start_ci, hard_end) {
                end_ci = boundary;
            }
        }
        // Captured before trimming: a trailing newline must not hide the fact
        // that this window already consumed the rest of the document.
        let consumed_to_end = end_ci >= total;
        while end_ci > start_ci + 1 && chars[end_ci - 1].1.is_whitespace() {
            end_ci -= 1;
        }

        let start_b = byte_at(start_ci);
        let end_b = byte_at(end_ci);
        if end_b > start_b {
            spans.push(ChunkSpan {
                text: text[start_b..end_b].to_string(),
                start: start_b,
                end: end_b,
            });
        }

        if consumed_to_end {
            break;
        }
        start_ci = match next_section {
            Some(section) => section,
            None => end_ci.saturating_sub(overlap).max(start_ci + 1),
        };
    }

    spans
}

/// Char indices where a section begins: the first character of a heading, which
/// is a paragraph opening on a line after a blank line (or the text's first line)
/// that is either a Markdown heading (one to six `#` and a space), or a single
/// line of at most 80 characters that does not end like a sentence or a clause
/// (`1. Scope`, `Refunds`).
fn section_starts(chars: &[(usize, char)]) -> Vec<usize> {
    let mut lines = Vec::new();
    let mut line_start = 0;
    for (index, (_, ch)) in chars.iter().enumerate() {
        if *ch == '\n' {
            lines.push((line_start, index));
            line_start = index + 1;
        }
    }
    lines.push((line_start, chars.len()));

    let trimmed = |(start, end): (usize, usize)| -> (usize, usize) {
        let mut first = start;
        while first < end && chars[first].1.is_whitespace() {
            first += 1;
        }
        let mut last = end;
        while last > first && chars[last - 1].1.is_whitespace() {
            last -= 1;
        }
        (first, last)
    };
    let blank = |line: (usize, usize)| {
        let (first, last) = trimmed(line);
        first == last
    };

    let mut starts = Vec::new();
    for (index, &line) in lines.iter().enumerate() {
        if blank(line) || (index > 0 && !blank(lines[index - 1])) {
            continue;
        }
        let (first, last) = trimmed(line);
        let hashes = chars[first..last]
            .iter()
            .take_while(|(_, ch)| *ch == '#')
            .count();
        let markdown = (1..=6).contains(&hashes)
            && chars
                .get(first + hashes)
                .is_some_and(|(_, ch)| *ch == ' ' || *ch == '\t');
        let single_line = lines.get(index + 1).is_none_or(|&next| blank(next));
        let ends_like_prose = matches!(chars[last - 1].1, '.' | '!' | '?' | ':' | ';' | ',');
        if markdown || (single_line && last - first <= 80 && !ends_like_prose) {
            starts.push(first);
        }
    }
    starts
}

/// Prefers a paragraph break, then a sentence end, then a word boundary in the
/// back half of the window so a chunk ends where a reader would expect.
fn preferred_break(chars: &[(usize, char)], start_ci: usize, hard_end: usize) -> Option<usize> {
    let floor = start_ci + (hard_end - start_ci) / 2;

    let mut index = hard_end;
    while index > floor + 1 {
        if chars[index - 1].1 == '\n' && chars[index - 2].1 == '\n' {
            return Some(index);
        }
        index -= 1;
    }

    let mut index = hard_end;
    while index > floor + 1 {
        let current = chars[index - 1].1;
        let next_is_space = chars
            .get(index)
            .map(|(_, c)| c.is_whitespace())
            .unwrap_or(true);
        if matches!(current, '.' | '!' | '?') && next_is_space {
            return Some(index);
        }
        index -= 1;
    }

    let mut index = hard_end;
    while index > floor {
        if chars[index - 1].1.is_whitespace() {
            return Some(index);
        }
        index -= 1;
    }

    None
}

#[derive(Debug, Deserialize)]
pub struct ResolveCitationRequest {
    pub doc_id: String,
    pub chunk_index: usize,
    /// Hash the caller was shown. Supplying it detects drift between the
    /// rendered answer and the stored chunk.
    #[serde(default)]
    pub chunk_sha256: Option<String>,
    #[serde(default)]
    pub doc_sha256: Option<String>,
    #[serde(default = "default_context_chars")]
    pub context_chars: usize,
}

fn default_context_chars() -> usize {
    280
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct ResolveCitationResponse {
    pub doc_id: String,
    pub filename: String,
    pub chunk_index: usize,
    pub byte_start: usize,
    pub byte_end: usize,
    pub chunk_sha256: String,
    pub doc_sha256: String,
    /// Text immediately preceding the span, for context in the source viewer.
    pub before: String,
    /// The exact cited bytes.
    pub span: String,
    pub after: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CitationOutcome {
    Resolved(Box<ResolveCitationResponse>),
    Refused { code: &'static str, message: String },
}

fn refuse(code: &'static str, message: impl Into<String>) -> CitationOutcome {
    CitationOutcome::Refused {
        code,
        message: message.into(),
    }
}

/// A document whose stored source still hashes to its recorded digest.
struct VerifiedSource {
    filename: String,
    doc_sha256: String,
    text: String,
}

enum SourceState {
    Verified(VerifiedSource),
    /// Ingested before F2a: no citation binding exists anywhere on the document.
    Legacy,
    Refused {
        code: &'static str,
        message: &'static str,
    },
}

const LEGACY_MESSAGE: &str =
    "This document predates verifiable citations. Re-ingest it to cite from it.";

/// Loads a document and verifies its stored source once, for every chunk cited from it.
fn load_source(
    conn: &Connection,
    doc_id: &str,
    expected_doc_sha256: Option<&str>,
) -> Result<SourceState, rusqlite::Error> {
    let document = conn
        .query_row(
            "SELECT filename, source_sha256, text_sha256, source_text FROM documents WHERE id = ?1",
            params![doc_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            },
        )
        .optional()?;

    let Some((filename, source_sha, stored_doc_sha, source_text)) = document else {
        return Ok(SourceState::Refused {
            code: "citation_unknown_document",
            message: "The cited document is no longer in the library.",
        });
    };

    // A partial binding is damage, not age: only a document with none of it is legacy.
    let (stored_doc_sha, text) =
        match (source_sha, stored_doc_sha, source_text) {
            (None, None, None) => return Ok(SourceState::Legacy),
            (_, Some(stored_doc_sha), Some(text)) => (stored_doc_sha, text),
            _ => return Ok(SourceState::Refused {
                code: "citation_unverifiable",
                message:
                    "This document's citation record is incomplete. Re-ingest it to cite from it.",
            }),
        };

    // A document re-ingested with different content gets a new text hash, so a
    // citation minted against the old content can no longer be resolved.
    if expected_doc_sha256.is_some_and(|expected| expected != stored_doc_sha) {
        return Ok(SourceState::Refused {
            code: "citation_document_changed",
            message: "The source document changed after this citation was created.",
        });
    }

    // The stored text is the canonical source. If it no longer hashes to the
    // recorded digest the row was tampered with underneath us.
    if sha256_hex(text.as_bytes()) != stored_doc_sha {
        return Ok(SourceState::Refused {
            code: "citation_source_corrupted",
            message: "The stored source text no longer matches its recorded hash.",
        });
    }

    Ok(SourceState::Verified(VerifiedSource {
        filename,
        doc_sha256: stored_doc_sha,
        text,
    }))
}

fn resolve_chunk(
    conn: &Connection,
    doc_id: &str,
    source: &VerifiedSource,
    chunk_index: usize,
    expected_chunk_sha256: Option<&str>,
    context_chars: usize,
) -> Result<CitationOutcome, rusqlite::Error> {
    let chunk = conn
        .query_row(
            "SELECT byte_start, byte_end, chunk_sha256 FROM document_chunks
             WHERE doc_id = ?1 AND chunk_index = ?2",
            params![doc_id, chunk_index as i64],
            |row| {
                Ok((
                    row.get::<_, Option<i64>>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            },
        )
        .optional()?;

    let Some((byte_start, byte_end, stored_chunk_sha)) = chunk else {
        return Ok(refuse(
            "citation_unknown_chunk",
            "The cited passage is no longer part of this document.",
        ));
    };

    let (Some(byte_start), Some(byte_end), Some(stored_chunk_sha)) =
        (byte_start, byte_end, stored_chunk_sha)
    else {
        return Ok(refuse(
            "citation_unverifiable",
            "This passage predates verifiable citations. Re-ingest the document to cite from it.",
        ));
    };

    let text = source.text.as_str();
    let start = byte_start.max(0) as usize;
    let end = byte_end.max(0) as usize;
    if start >= end || end > text.len() {
        return Ok(refuse(
            "citation_range_invalid",
            "The cited byte range falls outside the current source.",
        ));
    }
    if !text.is_char_boundary(start) || !text.is_char_boundary(end) {
        return Ok(refuse(
            "citation_range_invalid",
            "The cited byte range does not land on character boundaries.",
        ));
    }

    let span = &text[start..end];
    let recomputed = sha256_hex(span.as_bytes());
    if recomputed != stored_chunk_sha {
        return Ok(refuse(
            "citation_chunk_mismatch",
            "The cited passage no longer matches the text at that location.",
        ));
    }
    if expected_chunk_sha256.is_some_and(|expected| expected != recomputed) {
        return Ok(refuse(
            "citation_chunk_mismatch",
            "The quoted passage does not match the stored source.",
        ));
    }

    let budget = context_chars.min(2_000);
    Ok(CitationOutcome::Resolved(Box::new(
        ResolveCitationResponse {
            doc_id: doc_id.to_string(),
            filename: source.filename.clone(),
            chunk_index,
            byte_start: start,
            byte_end: end,
            chunk_sha256: recomputed,
            doc_sha256: source.doc_sha256.clone(),
            before: context_before(text, start, budget),
            span: span.to_string(),
            after: context_after(text, end, budget),
        },
    )))
}

/// Re-derives a cited span and verifies every binding. Pure over a connection so
/// it can be tested without a server.
pub fn resolve_citation_with(
    conn: &Connection,
    request: &ResolveCitationRequest,
) -> Result<CitationOutcome, rusqlite::Error> {
    match load_source(conn, &request.doc_id, request.doc_sha256.as_deref())? {
        SourceState::Verified(source) => resolve_chunk(
            conn,
            &request.doc_id,
            &source,
            request.chunk_index,
            request.chunk_sha256.as_deref(),
            request.context_chars,
        ),
        SourceState::Legacy => Ok(refuse("citation_unverifiable", LEGACY_MESSAGE)),
        SourceState::Refused { code, message } => Ok(refuse(code, message)),
    }
}

fn context_before(text: &str, start: usize, budget: usize) -> String {
    let mut begin = start.saturating_sub(budget);
    while begin < start && !text.is_char_boundary(begin) {
        begin += 1;
    }
    text[begin..start].to_string()
}

fn context_after(text: &str, end: usize, budget: usize) -> String {
    let mut stop = (end + budget).min(text.len());
    while stop > end && !text.is_char_boundary(stop) {
        stop -= 1;
    }
    text[end..stop].to_string()
}

/// Keeps only results whose citation resolves to exactly the served excerpt. A result
/// from a document with no citation binding at all (pre-F2a) passes through unchanged.
pub fn retain_verifiable(
    conn: &Connection,
    results: Vec<DocumentSearchResult>,
) -> Result<Vec<DocumentSearchResult>, rusqlite::Error> {
    let mut sources: HashMap<String, SourceState> = HashMap::new();
    let mut kept = Vec::with_capacity(results.len());
    for result in results {
        if !sources.contains_key(&result.doc_id) {
            let state = load_source(conn, &result.doc_id, result.doc_sha256.as_deref())?;
            sources.insert(result.doc_id.clone(), state);
        }
        let verdict = match &sources[&result.doc_id] {
            SourceState::Verified(source) => match resolve_chunk(
                conn,
                &result.doc_id,
                source,
                result.chunk_index,
                result.chunk_sha256.as_deref(),
                0,
            )? {
                CitationOutcome::Resolved(resolved) if resolved.span == result.excerpt => Ok(()),
                CitationOutcome::Resolved(_) => Err("citation_excerpt_drifted"),
                CitationOutcome::Refused { code, .. } => Err(code),
            },
            SourceState::Legacy
                if result.chunk_sha256.is_none()
                    && result.byte_start.is_none()
                    && result.byte_end.is_none() =>
            {
                Ok(())
            }
            SourceState::Legacy => Err("citation_unverifiable"),
            SourceState::Refused { code, .. } => Err(*code),
        };
        match verdict {
            Ok(()) => kept.push(result),
            Err(code) => tracing::warn!(
                doc_id = %result.doc_id,
                chunk_index = result.chunk_index,
                code,
                "withheld a search result that does not resolve to its excerpt"
            ),
        }
    }
    Ok(kept)
}

/// Endpoint: `POST /api/documents/citation/resolve`
pub async fn resolve_citation(
    State(_state): State<AppState>,
    Json(payload): Json<ResolveCitationRequest>,
) -> Result<Json<ResolveCitationResponse>, Response> {
    let _lock = db_lock().lock().unwrap();
    let conn = open_connection().map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sqlite_open_error",
            e.to_string(),
            None,
        )
    })?;

    let outcome = resolve_citation_with(&conn, &payload).map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "citation_query_error",
            e.to_string(),
            None,
        )
    })?;

    match outcome {
        CitationOutcome::Resolved(response) => Ok(Json(*response)),
        CitationOutcome::Refused { code, message } => {
            Err(api_error(StatusCode::CONFLICT, code, message, None))
        }
    }
}

/// A document's whole stored text, for the in-chat document viewer.
#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct DocumentSourceResponse {
    pub doc_id: String,
    pub filename: String,
    pub doc_sha256: String,
    pub text: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum DocumentSourceOutcome {
    Shown(DocumentSourceResponse),
    Refused {
        status: StatusCode,
        code: &'static str,
        message: &'static str,
    },
}

/// Shows a document only while its stored text still hashes to the recorded
/// digest: the same bar a citation from it must clear.
pub fn document_source_with(
    conn: &Connection,
    doc_id: &str,
) -> Result<DocumentSourceOutcome, rusqlite::Error> {
    let refused = |status, code, message| DocumentSourceOutcome::Refused {
        status,
        code,
        message,
    };
    Ok(match load_source(conn, doc_id, None)? {
        SourceState::Verified(source) => DocumentSourceOutcome::Shown(DocumentSourceResponse {
            doc_id: doc_id.to_string(),
            filename: source.filename,
            doc_sha256: source.doc_sha256,
            text: source.text,
        }),
        SourceState::Legacy => refused(
            StatusCode::CONFLICT,
            "document_source_unavailable",
            "This document was indexed before Camelid kept its source text. Attach it again to view it.",
        ),
        SourceState::Refused {
            code: "citation_unknown_document",
            ..
        } => refused(
            StatusCode::NOT_FOUND,
            "document_not_found",
            "This document is no longer in the library.",
        ),
        SourceState::Refused {
            code: "citation_source_corrupted",
            ..
        } => refused(
            StatusCode::CONFLICT,
            "document_source_corrupted",
            "The stored text no longer matches its recorded hash, so it is not shown.",
        ),
        SourceState::Refused { .. } => refused(
            StatusCode::CONFLICT,
            "document_source_incomplete",
            "This document's source record is incomplete. Attach it again to view it.",
        ),
    })
}

/// Endpoint: `GET /api/documents/:id/source`
pub async fn document_source(
    State(_state): State<AppState>,
    AxumPath(doc_id): AxumPath<String>,
) -> Result<Json<DocumentSourceResponse>, Response> {
    let _lock = db_lock().lock().unwrap();
    let conn = open_connection().map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "sqlite_open_error",
            e.to_string(),
            None,
        )
    })?;

    let outcome = document_source_with(&conn, &doc_id).map_err(|e| {
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "document_source_query_error",
            e.to_string(),
            None,
        )
    })?;

    match outcome {
        DocumentSourceOutcome::Shown(source) => Ok(Json(source)),
        DocumentSourceOutcome::Refused {
            status,
            code,
            message,
        } => Err(api_error(status, code, message.to_string(), None)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::documents::init_db;

    const SAMPLE: &str = "Enterprise customers may request a refund within 60 days of the invoice date.\n\nRefunds are issued to the original payment method. Processing takes five business days.\n\nTrial accounts are not eligible for refunds under any circumstance.";

    fn seed(conn: &Connection, doc_id: &str, text: &str) -> Vec<ChunkSpan> {
        let spans = chunk_text_with_spans(text, 120, 24);
        conn.execute(
            "INSERT OR REPLACE INTO documents
             (id, filename, file_type, byte_size, chunk_count, created_at, source_sha256, text_sha256, source_text)
             VALUES (?1, 'terms.txt', 'txt', ?2, ?3, 1, ?4, ?5, ?6)",
            params![
                doc_id,
                text.len() as i64,
                spans.len() as i64,
                sha256_hex(text.as_bytes()),
                sha256_hex(text.as_bytes()),
                text,
            ],
        )
        .unwrap();
        for (index, span) in spans.iter().enumerate() {
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
        }
        spans
    }

    fn request(doc_id: &str, chunk_index: usize) -> ResolveCitationRequest {
        ResolveCitationRequest {
            doc_id: doc_id.to_string(),
            chunk_index,
            chunk_sha256: None,
            doc_sha256: None,
            context_chars: 40,
        }
    }

    fn results_for(conn: &Connection, doc_id: &str) -> Vec<DocumentSearchResult> {
        let mut statement = conn
            .prepare(
                "SELECT c.chunk_index, c.content, c.byte_start, c.byte_end, c.chunk_sha256,
                        d.text_sha256
                 FROM document_chunks AS c JOIN documents AS d ON d.id = c.doc_id
                 WHERE c.doc_id = ?1 ORDER BY c.chunk_index",
            )
            .unwrap();
        statement
            .query_map(params![doc_id], |row| {
                Ok(DocumentSearchResult {
                    doc_id: doc_id.to_string(),
                    filename: "terms.txt".to_string(),
                    chunk_index: row.get::<_, i64>(0)? as usize,
                    excerpt: row.get(1)?,
                    score: 0.0,
                    byte_start: row.get::<_, Option<i64>>(2)?.map(|value| value as usize),
                    byte_end: row.get::<_, Option<i64>>(3)?.map(|value| value as usize),
                    chunk_sha256: row.get(4)?,
                    doc_sha256: row.get(5)?,
                    retrieval: "keyword",
                    similarity: None,
                })
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    fn served(results: &[DocumentSearchResult]) -> Vec<(&str, usize)> {
        results
            .iter()
            .map(|result| (result.doc_id.as_str(), result.chunk_index))
            .collect()
    }

    #[test]
    fn search_serves_every_intact_result() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        let spans = seed(&conn, "doc", SAMPLE);
        assert!(spans.len() > 1, "fixture must span several chunks");

        let results = results_for(&conn, "doc");
        let kept = retain_verifiable(&conn, results.clone()).unwrap();
        assert_eq!(served(&kept), served(&results));
    }

    #[test]
    fn search_withholds_an_excerpt_that_drifted_from_its_source() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        let spans = seed(&conn, "doc", SAMPLE);
        conn.execute(
            "UPDATE document_chunks SET content = REPLACE(content, '60 days', '90 days')
             WHERE doc_id = 'doc' AND chunk_index = 0",
            [],
        )
        .unwrap();

        let kept = retain_verifiable(&conn, results_for(&conn, "doc")).unwrap();
        let expected: Vec<_> = (1..spans.len()).map(|index| ("doc", index)).collect();
        assert_eq!(served(&kept), expected);
        assert!(kept
            .iter()
            .all(|result| !result.excerpt.contains("90 days")));
    }

    #[test]
    fn search_withholds_an_excerpt_rehashed_to_match_its_drift() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        seed(&conn, "doc", SAMPLE);
        let drifted = conn
            .query_row(
                "SELECT content FROM document_chunks WHERE doc_id = 'doc' AND chunk_index = 0",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap()
            .replace("60 days", "90 days");
        conn.execute(
            "UPDATE document_chunks SET content = ?1, chunk_sha256 = ?2
             WHERE doc_id = 'doc' AND chunk_index = 0",
            params![drifted, sha256_hex(drifted.as_bytes())],
        )
        .unwrap();

        let kept = retain_verifiable(&conn, results_for(&conn, "doc")).unwrap();
        assert!(kept.iter().all(|result| result.chunk_index != 0));
    }

    #[test]
    fn search_withholds_every_result_of_a_corrupted_document_only() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        seed(&conn, "corrupted", SAMPLE);
        let healthy = seed(&conn, "healthy", SAMPLE);
        conn.execute(
            "UPDATE documents SET source_text = REPLACE(source_text, '60 days', '90 days')
             WHERE id = 'corrupted'",
            [],
        )
        .unwrap();

        let mut results = results_for(&conn, "corrupted");
        results.extend(results_for(&conn, "healthy"));
        let kept = retain_verifiable(&conn, results).unwrap();
        let expected: Vec<_> = (0..healthy.len()).map(|index| ("healthy", index)).collect();
        assert_eq!(served(&kept), expected);
    }

    #[test]
    fn search_passes_rows_without_a_citation_spine_through() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
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

        let kept = retain_verifiable(&conn, results_for(&conn, "legacy")).unwrap();
        assert_eq!(served(&kept), vec![("legacy", 0)]);
    }

    #[test]
    fn search_withholds_a_chunk_whose_hash_was_erased_from_a_verified_document() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        let spans = seed(&conn, "doc", SAMPLE);
        conn.execute(
            "UPDATE document_chunks SET chunk_sha256 = NULL WHERE doc_id = 'doc' AND chunk_index = 0",
            [],
        )
        .unwrap();

        let kept = retain_verifiable(&conn, results_for(&conn, "doc")).unwrap();
        let expected: Vec<_> = (1..spans.len()).map(|index| ("doc", index)).collect();
        assert_eq!(served(&kept), expected);
    }

    #[test]
    fn search_withholds_a_chunk_stripped_of_its_whole_binding() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        let spans = seed(&conn, "doc", SAMPLE);
        conn.execute(
            "UPDATE document_chunks SET chunk_sha256 = NULL, byte_start = NULL, byte_end = NULL
             WHERE doc_id = 'doc' AND chunk_index = 0",
            [],
        )
        .unwrap();

        let kept = retain_verifiable(&conn, results_for(&conn, "doc")).unwrap();
        let expected: Vec<_> = (1..spans.len()).map(|index| ("doc", index)).collect();
        assert_eq!(served(&kept), expected);
    }

    #[test]
    fn a_document_with_its_hash_erased_is_damaged_not_legacy() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        seed(&conn, "doc", SAMPLE);
        conn.execute(
            "UPDATE documents SET text_sha256 = NULL WHERE id = 'doc'",
            [],
        )
        .unwrap();

        let kept = retain_verifiable(&conn, results_for(&conn, "doc")).unwrap();
        assert!(kept.is_empty(), "served {:?}", served(&kept));
        match resolve_citation_with(&conn, &request("doc", 0)).unwrap() {
            CitationOutcome::Refused { code, .. } => assert_eq!(code, "citation_unverifiable"),
            CitationOutcome::Resolved(_) => panic!("a damaged record must not resolve"),
        }
    }

    #[test]
    fn search_verifies_a_mix_of_documents_independently() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        seed(&conn, "corrupted", SAMPLE);
        let healthy = seed(&conn, "healthy", SAMPLE);
        conn.execute(
            "UPDATE documents SET source_text = REPLACE(source_text, '60 days', '90 days')
             WHERE id = 'corrupted'",
            [],
        )
        .unwrap();

        // Interleaved results exercise the per-document cache in both states.
        let corrupted = results_for(&conn, "corrupted");
        let mut mixed = Vec::new();
        for (index, result) in results_for(&conn, "healthy").into_iter().enumerate() {
            mixed.push(result);
            if let Some(bad) = corrupted.get(index) {
                mixed.push(bad.clone());
            }
        }
        let kept = retain_verifiable(&conn, mixed).unwrap();
        let expected: Vec<_> = (0..healthy.len()).map(|index| ("healthy", index)).collect();
        assert_eq!(served(&kept), expected);
    }

    #[test]
    fn spans_are_exact_byte_slices_of_the_source() {
        let inputs = [
            SAMPLE,
            "short",
            "",
            "   leading and trailing whitespace   ",
            "no-spaces-at-all-just-one-very-long-token-that-exceeds-the-target-window-size-by-a-lot",
            "Ünïcödé — naïve café résumé. Ünïcödé — naïve café résumé. Ünïcödé — naïve café résumé.",
            "a\n\nb\n\nc\n\nd\n\ne\n\nf\n\ng\n\nh\n\ni\n\nj\n\nk\n\nl\n\nm\n\nn",
        ];
        for input in inputs {
            for (target, overlap) in [(16usize, 4usize), (64, 16), (120, 24), (1, 0)] {
                for span in chunk_text_with_spans(input, target, overlap) {
                    assert_eq!(
                        span.text,
                        &input[span.start..span.end],
                        "span text diverged from its byte range for input {input:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn spans_cover_the_document_and_advance() {
        let spans = chunk_text_with_spans(SAMPLE, 120, 24);
        assert!(spans.len() > 1, "sample should split into several chunks");
        for pair in spans.windows(2) {
            assert!(
                pair[1].start > pair[0].start,
                "chunker must make forward progress"
            );
        }
        assert!(spans.first().unwrap().start == 0);
        assert!(spans.last().unwrap().end == SAMPLE.len());
    }

    #[test]
    fn trailing_whitespace_does_not_cause_runaway_chunking() {
        // A 491-char document that ends in a newline previously produced 55
        // near-duplicate chunks because the end-of-document break never fired.
        let body = "Enterprise customers may request a refund within 60 days of the invoice date. \
Refunds are issued to the original payment method and processing takes five business days.\n\n\
Trial accounts are not eligible for refunds under any circumstance.\n";
        assert!(body.chars().count() < 512, "fixture must fit in one window");
        let spans = chunk_text_with_spans(body, 512, 64);
        assert_eq!(
            spans.len(),
            1,
            "expected a single chunk, got {}",
            spans.len()
        );
    }

    #[test]
    fn documents_ending_in_whitespace_chunk_like_trimmed_ones() {
        let base = "First paragraph of the policy.\n\nSecond paragraph of the policy.";
        for suffix in ["", "\n", "\n\n", "   ", "\n   \n"] {
            let text = format!("{base}{suffix}");
            let spans = chunk_text_with_spans(&text, 512, 64);
            assert_eq!(
                spans.len(),
                1,
                "suffix {suffix:?} produced {} chunks",
                spans.len()
            );
            for span in &spans {
                assert_eq!(span.text, text[span.start..span.end]);
            }
        }
    }

    #[test]
    fn long_documents_chunk_a_bounded_number_of_times() {
        // Guards the advance invariant: chunk count must stay proportional to
        // length, not explode the way the runaway loop did.
        let para = "This sentence exists to give the chunker realistic prose to split on. ";
        let text = format!("{}\n", para.repeat(40));
        let spans = chunk_text_with_spans(&text, 512, 64);
        let upper_bound = text.chars().count() / (512 - 64) + 2;
        assert!(
            spans.len() <= upper_bound,
            "expected <= {upper_bound} chunks, got {}",
            spans.len()
        );
        for span in &spans {
            assert_eq!(span.text, text[span.start..span.end]);
        }
    }

    #[test]
    fn spans_never_begin_or_end_on_whitespace() {
        for span in chunk_text_with_spans(SAMPLE, 80, 20) {
            assert!(!span.text.starts_with(char::is_whitespace));
            assert!(!span.text.ends_with(char::is_whitespace));
        }
    }

    const SECTIONED: &str = "Support Policy\n\n\
Version 4.2, effective 1 March 2026. This policy applies to every paid plan; where an order form says something different, the order form wins.\n\n\
1. Refunds\n\n\
Enterprise customers may request a refund within 60 days of the invoice date. Refunds are issued to the original payment method within five business days.\n\n\
2. Data export\n\n\
Customers can export their data at any time from the admin console. A full export requested in writing is fulfilled within thirty days.\n\n\
3. Incident handling\n\n\
Security incidents that affect customer data are reported to the account owner within seventy-two hours of confirmation.\n";

    #[test]
    fn each_section_gets_its_own_chunk_without_overlap() {
        let spans = chunk_text_with_spans(SECTIONED, 512, 64);
        let headings: Vec<&str> = spans
            .iter()
            .map(|span| span.text.lines().next().unwrap())
            .collect();
        assert_eq!(
            headings,
            [
                "Support Policy",
                "1. Refunds",
                "2. Data export",
                "3. Incident handling"
            ]
        );
        for span in &spans {
            assert_eq!(span.text, &SECTIONED[span.start..span.end]);
        }
        for pair in spans.windows(2) {
            assert!(
                pair[1].start >= pair[0].end,
                "no chunk repeats the end of the previous section"
            );
        }
    }

    #[test]
    fn sections_shorter_than_a_quarter_window_share_a_chunk() {
        let text = "Notes\n\nShort one.\n\nA\n\nAnother short one.\n\nB\n\nThird short one here.\n";
        assert_eq!(chunk_text_with_spans(text, 512, 64).len(), 1);
        let small: Vec<String> = chunk_text_with_spans(text, 64, 8)
            .into_iter()
            .map(|span| span.text)
            .collect();
        assert_eq!(
            small,
            [
                "Notes\n\nShort one.",
                "A\n\nAnother short one.",
                "B\n\nThird short one here."
            ]
        );
    }

    #[test]
    fn prose_without_headings_still_overlaps() {
        let para = "This sentence exists to give the chunker realistic prose to split on. ";
        let text = para.repeat(20);
        let spans = chunk_text_with_spans(&text, 512, 64);
        assert!(spans.len() > 1);
        for pair in spans.windows(2) {
            assert!(
                pair[1].start < pair[0].end,
                "windows inside a section overlap"
            );
        }
    }

    #[test]
    fn headings_are_short_lines_or_markdown_after_a_blank_line() {
        let text = "Title\n\n\
Prose that ends like a sentence.\n\n\
A line longer than eighty characters that carries no terminal punctuation at all here\n\n\
# Markdown heading\nwith body text directly below\n\n\
#[derive(Debug)]\nstruct NotAHeading;\n\n\
#hashtag, not a heading\nwith a second line\n\n\
####### seven hashes\nwith a second line\n\n\
- a list item\n- another item\n\n\
Section two\n\n\
Mid-paragraph line\nThat is not a heading";
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let starts: Vec<&str> = section_starts(&chars)
            .into_iter()
            .map(|index| text[chars[index].0..].lines().next().unwrap())
            .collect();
        assert_eq!(starts, ["Title", "# Markdown heading", "Section two"]);
    }

    #[test]
    fn resolve_returns_the_exact_cited_span() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        let spans = seed(&conn, "doc-1", SAMPLE);

        let outcome = resolve_citation_with(&conn, &request("doc-1", 0)).unwrap();
        let CitationOutcome::Resolved(resolved) = outcome else {
            panic!("expected the citation to resolve");
        };
        assert_eq!(resolved.span, spans[0].text);
        assert_eq!(resolved.byte_start, spans[0].start);
        assert_eq!(resolved.byte_end, spans[0].end);
        assert_eq!(resolved.chunk_sha256, spans[0].sha256());
        assert_eq!(
            &SAMPLE[resolved.byte_start..resolved.byte_end],
            resolved.span
        );
    }

    #[test]
    fn resolve_refuses_when_the_stored_text_is_tampered_with() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        seed(&conn, "doc-1", SAMPLE);

        let corrupted = SAMPLE.replace("60 days", "30 days");
        assert_eq!(
            corrupted.len(),
            SAMPLE.len(),
            "same length keeps ranges valid"
        );
        conn.execute(
            "UPDATE documents SET source_text = ?1 WHERE id = 'doc-1'",
            params![corrupted],
        )
        .unwrap();

        let outcome = resolve_citation_with(&conn, &request("doc-1", 0)).unwrap();
        match outcome {
            CitationOutcome::Refused { code, .. } => {
                assert_eq!(code, "citation_source_corrupted")
            }
            CitationOutcome::Resolved(_) => panic!("a tampered source must refuse"),
        }
    }

    #[test]
    fn resolve_refuses_when_the_chunk_hash_no_longer_matches() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        seed(&conn, "doc-1", SAMPLE);
        conn.execute(
            "UPDATE document_chunks SET chunk_sha256 = ?1 WHERE doc_id = 'doc-1' AND chunk_index = 0",
            params![sha256_hex(b"something else")],
        )
        .unwrap();

        match resolve_citation_with(&conn, &request("doc-1", 0)).unwrap() {
            CitationOutcome::Refused { code, .. } => assert_eq!(code, "citation_chunk_mismatch"),
            CitationOutcome::Resolved(_) => panic!("a drifted chunk must refuse"),
        }
    }

    #[test]
    fn resolve_refuses_a_caller_hash_that_does_not_match() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        seed(&conn, "doc-1", SAMPLE);

        let mut req = request("doc-1", 0);
        req.chunk_sha256 = Some(sha256_hex(b"quote the model invented"));
        match resolve_citation_with(&conn, &req).unwrap() {
            CitationOutcome::Refused { code, .. } => assert_eq!(code, "citation_chunk_mismatch"),
            CitationOutcome::Resolved(_) => panic!("an unmatched quote must refuse"),
        }
    }

    #[test]
    fn resolve_refuses_when_the_document_was_reingested_with_new_content() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        seed(&conn, "doc-1", SAMPLE);
        let original_doc_sha = sha256_hex(SAMPLE.as_bytes());

        conn.execute("DELETE FROM document_chunks WHERE doc_id = 'doc-1'", [])
            .unwrap();
        seed(
            &conn,
            "doc-1",
            "A completely different set of terms now applies to everyone.",
        );

        let mut req = request("doc-1", 0);
        req.doc_sha256 = Some(original_doc_sha);
        match resolve_citation_with(&conn, &req).unwrap() {
            CitationOutcome::Refused { code, .. } => assert_eq!(code, "citation_document_changed"),
            CitationOutcome::Resolved(_) => {
                panic!("a re-ingested document must refuse old citations")
            }
        }
    }

    #[test]
    fn resolve_refuses_an_out_of_range_byte_span() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        seed(&conn, "doc-1", SAMPLE);
        conn.execute(
            "UPDATE document_chunks SET byte_end = 999999 WHERE doc_id = 'doc-1' AND chunk_index = 0",
            [],
        )
        .unwrap();

        match resolve_citation_with(&conn, &request("doc-1", 0)).unwrap() {
            CitationOutcome::Refused { code, .. } => assert_eq!(code, "citation_range_invalid"),
            CitationOutcome::Resolved(_) => panic!("an out-of-range span must refuse"),
        }
    }

    #[test]
    fn resolve_refuses_unknown_documents_and_chunks() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        seed(&conn, "doc-1", SAMPLE);

        match resolve_citation_with(&conn, &request("missing", 0)).unwrap() {
            CitationOutcome::Refused { code, .. } => assert_eq!(code, "citation_unknown_document"),
            CitationOutcome::Resolved(_) => panic!("unknown document must refuse"),
        }
        match resolve_citation_with(&conn, &request("doc-1", 999)).unwrap() {
            CitationOutcome::Refused { code, .. } => assert_eq!(code, "citation_unknown_chunk"),
            CitationOutcome::Resolved(_) => panic!("unknown chunk must refuse"),
        }
    }

    #[test]
    fn resolve_refuses_legacy_rows_without_a_citation_spine() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        conn.execute(
            "INSERT INTO documents (id, filename, file_type, byte_size, chunk_count, created_at)
             VALUES ('legacy', 'old.txt', 'txt', 10, 1, 1)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO document_chunks (doc_id, chunk_index, content) VALUES ('legacy', 0, 'text')",
            [],
        )
        .unwrap();

        match resolve_citation_with(&conn, &request("legacy", 0)).unwrap() {
            CitationOutcome::Refused { code, .. } => assert_eq!(code, "citation_unverifiable"),
            CitationOutcome::Resolved(_) => panic!("legacy rows cannot be verified"),
        }
    }

    fn source_refusal(conn: &Connection, doc_id: &str) -> (StatusCode, &'static str) {
        match document_source_with(conn, doc_id).unwrap() {
            DocumentSourceOutcome::Refused { status, code, .. } => (status, code),
            DocumentSourceOutcome::Shown(source) => {
                panic!(
                    "{doc_id} must not be shown, got {} bytes",
                    source.text.len()
                )
            }
        }
    }

    #[test]
    fn document_source_shows_the_whole_verified_text() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        seed(&conn, "doc", SAMPLE);

        match document_source_with(&conn, "doc").unwrap() {
            DocumentSourceOutcome::Shown(source) => {
                assert_eq!(source.doc_id, "doc");
                assert_eq!(source.filename, "terms.txt");
                assert_eq!(source.text, SAMPLE);
                assert_eq!(source.doc_sha256, sha256_hex(SAMPLE.as_bytes()));
            }
            other => panic!("an intact document must be shown, got {other:?}"),
        }
    }

    #[test]
    fn document_source_refuses_a_tampered_text_instead_of_showing_it() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        seed(&conn, "doc", SAMPLE);
        conn.execute(
            "UPDATE documents SET source_text = REPLACE(source_text, '60 days', '90 days') WHERE id = 'doc'",
            [],
        )
        .unwrap();

        assert_eq!(
            source_refusal(&conn, "doc"),
            (StatusCode::CONFLICT, "document_source_corrupted")
        );
    }

    #[test]
    fn document_source_names_why_it_cannot_show_a_document() {
        let conn = Connection::open_in_memory().unwrap();
        init_db(&conn).unwrap();
        assert_eq!(
            source_refusal(&conn, "missing"),
            (StatusCode::NOT_FOUND, "document_not_found")
        );

        conn.execute(
            "INSERT INTO documents (id, filename, file_type, byte_size, chunk_count, created_at)
             VALUES ('legacy', 'old.txt', 'txt', 10, 1, 1)",
            [],
        )
        .unwrap();
        assert_eq!(
            source_refusal(&conn, "legacy"),
            (StatusCode::CONFLICT, "document_source_unavailable")
        );

        seed(&conn, "damaged", SAMPLE);
        conn.execute(
            "UPDATE documents SET text_sha256 = NULL WHERE id = 'damaged'",
            [],
        )
        .unwrap();
        assert_eq!(
            source_refusal(&conn, "damaged"),
            (StatusCode::CONFLICT, "document_source_incomplete")
        );
    }
}
