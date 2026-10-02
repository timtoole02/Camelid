//! Knowledge Library collections: named sets of library documents that a chat
//! or a project searches together.
//!
//! A collection only groups documents. Deleting a collection leaves its
//! documents in the library and stops watching its folders (see
//! `document_folders`); deleting a document removes it from every
//! collection. Membership survives re-ingesting a document under the same id.

use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use super::documents::{db_lock, open_connection};
use super::{api_error, AppState};

pub(crate) const MAX_NAME_CHARS: usize = 80;
/// A scoped search binds one SQL variable per document; SQLite allows 32,766.
pub(crate) const MAX_SCOPE_DOCUMENTS: usize = 30_000;

pub(crate) fn init_schema(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS document_collections (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS document_collection_members (
            collection_id TEXT NOT NULL REFERENCES document_collections(id) ON DELETE CASCADE,
            doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            added_at INTEGER NOT NULL,
            PRIMARY KEY (collection_id, doc_id)
        );
        CREATE INDEX IF NOT EXISTS document_collection_members_by_doc
            ON document_collection_members(doc_id);",
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CollectionView {
    pub id: String,
    pub name: String,
    pub created_at: i64,
    /// Member documents in the order they were added.
    pub doc_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct CollectionNameRequest {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct CollectionDocumentsRequest {
    pub doc_ids: Vec<String>,
}

#[derive(Debug)]
pub(crate) enum CollectionError {
    InvalidName(&'static str),
    NameTaken(String),
    CollectionNotFound(String),
    DocumentNotFound(String),
    ScopeTooLarge(usize),
    Database(rusqlite::Error),
}

impl From<rusqlite::Error> for CollectionError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Database(error)
    }
}

impl CollectionError {
    pub(crate) fn into_response(self) -> Response {
        let (status, code, message, param) = match self {
            Self::InvalidName(reason) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_collection_name",
                reason.to_string(),
                Some("name"),
            ),
            Self::NameTaken(name) => (
                StatusCode::CONFLICT,
                "collection_name_taken",
                format!("A collection named {name:?} already exists."),
                Some("name"),
            ),
            Self::CollectionNotFound(id) => (
                StatusCode::NOT_FOUND,
                "collection_not_found",
                format!("No collection has the id {id:?}."),
                Some("collection_ids"),
            ),
            Self::DocumentNotFound(id) => (
                StatusCode::NOT_FOUND,
                "document_not_found",
                format!("No document has the id {id:?}."),
                Some("doc_ids"),
            ),
            Self::ScopeTooLarge(count) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "search_scope_too_large",
                format!(
                    "The search scope holds {count} documents; search at most {MAX_SCOPE_DOCUMENTS} at once, or search the whole library."
                ),
                Some("collection_ids"),
            ),
            Self::Database(error) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "collection_database_error",
                error.to_string(),
                None,
            ),
        };
        api_error(status, code, message, param)
    }
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn is_unique_violation(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(failure, _)
            if failure.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE
    )
}

pub(crate) fn validate_name(raw: &str) -> Result<String, CollectionError> {
    let name = raw.trim();
    if name.is_empty() {
        return Err(CollectionError::InvalidName("Give the collection a name."));
    }
    if name.chars().count() > MAX_NAME_CHARS {
        return Err(CollectionError::InvalidName(
            "Collection names are at most 80 characters.",
        ));
    }
    if name.chars().any(char::is_control) {
        return Err(CollectionError::InvalidName(
            "Collection names cannot contain control characters.",
        ));
    }
    Ok(name.to_string())
}

fn members(conn: &Connection, collection_id: &str) -> Result<Vec<String>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT doc_id FROM document_collection_members WHERE collection_id = ?1 ORDER BY rowid",
    )?;
    let rows = statement.query_map(params![collection_id], |row| row.get(0))?;
    rows.collect()
}

pub(crate) fn get(conn: &Connection, id: &str) -> Result<CollectionView, CollectionError> {
    let row = conn
        .query_row(
            "SELECT name, created_at FROM document_collections WHERE id = ?1",
            params![id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()?;
    let (name, created_at) =
        row.ok_or_else(|| CollectionError::CollectionNotFound(id.to_string()))?;
    Ok(CollectionView {
        id: id.to_string(),
        name,
        created_at,
        doc_ids: members(conn, id)?,
    })
}

/// Every collection, by name.
pub(crate) fn list(conn: &Connection) -> Result<Vec<CollectionView>, rusqlite::Error> {
    let mut statement = conn.prepare(
        "SELECT id, name, created_at FROM document_collections ORDER BY name COLLATE NOCASE, id",
    )?;
    let collections = statement
        .query_map([], |row| {
            Ok(CollectionView {
                id: row.get(0)?,
                name: row.get(1)?,
                created_at: row.get(2)?,
                doc_ids: Vec::new(),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    collections
        .into_iter()
        .map(|collection| {
            let doc_ids = members(conn, &collection.id)?;
            Ok(CollectionView {
                doc_ids,
                ..collection
            })
        })
        .collect()
}

pub(crate) fn create(conn: &Connection, raw_name: &str) -> Result<CollectionView, CollectionError> {
    let name = validate_name(raw_name)?;
    let id = uuid::Uuid::new_v4().to_string();
    let created_at = now_secs();
    conn.execute(
        "INSERT INTO document_collections (id, name, created_at) VALUES (?1, ?2, ?3)",
        params![id, name, created_at],
    )
    .map_err(|error| {
        if is_unique_violation(&error) {
            CollectionError::NameTaken(name.clone())
        } else {
            error.into()
        }
    })?;
    Ok(CollectionView {
        id,
        name,
        created_at,
        doc_ids: Vec::new(),
    })
}

pub(crate) fn rename(
    conn: &Connection,
    id: &str,
    raw_name: &str,
) -> Result<CollectionView, CollectionError> {
    let name = validate_name(raw_name)?;
    let changed = conn
        .execute(
            "UPDATE document_collections SET name = ?2 WHERE id = ?1",
            params![id, name],
        )
        .map_err(|error| {
            if is_unique_violation(&error) {
                CollectionError::NameTaken(name.clone())
            } else {
                error.into()
            }
        })?;
    if changed == 0 {
        return Err(CollectionError::CollectionNotFound(id.to_string()));
    }
    get(conn, id)
}

pub(crate) fn delete(conn: &Connection, id: &str) -> Result<(), CollectionError> {
    let removed = conn.execute(
        "DELETE FROM document_collections WHERE id = ?1",
        params![id],
    )?;
    if removed == 0 {
        return Err(CollectionError::CollectionNotFound(id.to_string()));
    }
    Ok(())
}

/// Fails on the first id that names no collection.
pub(crate) fn require_collections(
    conn: &Connection,
    ids: &[String],
) -> Result<(), CollectionError> {
    let mut statement = conn.prepare_cached("SELECT 1 FROM document_collections WHERE id = ?1")?;
    for id in ids {
        if !statement.exists(params![id])? {
            return Err(CollectionError::CollectionNotFound(id.clone()));
        }
    }
    Ok(())
}

/// Adds `doc_id` to each collection; already a member is not an error. The
/// caller has checked the collections exist and owns the transaction.
pub(crate) fn add_memberships(
    conn: &Connection,
    collection_ids: &[String],
    doc_id: &str,
) -> Result<(), rusqlite::Error> {
    let added_at = now_secs();
    let mut statement = conn.prepare_cached(
        "INSERT OR IGNORE INTO document_collection_members (collection_id, doc_id, added_at) VALUES (?1, ?2, ?3)",
    )?;
    for collection_id in collection_ids {
        statement.execute(params![collection_id, doc_id, added_at])?;
    }
    Ok(())
}

/// Adds documents to a collection, all or none: an unknown document id adds nothing.
pub(crate) fn add_documents(
    conn: &mut Connection,
    id: &str,
    doc_ids: &[String],
) -> Result<CollectionView, CollectionError> {
    let tx = conn.transaction()?;
    require_collections(&tx, &[id.to_string()])?;
    {
        let mut exists = tx.prepare_cached("SELECT 1 FROM documents WHERE id = ?1")?;
        for doc_id in doc_ids {
            if !exists.exists(params![doc_id])? {
                return Err(CollectionError::DocumentNotFound(doc_id.clone()));
            }
        }
    }
    let collection = [id.to_string()];
    for doc_id in doc_ids {
        add_memberships(&tx, &collection, doc_id)?;
    }
    tx.commit()?;
    get(conn, id)
}

/// Removing a document that is not a member is not an error.
pub(crate) fn remove_document(
    conn: &Connection,
    id: &str,
    doc_id: &str,
) -> Result<(), CollectionError> {
    require_collections(conn, &[id.to_string()])?;
    conn.execute(
        "DELETE FROM document_collection_members WHERE collection_id = ?1 AND doc_id = ?2",
        params![id, doc_id],
    )?;
    Ok(())
}

/// The documents a search covers: `None` for the whole library when neither
/// list is given, otherwise the named documents followed by every member of
/// the named collections, without repeats.
pub(crate) fn resolve_scope(
    conn: &Connection,
    doc_ids: Option<&[String]>,
    collection_ids: Option<&[String]>,
) -> Result<Option<Vec<String>>, CollectionError> {
    if doc_ids.is_none() && collection_ids.is_none() {
        return Ok(None);
    }
    let mut seen = HashSet::new();
    let mut scope = Vec::new();
    for doc_id in doc_ids.unwrap_or_default() {
        if seen.insert(doc_id.clone()) {
            scope.push(doc_id.clone());
        }
    }
    let collection_ids = collection_ids.unwrap_or_default();
    require_collections(conn, collection_ids)?;
    for collection_id in collection_ids {
        for doc_id in members(conn, collection_id)? {
            if seen.insert(doc_id.clone()) {
                scope.push(doc_id);
            }
        }
    }
    if scope.len() > MAX_SCOPE_DOCUMENTS {
        return Err(CollectionError::ScopeTooLarge(scope.len()));
    }
    Ok(Some(scope))
}

fn with_connection<T>(
    work: impl FnOnce(&mut Connection) -> Result<T, CollectionError>,
) -> Result<T, CollectionError> {
    let _lock = db_lock().lock().unwrap();
    let mut conn = open_connection()?;
    work(&mut conn)
}

/// Endpoint: `GET /api/collections`
pub async fn list_collections(
    State(_state): State<AppState>,
) -> Result<Json<Vec<CollectionView>>, Response> {
    with_connection(|conn| Ok(list(conn)?))
        .map(Json)
        .map_err(CollectionError::into_response)
}

/// Endpoint: `POST /api/collections`
pub async fn create_collection(
    State(_state): State<AppState>,
    Json(payload): Json<CollectionNameRequest>,
) -> Result<impl IntoResponse, Response> {
    with_connection(|conn| create(conn, &payload.name))
        .map(|collection| (StatusCode::CREATED, Json(collection)))
        .map_err(CollectionError::into_response)
}

/// Endpoint: `PATCH /api/collections/:id`
pub async fn rename_collection(
    State(_state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(payload): Json<CollectionNameRequest>,
) -> Result<Json<CollectionView>, Response> {
    with_connection(|conn| rename(conn, &id, &payload.name))
        .map(Json)
        .map_err(CollectionError::into_response)
}

/// Endpoint: `DELETE /api/collections/:id`
pub async fn delete_collection(
    State(_state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, Response> {
    with_connection(|conn| delete(conn, &id))
        .map(|()| StatusCode::NO_CONTENT)
        .map_err(CollectionError::into_response)
}

/// Endpoint: `POST /api/collections/:id/documents`
pub async fn add_collection_documents(
    State(_state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(payload): Json<CollectionDocumentsRequest>,
) -> Result<Json<CollectionView>, Response> {
    with_connection(|conn| add_documents(conn, &id, &payload.doc_ids))
        .map(Json)
        .map_err(CollectionError::into_response)
}

/// Endpoint: `DELETE /api/collections/:id/documents/:doc_id`
pub async fn remove_collection_document(
    State(_state): State<AppState>,
    AxumPath((id, doc_id)): AxumPath<(String, String)>,
) -> Result<StatusCode, Response> {
    with_connection(|conn| remove_document(conn, &id, &doc_id))
        .map(|()| StatusCode::NO_CONTENT)
        .map_err(CollectionError::into_response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::document_vectors::tests::seed_passages;
    use crate::api::documents::init_db;

    fn library() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("PRAGMA foreign_keys = ON;", []).unwrap();
        init_db(&conn).unwrap();
        seed_passages(&conn, "handbook", &["Leave is booked two weeks ahead."]);
        seed_passages(&conn, "policy", &["Refunds are issued within 60 days."]);
        seed_passages(&conn, "notes", &["The courier leaves at dawn."]);
        conn
    }

    fn ids(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn names_are_trimmed_bounded_and_unique_regardless_of_case() {
        let conn = library();
        let hr = create(&conn, "  HR policies ").unwrap();
        assert_eq!(hr.name, "HR policies");
        assert!(matches!(
            create(&conn, "hr POLICIES"),
            Err(CollectionError::NameTaken(_))
        ));
        assert!(matches!(
            create(&conn, "   "),
            Err(CollectionError::InvalidName(_))
        ));
        assert!(matches!(
            create(&conn, &"x".repeat(81)),
            Err(CollectionError::InvalidName(_))
        ));
        assert!(
            create(&conn, &"é".repeat(80)).is_ok(),
            "the limit counts characters, not bytes"
        );
        assert!(matches!(
            create(&conn, "tab\there"),
            Err(CollectionError::InvalidName(_))
        ));

        let finance = create(&conn, "Finance").unwrap();
        assert!(matches!(
            rename(&conn, &finance.id, "hr policies"),
            Err(CollectionError::NameTaken(_))
        ));
        assert_eq!(
            rename(&conn, &finance.id, "Finance 2027").unwrap().name,
            "Finance 2027"
        );
        assert_eq!(
            rename(&conn, &hr.id, "hr policies").unwrap().name,
            "hr policies",
            "a collection may change its own case"
        );
        assert!(matches!(
            rename(&conn, "missing", "Anything"),
            Err(CollectionError::CollectionNotFound(_))
        ));
    }

    #[test]
    fn collections_list_by_name_with_members_in_the_order_added() {
        let mut conn = library();
        let zeta = create(&conn, "zeta").unwrap();
        let alpha = create(&conn, "Alpha").unwrap();
        add_documents(&mut conn, &zeta.id, &ids(&["policy", "handbook"])).unwrap();
        add_documents(&mut conn, &zeta.id, &ids(&["handbook", "notes"])).unwrap();
        let listed = list(&conn).unwrap();
        assert_eq!(
            listed.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            ["Alpha", "zeta"]
        );
        assert_eq!(listed[0].doc_ids, Vec::<String>::new());
        assert_eq!(
            listed[1].doc_ids,
            ids(&["policy", "handbook", "notes"]),
            "adding a member twice keeps one entry"
        );
        assert_eq!(get(&conn, &alpha.id).unwrap(), listed[0]);
    }

    #[test]
    fn adding_an_unknown_document_adds_nothing() {
        let mut conn = library();
        let hr = create(&conn, "HR").unwrap();
        let error = add_documents(&mut conn, &hr.id, &ids(&["handbook", "ghost"])).unwrap_err();
        assert!(matches!(error, CollectionError::DocumentNotFound(ref id) if id == "ghost"));
        assert!(get(&conn, &hr.id).unwrap().doc_ids.is_empty());
        assert!(matches!(
            add_documents(&mut conn, "missing", &ids(&["handbook"])),
            Err(CollectionError::CollectionNotFound(_))
        ));
    }

    #[test]
    fn removing_a_member_or_the_collection_keeps_the_documents() {
        let mut conn = library();
        let hr = create(&conn, "HR").unwrap();
        add_documents(&mut conn, &hr.id, &ids(&["handbook", "policy"])).unwrap();
        remove_document(&conn, &hr.id, "handbook").unwrap();
        remove_document(&conn, &hr.id, "handbook").unwrap();
        assert_eq!(get(&conn, &hr.id).unwrap().doc_ids, ids(&["policy"]));
        assert!(matches!(
            remove_document(&conn, "missing", "policy"),
            Err(CollectionError::CollectionNotFound(_))
        ));

        delete(&conn, &hr.id).unwrap();
        assert!(matches!(
            delete(&conn, &hr.id),
            Err(CollectionError::CollectionNotFound(_))
        ));
        let documents: i64 = conn
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
            .unwrap();
        let memberships: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM document_collection_members",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!((documents, memberships), (3, 0));
    }

    #[test]
    fn deleting_a_document_removes_it_from_every_collection() {
        let mut conn = library();
        let hr = create(&conn, "HR").unwrap();
        let all = create(&conn, "Everything").unwrap();
        add_documents(&mut conn, &hr.id, &ids(&["handbook"])).unwrap();
        add_documents(&mut conn, &all.id, &ids(&["handbook", "policy"])).unwrap();
        conn.execute("DELETE FROM documents WHERE id = 'handbook'", [])
            .unwrap();
        assert!(get(&conn, &hr.id).unwrap().doc_ids.is_empty());
        assert_eq!(get(&conn, &all.id).unwrap().doc_ids, ids(&["policy"]));
    }

    #[test]
    fn a_scope_is_the_named_documents_then_each_collections_members_once() {
        let mut conn = library();
        let hr = create(&conn, "HR").unwrap();
        let ops = create(&conn, "Ops").unwrap();
        add_documents(&mut conn, &hr.id, &ids(&["handbook", "policy"])).unwrap();
        add_documents(&mut conn, &ops.id, &ids(&["notes", "handbook"])).unwrap();

        assert_eq!(
            resolve_scope(&conn, None, None).unwrap(),
            None,
            "no lists means the whole library"
        );
        assert_eq!(
            resolve_scope(
                &conn,
                Some(&ids(&["policy"])),
                Some(&[hr.id.clone(), ops.id.clone()])
            )
            .unwrap(),
            Some(ids(&["policy", "handbook", "notes"]))
        );
        let empty = create(&conn, "Empty").unwrap();
        assert_eq!(
            resolve_scope(&conn, None, Some(&[empty.id])).unwrap(),
            Some(Vec::new())
        );
        assert!(matches!(
            resolve_scope(&conn, None, Some(&[hr.id, "gone".to_string()])),
            Err(CollectionError::CollectionNotFound(ref id)) if id == "gone"
        ));
    }
}
