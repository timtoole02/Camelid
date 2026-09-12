//! User-reviewed file outputs. Preparing a review never writes its destination.
//! A durable before/after journal precedes apply and undo; neither operation
//! overwrites a file whose contents have changed since the reviewed version.
use super::{api_error, AppState};
use crate::chat::Sandbox;
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    io::{Read, Write},
    path::{Component, Path as FsPath, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const MAX_BYTES: usize = 256 * 1024;
const MAX_REVIEWS: usize = 128;
#[derive(Clone)]
pub(super) struct ChangeManager {
    directory: Arc<PathBuf>,
    lock: Arc<Mutex<()>>,
}
impl Default for ChangeManager {
    fn default() -> Self {
        Self {
            directory: Arc::new(
                crate::chat::workspace_memory::default_store_path()
                    .with_file_name("change-reviews"),
            ),
            lock: Arc::new(Mutex::new(())),
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Review {
    id: String,
    workspace: PathBuf,
    path: String,
    before: Option<String>,
    after: String,
    source: String,
    status: String,
    created_at: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Proposal {
    workspace: PathBuf,
    path: String,
    content: String,
    #[serde(default)]
    source: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Decision {
    approved: bool,
}

type Result<T> = std::result::Result<T, String>;
fn sandbox(root: &FsPath) -> Result<Sandbox> {
    Sandbox::new(root, false, Duration::from_secs(1))
        .map_err(|_| "The workspace folder is no longer accessible.".into())
}
fn destination(root: &FsPath, relative: &str) -> Result<PathBuf> {
    if relative.is_empty()
        || relative.len() > 1024
        || relative.contains(['\\', ':', '\0'])
        || FsPath::new(relative)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(
            "Use a relative file path inside the selected workspace, without parent traversal."
                .into(),
        );
    }
    if relative
        .split('/')
        .any(|part| part.eq_ignore_ascii_case(".git") || part.eq_ignore_ascii_case(".camelid"))
    {
        return Err("Git metadata and Camelid's state directories cannot be changed here.".into());
    }
    let sb = sandbox(root)?;
    if sb.root() != root {
        return Err("The workspace folder changed since this review was created.".into());
    }
    // Reject symlinks in every component, including links that currently point
    // inside the workspace. Re-run this check immediately before publication.
    let mut path = root.to_path_buf();
    for part in relative.split('/') {
        path.push(part);
        if fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err("Symbolic links cannot be reviewed or changed here.".into());
        }
    }
    sb.resolve_output(relative).map_err(|_| "The destination must be a regular file inside the workspace, with an existing parent folder.".into())
}
fn read_text(path: &FsPath, limit: usize) -> Result<Option<String>> {
    let meta = match fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("Could not inspect the file.".into()),
    };
    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > limit as u64 {
        return Err("Only regular UTF-8 files within the size limit are supported.".into());
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path).map_err(|_| "Could not read the file.")?;
    if !file.metadata().is_ok_and(|m| m.is_file()) {
        return Err("The file changed while it was being opened.".into());
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "Could not read the file.")?;
    if bytes.len() > limit {
        return Err("The file exceeds the size limit.".into());
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_| "Only UTF-8 text files are supported.".into())
}
fn same_version(path: &FsPath, expected: &Option<String>) -> Result<()> {
    if &read_text(path, MAX_BYTES)? != expected {
        return Err("The file changed after this version was reviewed. Nothing was overwritten; create a new review.".into());
    }
    Ok(())
}
fn publish(review: &Review, expected: &Option<String>, replacement: &Option<String>) -> Result<()> {
    let path = destination(&review.workspace, &review.path)?;
    same_version(&path, expected)?;
    if let Some(content) = replacement {
        let mut temp =
            tempfile::NamedTempFile::new_in(path.parent().ok_or("Missing parent folder.")?)
                .map_err(|_| "Could not prepare the replacement file.")?;
        if let Ok(meta) = fs::metadata(&path) {
            if meta.permissions().readonly() {
                return Err("The destination is read-only.".into());
            }
            temp.as_file()
                .set_permissions(meta.permissions())
                .map_err(|_| "Could not preserve file permissions.")?;
        }
        temp.write_all(content.as_bytes())
            .and_then(|_| temp.as_file().sync_all())
            .map_err(|_| "Could not save the replacement file.")?;
        let checked = destination(&review.workspace, &review.path)?;
        if checked != path {
            return Err("The destination changed during review.".into());
        }
        same_version(&path, expected)?;
        if expected.is_none() {
            temp.persist_noclobber(&path)
                .map_err(|_| "The destination appeared before the file could be created.")?;
        } else {
            crate::chat::replace_temp_atomically(temp.path(), &path)
                .map_err(|_| "Could not replace the file; the saved review is still available.")?;
        }
    } else {
        // Only Undo of a newly created file reaches this branch.
        destination(&review.workspace, &review.path)?;
        same_version(&path, expected)?;
        fs::remove_file(&path).map_err(|_| "Could not remove the created file.")?;
    }
    Ok(())
}
impl ChangeManager {
    fn file(&self, id: &str) -> Result<PathBuf> {
        if id.len() != 32 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("Invalid review ID.".into());
        }
        Ok(self.directory.join(format!("{id}.json")))
    }
    fn initialize(&self) -> Result<()> {
        fs::create_dir_all(&*self.directory).map_err(|_| "Could not create the review store.")?;
        let meta = fs::symlink_metadata(&*self.directory)
            .map_err(|_| "Could not inspect the review store.")?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err("The review store must be a regular directory.".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&*self.directory, fs::Permissions::from_mode(0o700))
                .map_err(|_| "Could not protect the review store.")?;
        }
        Ok(())
    }
    fn save(&self, review: &Review) -> Result<()> {
        self.initialize()?;
        let mut file = tempfile::NamedTempFile::new_in(&*self.directory)
            .map_err(|_| "Could not prepare the review journal.")?;
        let data = serde_json::to_vec(review).map_err(|_| "Could not encode the review.")?;
        file.write_all(&data)
            .and_then(|_| file.as_file().sync_all())
            .map_err(|_| "Could not save the review journal; nothing was changed.")?;
        file.persist(self.file(&review.id)?)
            .map_err(|_| "Could not commit the review journal.")?;
        #[cfg(unix)]
        {
            fs::File::open(&*self.directory)
                .and_then(|dir| dir.sync_all())
                .map_err(|_| "Could not flush the review journal.")?;
        }
        Ok(())
    }
    fn load(&self, id: &str) -> Result<Review> {
        let raw = read_text(&self.file(id)?, MAX_BYTES * 12 + 8192)?.ok_or("Review not found.")?;
        let mut review: Review =
            serde_json::from_str(&raw).map_err(|_| "Saved review is invalid.")?;
        if review.id != id
            || review.after.len() > MAX_BYTES
            || review.before.as_ref().is_some_and(|s| s.len() > MAX_BYTES)
        {
            return Err("Saved review exceeds its limits.".into());
        }
        // Crash recovery never executes work. The journal already has the
        // original contents before either apply or undo is attempted.
        if matches!(review.status.as_str(), "applying" | "undoing") {
            let current =
                destination(&review.workspace, &review.path).and_then(|p| read_text(&p, MAX_BYTES));
            review.status = match current {
                Ok(ref bytes) if bytes == &Some(review.after.clone()) => "applied",
                Ok(ref bytes) if bytes == &review.before => {
                    if review.status == "undoing" {
                        "undone"
                    } else {
                        "pending"
                    }
                }
                _ => "conflict",
            }
            .into();
            self.save(&review)?;
        }
        Ok(review)
    }
    fn ids(&self) -> Result<Vec<String>> {
        if !self.directory.exists() {
            return Ok(vec![]);
        }
        let entries =
            fs::read_dir(&*self.directory).map_err(|_| "Could not list saved reviews.")?;
        Ok(entries
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                if p.extension()?.to_str()? != "json" {
                    return None;
                }
                let id = p.file_stem()?.to_str()?.to_string();
                self.file(&id).ok().map(|_| id)
            })
            .collect())
    }
    fn propose(&self, request: Proposal) -> Result<Review> {
        if request.content.len() > MAX_BYTES
            || request.content.contains('\0')
            || request.source.len() > 200
        {
            return Err(
                "File outputs must be UTF-8 text of at most 256 KB, without NUL bytes.".into(),
            );
        }
        if self.ids()?.len() >= MAX_REVIEWS {
            return Err(
                "The review history is full. Remove an old review before adding another.".into(),
            );
        }
        let sb = sandbox(&request.workspace)?;
        let path = destination(sb.root(), &request.path)?;
        let before = read_text(&path, MAX_BYTES)?;
        if before.as_deref() == Some(&request.content) {
            return Err("The proposed file is identical to the current file.".into());
        }
        let review = Review {
            id: uuid::Uuid::new_v4().simple().to_string(),
            workspace: sb.root().to_path_buf(),
            path: request.path,
            before,
            after: request.content,
            source: request.source,
            status: "pending".into(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };
        self.save(&review)?;
        Ok(review)
    }
    fn decide(&self, id: &str, approved: bool) -> Result<Review> {
        let mut review = self.load(id)?;
        if review.status != "pending" {
            return Err("This review has already been decided. It will not apply again.".into());
        }
        if !approved {
            review.status = "rejected".into();
            self.save(&review)?;
            return Ok(review);
        }
        let path = destination(&review.workspace, &review.path)?;
        same_version(&path, &review.before)?;
        review.status = "applying".into();
        self.save(&review)?;
        let result = publish(&review, &review.before, &Some(review.after.clone()));
        if let Err(error) = result {
            review.status = "pending".into();
            self.save(&review)?;
            return Err(error);
        }
        review.status = "applied".into();
        // If this last metadata flush fails, the durable applying journal
        // reconstructs the outcome and keeps Undo available on the next read.
        let _ = self.save(&review);
        Ok(review)
    }
    fn undo(&self, id: &str) -> Result<Review> {
        let mut review = self.load(id)?;
        if review.status != "applied" {
            return Err("Only an applied change can be undone.".into());
        }
        let path = destination(&review.workspace, &review.path)?;
        same_version(&path, &Some(review.after.clone()))?;
        review.status = "undoing".into();
        self.save(&review)?;
        if let Err(error) = publish(&review, &Some(review.after.clone()), &review.before) {
            review.status = "applied".into();
            self.save(&review)?;
            return Err(error);
        }
        review.status = "undone".into();
        let _ = self.save(&review);
        Ok(review)
    }
}
fn view(review: &Review, contents: bool) -> Value {
    let mut value = json!({"id":review.id,"workspace":review.workspace,"path":review.path,"status":review.status,"source":review.source,"created_at":review.created_at,"created":review.before.is_none(),"before_bytes":review.before.as_ref().map_or(0, String::len),"after_bytes":review.after.len()});
    if contents {
        value["before"] = json!(review.before);
        value["after"] = json!(review.after);
        value["diff"] = json!(crate::chat::file_change_diff(
            review.before.as_deref().unwrap_or(""),
            &review.after
        ));
    }
    value
}
fn authorize(state: &AppState, headers: &HeaderMap) -> bool {
    state.serve_addr.ip().is_loopback()
        && super::workspace::local_management_request_allowed(headers)
        && headers
            .get("x-camelid-changes")
            .and_then(|v| v.to_str().ok())
            == Some("1")
}
async fn run(
    state: AppState,
    headers: HeaderMap,
    operation: impl FnOnce(ChangeManager) -> Result<Value> + Send + 'static,
) -> Response {
    if !authorize(&state, &headers) {
        return api_error(
            StatusCode::FORBIDDEN,
            "changes_local_only",
            "File reviews require Camelid's local, same-origin interface.".into(),
            None,
        );
    }
    let manager = state.changes.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _guard = manager
            .lock
            .lock()
            .map_err(|_| "Review store is unavailable.")?;
        operation(manager.clone())
    })
    .await;
    match result {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(message)) => api_error(StatusCode::CONFLICT, "change_review_error", message, None),
        Err(_) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "change_review_error",
            "Review operation could not finish.".into(),
            None,
        ),
    }
}
pub(super) async fn list(State(state): State<AppState>, headers: HeaderMap) -> Response {
    run(state, headers, |m| {
        let mut reviews = m
            .ids()?
            .iter()
            .map(|id| m.load(id))
            .collect::<Result<Vec<_>>>()?;
        reviews.sort_by_key(|r| std::cmp::Reverse(r.created_at));
        Ok(json!({"reviews":reviews.iter().map(|r| view(r,false)).collect::<Vec<_>>()}))
    })
    .await
}
pub(super) async fn prepare(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(proposal): Json<Proposal>,
) -> Response {
    run(state, headers, |m| {
        m.propose(proposal).map(|r| view(&r, true))
    })
    .await
}
pub(super) async fn get(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    run(state, headers, move |m| m.load(&id).map(|r| view(&r, true))).await
}
pub(super) async fn decide(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(decision): Json<Decision>,
) -> Response {
    run(state, headers, move |m| {
        m.decide(&id, decision.approved).map(|r| view(&r, true))
    })
    .await
}
pub(super) async fn undo(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    run(state, headers, move |m| m.undo(&id).map(|r| view(&r, true))).await
}
pub(super) async fn remove(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    run(state, headers, move |m| {
        m.load(&id)?;
        fs::remove_file(m.file(&id)?).map_err(|_| "Could not remove the saved review.")?;
        Ok(json!({"removed":true}))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (tempfile::TempDir, ChangeManager, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let workspace = dir.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        let workspace = fs::canonicalize(workspace).unwrap();
        let manager = ChangeManager {
            directory: Arc::new(dir.path().join("reviews")),
            lock: Arc::new(Mutex::new(())),
        };
        (dir, manager, workspace)
    }
    fn proposal(workspace: &FsPath, path: &str, content: &str) -> Proposal {
        Proposal {
            workspace: workspace.to_path_buf(),
            path: path.into(),
            content: content.into(),
            source: "Test output".into(),
        }
    }
    #[test]
    fn approval_replay_and_durable_undo() {
        let (_dir, m, w) = fixture();
        let path = w.join("hello.txt");
        fs::write(&path, "before\n").unwrap();
        let review = m.propose(proposal(&w, "hello.txt", "after\n")).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "before\n");
        assert_eq!(m.decide(&review.id, true).unwrap().status, "applied");
        assert!(m.decide(&review.id, true).is_err());
        let restarted = ChangeManager {
            directory: m.directory.clone(),
            lock: Arc::new(Mutex::new(())),
        };
        assert_eq!(restarted.undo(&review.id).unwrap().status, "undone");
        assert_eq!(fs::read_to_string(path).unwrap(), "before\n");
        assert!(restarted.undo(&review.id).is_err());
    }
    #[test]
    fn denial_creation_and_version_conflicts() {
        let (_dir, m, w) = fixture();
        let path = w.join("new.txt");
        let rejected = m.propose(proposal(&w, "new.txt", "no")).unwrap();
        m.decide(&rejected.id, false).unwrap();
        assert!(!path.exists());
        assert!(m.decide(&rejected.id, true).is_err());
        let review = m.propose(proposal(&w, "new.txt", "ours")).unwrap();
        fs::write(&path, "user edit").unwrap();
        assert!(m.decide(&review.id, true).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "user edit");
        fs::remove_file(&path).unwrap();
        m.decide(&review.id, true).unwrap();
        fs::write(&path, "later edit").unwrap();
        assert!(m.undo(&review.id).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "later edit");
        fs::write(&path, "ours").unwrap();
        m.undo(&review.id).unwrap();
        assert!(!path.exists());
    }
    #[test]
    fn recovery_and_unsafe_paths() {
        let (_dir, m, w) = fixture();
        for path in [
            "../outside",
            "/absolute",
            ".git/config",
            ".camelid/state",
            "a/../b",
            "a\\b",
            "file:stream",
        ] {
            assert!(m.propose(proposal(&w, path, "bad")).is_err(), "{path}");
        }
        let mut review = m.propose(proposal(&w, "new.txt", "after")).unwrap();
        review.status = "applying".into();
        m.save(&review).unwrap();
        fs::write(w.join("new.txt"), "after").unwrap();
        assert_eq!(m.load(&review.id).unwrap().status, "applied");
        review.status = "undoing".into();
        m.save(&review).unwrap();
        fs::remove_file(w.join("new.txt")).unwrap();
        assert_eq!(m.load(&review.id).unwrap().status, "undone");
        assert!(m
            .propose(proposal(&w, "huge.txt", &"x".repeat(MAX_BYTES + 1)))
            .is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(w.join("new.txt"), w.join("link")).unwrap();
            assert!(m.propose(proposal(&w, "link", "bad")).is_err());
        }
    }
    #[test]
    fn empty_file_creation_and_snapshot_permissions() {
        let (_dir, m, w) = fixture();
        let review = m.propose(proposal(&w, "empty.txt", "")).unwrap();
        assert!(!w.join("empty.txt").exists());
        m.decide(&review.id, true).unwrap();
        assert_eq!(fs::read(w.join("empty.txt")).unwrap(), Vec::<u8>::new());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(m.file(&review.id).unwrap())
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
            assert_eq!(
                fs::metadata(&*m.directory).unwrap().permissions().mode() & 0o777,
                0o700
            );
            fs::set_permissions(w.join("empty.txt"), fs::Permissions::from_mode(0o640)).unwrap();
            let edit = m.propose(proposal(&w, "empty.txt", "updated")).unwrap();
            m.decide(&edit.id, true).unwrap();
            assert_eq!(
                fs::metadata(w.join("empty.txt"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o640
            );
            m.undo(&edit.id).unwrap();
        }
        m.undo(&review.id).unwrap();
        assert!(!w.join("empty.txt").exists());
    }
    #[tokio::test]
    async fn routes_require_local_intent() {
        use axum::{body::Body, http::Request};
        use tower::ServiceExt as _;
        let app = super::super::router_with_state(AppState::default());
        for (origin, intent) in [
            ("https://foreign.example", true),
            ("http://127.0.0.1:8181", false),
        ] {
            let mut req = Request::builder()
                .uri("/api/changes")
                .header("host", "127.0.0.1:8181")
                .header("origin", origin);
            if intent {
                req = req.header("x-camelid-changes", "1");
            }
            assert_eq!(
                app.clone()
                    .oneshot(req.body(Body::empty()).unwrap())
                    .await
                    .unwrap()
                    .status(),
                StatusCode::FORBIDDEN
            );
        }
    }
}
