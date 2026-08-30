//! Checkpoints: snapshot a file before the agent changes it, so the user can
//! see and undo what it did.
//!
//! Content snapshots only — never a `git stash`. Two reasons (DECISIONS
//! D-DROVER-5): one code path is easier to prove against the sandbox jail than
//! two, and the agent must not mutate git state the sandbox does not own. A
//! workspace that is not a repo, or is a repo with staged work the user cares
//! about, behaves identically.
//!
//! Snapshots live under `.camelid/checkpoints/` inside the workspace and are
//! resolved through the same canonical-prefix check as every other path, so a
//! checkpoint can neither be written nor restored outside the sandbox root.

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use sha2::{Digest, Sha256};

use super::tools::Sandbox;

const MAX_CHECKPOINT_BYTES: u64 = 16 * 1024 * 1024;

/// One saved file state, taken immediately before a write.
#[derive(Clone)]
pub struct Checkpoint {
    /// Workspace-relative path of the file that was about to change.
    pub rel: String,
    /// Where the previous contents were saved, or `None` if the file did not
    /// exist yet (undo therefore means "delete it again").
    pub backup: Option<PathBuf>,
    backup_hash: Option<[u8; 32]>,
    pub tool: String,
    /// Hash of the file as the agent left it, recorded when the mutation
    /// committed. If the file no longer matches at undo time, someone else --
    /// usually the user, by hand -- changed it since, and a blind restore
    /// would destroy their work.
    pub post_hash: Option<[u8; 32]>,
}

/// A snapshot taken before a mutation that has not happened yet. It becomes a
/// [`Checkpoint`] only if the mutation succeeds; a failed tool call must not
/// leave a phantom entry for /undo to "revert".
pub struct Pending {
    rel: String,
    backup: Option<PathBuf>,
    backup_hash: Option<[u8; 32]>,
    tool: String,
    target: PathBuf,
}

/// A bounded, collision-resistant digest used to detect both ordinary edits
/// and deliberate checkpoint-store tampering.
fn content_hash(path: &Path) -> Option<[u8; 32]> {
    let mut file = open_regular_read(path, MAX_CHECKPOINT_BYTES).ok()?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 16 * 1024];
    let mut total = 0u64;
    loop {
        let read = file.read(&mut buf).ok()?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        if total > MAX_CHECKPOINT_BYTES {
            return None;
        }
        h.update(&buf[..read]);
    }
    Some(h.finalize().into())
}

fn open_regular_read(path: &Path, max_bytes: u64) -> Result<File, String> {
    let meta = std::fs::symlink_metadata(path)
        .map_err(|e| format!("cannot inspect {}: {e}", path.display()))?;
    if meta.file_type().is_symlink() || !meta.is_file() {
        return Err(format!("refusing non-regular file {}", path.display()));
    }
    if meta.len() > max_bytes {
        return Err(format!("{} is too large to checkpoint", path.display()));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let opened = file
        .metadata()
        .map_err(|e| format!("cannot inspect opened {}: {e}", path.display()))?;
    if !opened.is_file() || opened.len() > max_bytes {
        return Err(format!(
            "refusing changed/non-regular file {}",
            path.display()
        ));
    }
    Ok(file)
}

fn secure_checkpoint_dir(sandbox: &Sandbox) -> Result<PathBuf, String> {
    let camelid = sandbox.root().join(".camelid");
    ensure_plain_directory(&camelid)?;
    let checkpoints = camelid.join("checkpoints");
    ensure_plain_directory(&checkpoints)?;
    Ok(checkpoints)
}

fn ensure_plain_directory(path: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => Err(format!(
            "refusing symlink state directory {}",
            path.display()
        )),
        Ok(meta) if !meta.is_dir() => Err(format!(
            "refusing non-directory state path {}",
            path.display()
        )),
        Ok(_) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let builder = std::fs::DirBuilder::new();
            #[cfg(unix)]
            let builder = {
                use std::os::unix::fs::DirBuilderExt;

                let mut builder = builder;
                builder.mode(0o700);
                builder
            };
            builder
                .create(path)
                .map_err(|e| format!("cannot create state directory {}: {e}", path.display()))
        }
        Err(err) => Err(format!("cannot inspect {}: {err}", path.display())),
    }
}

fn copy_regular_create_new(source_path: &Path, destination: &Path) -> Result<(), String> {
    let mut source = open_regular_read(source_path, MAX_CHECKPOINT_BYTES)?;
    let source_permissions = source
        .metadata()
        .map_err(|e| format!("cannot inspect {}: {e}", source_path.display()))
        .map(|meta| meta.permissions())?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut destination_file = options
        .open(destination)
        .map_err(|e| format!("cannot create {}: {e}", destination.display()))?;
    let mut limited = std::io::Read::by_ref(&mut source).take(MAX_CHECKPOINT_BYTES + 1);
    let copied = std::io::copy(&mut limited, &mut destination_file)
        .and_then(|count| {
            if count > MAX_CHECKPOINT_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "source grew beyond the checkpoint limit",
                ));
            }
            destination_file.sync_all()
        })
        .map_err(|e| format!("cannot copy checkpoint: {e}"));
    if let Err(error) = copied {
        drop(destination_file);
        let _ = std::fs::remove_file(destination);
        return Err(error);
    }
    if let Err(error) = std::fs::set_permissions(destination, source_permissions) {
        drop(destination_file);
        let _ = std::fs::remove_file(destination);
        return Err(format!("cannot preserve checkpoint permissions: {error}"));
    }
    Ok(())
}

fn restore_regular(source: &Path, target: &Path) -> Result<(), String> {
    let target_existed = match std::fs::symlink_metadata(target) {
        Ok(meta) if meta.file_type().is_symlink() || !meta.is_file() => {
            return Err(format!("refusing non-regular target {}", target.display()))
        }
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(format!("cannot inspect {}: {error}", target.display())),
    };
    let parent = target
        .parent()
        .ok_or_else(|| format!("target {} has no parent", target.display()))?;
    let name = target
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("restore");
    let temp = next_store_path(parent, ".camelid-restore", name);
    copy_regular_create_new(source, &temp)?;

    // Revalidate immediately before publication. An existing destination is
    // replaced by the platform's atomic primitive (ReplaceFileW on Windows,
    // rename on Unix); a formerly absent destination uses atomic no-clobber
    // publication. No path removes the old bytes before the new file commits.
    let published = if target_existed {
        match std::fs::symlink_metadata(target) {
            Ok(meta) if meta.is_file() && !meta.file_type().is_symlink() => {
                super::tools::replace_temp_atomically(&temp, target)
                    .map_err(|e| format!("cannot publish restored {}: {e}", target.display()))
            }
            Ok(_) => Err(format!(
                "refusing changed restore target {}",
                target.display()
            )),
            Err(error) => Err(format!(
                "restore target {} changed before publication: {error}",
                target.display()
            )),
        }
    } else {
        super::tools::publish_temp_noclobber(&temp, target)
            .map_err(|e| format!("cannot publish restored {}: {e}", target.display()))
    };
    // No-clobber hard-link publication leaves the temporary name behind; the
    // replacement paths consume it. Cleanup is harmless in either case.
    let _ = std::fs::remove_file(&temp);
    published
}

fn next_store_path(dir: &Path, prefix: &str, _rel: &str) -> PathBuf {
    // The relative workspace path belongs in the in-memory checkpoint record,
    // not in one filesystem component: a valid deeply nested path can exceed
    // NAME_MAX after flattening. A UUID keeps names short and unguessable while
    // create_new remains the final no-clobber authority.
    dir.join(format!("{prefix}_{}.bin", uuid::Uuid::new_v4()))
}

/// Canonical form of `path` (resolving the parent when the file does not exist
/// yet, so a symlinked final component cannot smuggle the target elsewhere).
fn canonical_target(path: &Path) -> Option<PathBuf> {
    match std::fs::canonicalize(path) {
        Ok(c) => Some(c),
        Err(_) => {
            let p = std::fs::canonicalize(path.parent()?).ok()?;
            Some(p.join(path.file_name()?))
        }
    }
}

/// The workspace-relative path of `target`, computed canonical-to-canonical.
///
/// Both sides MUST be canonicalised together: a raw path may arrive in a
/// different spelling of the same location (macOS `/var` vs `/private/var`;
/// Windows 8.3 short names like `RUNNERA~1` vs the long form), and a mismatch
/// silently falls back to an absolute path — whose drive colon then lands in a
/// flattened backup filename, which NTFS parses as an alternate-data-stream
/// name and fails with os error 87.
fn canonical_rel(sandbox: &Sandbox, target: &Path) -> Option<(PathBuf, String)> {
    let root = std::fs::canonicalize(sandbox.root()).ok()?;
    let canon = canonical_target(target)?;
    let rel = canon.strip_prefix(&root).ok()?.display().to_string();
    Some((canon, rel))
}

fn log() -> &'static Mutex<Vec<Checkpoint>> {
    static L: OnceLock<Mutex<Vec<Checkpoint>>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn clear() {
    if let Ok(mut g) = log().lock() {
        g.clear();
    }
}

pub fn all() -> Vec<Checkpoint> {
    log().lock().map(|g| g.clone()).unwrap_or_default()
}

/// Snapshot `path` before it is written. Pair with [`finish`].
///
/// Best-effort by design: a failure to snapshot must not block the edit the
/// user asked for, so problems are swallowed. The one thing it will not do is
/// write outside the jail.
pub fn prepare(sandbox: &Sandbox, path: &Path, tool: &str) -> Option<Pending> {
    // Only files inside the workspace are snapshotted (canonical_rel returns
    // None for anything outside the canonical root). With --allow-fs the agent
    // may legitimately write elsewhere, but copying those into an in-workspace
    // store would pull outside content across the boundary the store lives
    // behind — so they get no undo, rather than a leak.
    let (target_canon, rel) = canonical_rel(sandbox, path)?;
    let dir = secure_checkpoint_dir(sandbox).ok()?;

    let backup = if target_canon.exists() {
        // Collision-proof across processes: a subagent shares this store, and
        // create_new also fails closed if an attacker predicts the name and
        // plants a symlink or file there first.
        let dest = next_store_path(&dir, "backup", &rel);
        copy_regular_create_new(&target_canon, &dest).ok()?;
        Some(dest)
    } else {
        None
    };
    let backup_hash = backup.as_deref().and_then(content_hash);

    Some(Pending {
        rel,
        backup,
        backup_hash,
        tool: tool.to_string(),
        target: target_canon,
    })
}

/// Commit or discard a prepared snapshot, depending on whether the mutation
/// succeeded. A failed write leaves the file untouched, so recording a
/// checkpoint for it would hand /undo a no-op that LOOKS like a revert while
/// the real last change stays applied.
pub fn finish(pending: Option<Pending>, mutated: bool) {
    let Some(p) = pending else { return };
    if !mutated {
        if let Some(b) = &p.backup {
            let _ = std::fs::remove_file(b);
        }
        return;
    }
    let post_hash = content_hash(&p.target);
    if let Ok(mut g) = log().lock() {
        g.push(Checkpoint {
            rel: p.rel,
            backup: p.backup,
            backup_hash: p.backup_hash,
            tool: p.tool,
            post_hash,
        });
    }
}

/// Undo the most recent checkpoint. Returns what it did.
///
/// Refuses (without `force`) when the file no longer matches the state the
/// agent left it in: that means the user edited it since, and a blind restore
/// would destroy their work to revert the agent's. Before any restore, the
/// current state is parked in the store (`undone_*`), so even a forced undo
/// destroys nothing irrecoverably.
pub fn undo(sandbox: &Sandbox, force: bool) -> Result<String, String> {
    let cp = {
        let g = log().lock().map_err(|_| "checkpoint log poisoned")?;
        g.last().cloned().ok_or("nothing to undo")?
    };
    let target = sandbox.resolve_output(&cp.rel)?;

    if let Some(backup) = &cp.backup {
        let actual = content_hash(backup)
            .ok_or_else(|| format!("checkpoint backup for {} is unavailable or unsafe", cp.rel))?;
        if cp.backup_hash != Some(actual) {
            return Err(format!(
                "checkpoint backup for {} changed after it was created; refusing restore",
                cp.rel
            ));
        }
    }

    if !force {
        let expected = cp.post_hash.ok_or_else(|| {
            format!(
                "{} cannot be verified as the agent-written version; refusing to overwrite it \
                 (use `/undo force` if that is what you want)",
                cp.rel
            )
        })?;
        let now = content_hash(&target).ok_or_else(|| {
            format!(
                "{} changed or disappeared after the agent wrote it; refusing to restore over \
                 that state (use `/undo force` if that is what you want)",
                cp.rel
            )
        })?;
        if expected != now {
            return Err(format!(
                "{} was changed after the agent wrote it (by you?). /undo would overwrite \
                 those changes — use `/undo force` if that is what you want",
                cp.rel
            ));
        }
    }

    // Park what is being overwritten, outside the LIFO log (pushing it onto
    // the log would turn walk-back into a toggle).
    if target.exists() {
        let dir = secure_checkpoint_dir(sandbox)?;
        let parked = next_store_path(&dir, "undone", &cp.rel);
        copy_regular_create_new(&target, &parked).map_err(|e| {
            format!(
                "could not preserve the current {} before undo; nothing was changed: {e}",
                cp.rel
            )
        })?;
    }

    let result = match &cp.backup {
        Some(b) => {
            restore_regular(b, &target).map_err(|e| format!("restore failed: {e}"))?;
            Ok(format!("restored {}", cp.rel))
        }
        None => {
            // The file did not exist before the agent made it.
            match std::fs::symlink_metadata(&target) {
                Ok(meta) if meta.file_type().is_symlink() || !meta.is_file() => {
                    return Err(format!(
                        "refusing to remove non-regular undo target {}",
                        cp.rel
                    ))
                }
                Ok(_) => {
                    std::fs::remove_file(&target).map_err(|e| format!("remove failed: {e}"))?
                }
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => return Err(format!("remove failed: {err}")),
            }
            Ok(format!("removed {} (it was newly created)", cp.rel))
        }
    };
    // Keep the checkpoint retryable until the restore/removal really succeeds.
    if result.is_ok() {
        if let Ok(mut g) = log().lock() {
            g.pop();
        }
    }
    result
}

/// A unified-ish diff of every checkpointed file against what is on disk now.
pub fn diff(sandbox: &Sandbox) -> String {
    let cps = all();
    if cps.is_empty() {
        return "no changes this session".to_string();
    }
    let mut out = String::new();
    for cp in &cps {
        let now = sandbox
            .resolve(&cp.rel, false)
            .map_err(|error| format!("unsafe current path: {error}"))
            .and_then(|path| read_bounded_text(&path));
        let before = match cp.backup.as_ref() {
            Some(path) => read_bounded_text(path),
            None => Ok(None),
        };
        out.push_str(&format!("--- {} ({})\n", cp.rel, cp.tool));
        match (before, now) {
            (Err(error), _) => {
                out.push_str(&format!("(before image unavailable: {error})\n"));
            }
            (_, Err(error)) => {
                out.push_str(&format!("(current file unavailable: {error})\n"));
            }
            (Ok(None), Ok(Some(after))) => {
                for line in after.lines().take(40) {
                    out.push_str(&format!("+ {line}\n"));
                }
            }
            (Ok(Some(b)), Ok(None)) => {
                for line in b.lines().take(40) {
                    out.push_str(&format!("- {line}\n"));
                }
            }
            (Ok(Some(b)), Ok(Some(a))) => out.push_str(&line_diff(&b, &a)),
            (Ok(None), Ok(None)) => out.push_str("(gone)\n"),
        }
    }
    out
}

/// Read at most one checkpoint-sized regular UTF-8 file. The metadata check is
/// repeated on the opened handle by `open_regular_read`; Unix also uses
/// O_NOFOLLOW, so `/diff` cannot be redirected to a FIFO or outside symlink.
fn read_bounded_text(path: &Path) -> Result<Option<String>, String> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("cannot inspect {}: {error}", path.display())),
        Ok(_) => {}
    }
    let file = open_regular_read(path, MAX_CHECKPOINT_BYTES)?;
    let mut bytes = Vec::new();
    file.take(MAX_CHECKPOINT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_CHECKPOINT_BYTES {
        return Err(format!(
            "{} exceeds the {MAX_CHECKPOINT_BYTES}-byte checkpoint limit",
            path.display()
        ));
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_| format!("{} is not UTF-8 text", path.display()))
}

/// A positional line diff via LCS. A set-membership diff cannot see a moved or
/// duplicated line and would print an affirmatively false "(no textual
/// change)" for a reorder; this one is order-aware. Inputs beyond the DP cap
/// fall back to an honest coarse marker rather than a wrong diff.
pub fn line_diff(before: &str, after: &str) -> String {
    const MAX_LINES: usize = 400;
    const MAX_SHOWN: usize = 80;

    if before == after {
        return "(identical)\n".to_string();
    }
    let b: Vec<&str> = before.lines().collect();
    let a: Vec<&str> = after.lines().collect();
    if b.len() > MAX_LINES || a.len() > MAX_LINES {
        return format!(
            "(files differ; too large for an inline diff — {} → {} lines)\n",
            b.len(),
            a.len()
        );
    }

    // LCS table, then walk back emitting -/+ in order.
    let mut dp = vec![vec![0u16; a.len() + 1]; b.len() + 1];
    for i in (0..b.len()).rev() {
        for j in (0..a.len()).rev() {
            dp[i][j] = if b[i] == a[j] {
                dp[i + 1][j + 1] + 1
            } else {
                dp[i + 1][j].max(dp[i][j + 1])
            };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut out = String::new();
    let mut shown = 0;
    let mut truncated = false;
    while i < b.len() || j < a.len() {
        if shown >= MAX_SHOWN {
            truncated = true;
            break;
        }
        if i < b.len() && j < a.len() && b[i] == a[j] {
            i += 1;
            j += 1;
        } else if j < a.len() && (i >= b.len() || dp[i][j + 1] >= dp[i + 1][j]) {
            out.push_str(&format!("+ {}\n", a[j]));
            j += 1;
            shown += 1;
        } else {
            out.push_str(&format!("- {}\n", b[i]));
            i += 1;
            shown += 1;
        }
    }
    if truncated {
        out.push_str("…(diff truncated)\n");
    }
    if out.is_empty() {
        // Bytes differ but no line does: trailing newline or whitespace change.
        out.push_str("(line endings / trailing whitespace differ)\n");
    }
    out
}

/// `3 change(s): src/a.rs, src/b.rs`
pub fn summary() -> String {
    let cps = all();
    if cps.is_empty() {
        return "no checkpoints this session".to_string();
    }
    let names: Vec<String> = cps.iter().map(|c| c.rel.clone()).collect();
    format!("{} change(s): {}", cps.len(), names.join(", "))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::time::Duration;

    /// The log is process-wide, like the plan and the MCP registry.
    pub(crate) fn cp_lock() -> std::sync::MutexGuard<'static, ()> {
        static L: OnceLock<Mutex<()>> = OnceLock::new();
        L.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn sb(root: &Path) -> Sandbox {
        Sandbox::new(root, false, Duration::from_secs(5)).unwrap()
    }

    #[test]
    fn undo_restores_a_file_byte_identical() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let f = d.path().join("a.txt");
        let original = "line one\nline two\nline three\n";
        std::fs::write(&f, original).unwrap();

        let pending = prepare(&sandbox, &f, "edit_file");
        std::fs::write(&f, "totally different\n").unwrap();
        finish(pending, true);
        assert_ne!(std::fs::read_to_string(&f).unwrap(), original);

        undo(&sandbox, false).unwrap();
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            original,
            "undo must restore byte-identical content"
        );
        clear();
    }

    #[cfg(windows)]
    #[test]
    fn failed_atomic_restore_leaves_existing_target_bytes_intact() {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::FILE_SHARE_READ;

        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("backup.txt");
        let target = dir.path().join("target.txt");
        std::fs::write(&source, "restored bytes").unwrap();
        std::fs::write(&target, "current bytes").unwrap();
        // Deny delete sharing so ReplaceFileW deterministically fails. The old
        // remove-then-rename path destroyed target before observing this error.
        let _held = OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(&target)
            .unwrap();

        assert!(restore_regular(&source, &target).is_err());
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "current bytes");
    }

    #[test]
    fn undo_removes_a_file_the_agent_created() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let f = d.path().join("new.txt");

        let pending = prepare(&sandbox, &f, "write_file"); // does not exist yet
        std::fs::write(&f, "created by the agent").unwrap();
        finish(pending, true);
        assert!(f.exists());

        undo(&sandbox, false).unwrap();
        assert!(!f.exists(), "a newly created file should be removed");
        clear();
    }

    #[test]
    fn undo_walks_back_one_change_at_a_time() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let f = d.path().join("a.txt");
        std::fs::write(&f, "v1").unwrap();

        let p1 = prepare(&sandbox, &f, "edit_file");
        std::fs::write(&f, "v2").unwrap();
        finish(p1, true);
        let p2 = prepare(&sandbox, &f, "edit_file");
        std::fs::write(&f, "v3").unwrap();
        finish(p2, true);

        undo(&sandbox, false).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "v2");
        undo(&sandbox, false).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "v1");
        assert!(undo(&sandbox, false).is_err(), "nothing left to undo");
        clear();
    }

    #[test]
    fn diff_shows_the_actual_on_disk_delta() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let f = d.path().join("a.txt");
        std::fs::write(&f, "keep\nremove me\n").unwrap();

        let pending = prepare(&sandbox, &f, "edit_file");
        std::fs::write(&f, "keep\nadded line\n").unwrap();
        finish(pending, true);

        let out = diff(&sandbox);
        assert!(out.contains("- remove me"), "{out}");
        assert!(out.contains("+ added line"), "{out}");
        assert!(
            !out.contains("- keep"),
            "unchanged lines must not show: {out}"
        );
        assert!(summary().contains("1 change(s)"));
        clear();
    }

    #[test]
    fn diff_refuses_an_oversized_current_file_without_reading_it() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let target = d.path().join("large.txt");
        std::fs::write(&target, "small before\n").unwrap();

        let pending = prepare(&sandbox, &target, "write_file");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&target)
            .unwrap();
        // Sparse on normal test filesystems, so the regression is cheap while
        // still exercising the pre-allocation size guard.
        file.set_len(MAX_CHECKPOINT_BYTES + 1).unwrap();
        finish(pending, true);

        let rendered = diff(&sandbox);
        assert!(rendered.contains("current file unavailable"), "{rendered}");
        assert!(rendered.contains("too large"), "{rendered}");
        clear();
    }

    #[test]
    fn deeply_nested_paths_still_receive_working_checkpoints() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let component = "nested".repeat(12);
        let mut parent = d.path().to_path_buf();
        for suffix in ["a", "b", "c", "d"] {
            parent.push(format!("{component}{suffix}"));
        }
        std::fs::create_dir_all(&parent).unwrap();
        let target = parent.join("source.txt");
        std::fs::write(&target, "original").unwrap();

        let pending = prepare(&sandbox, &target, "edit_file");
        assert!(pending.is_some(), "long relative path silently lost undo");
        std::fs::write(&target, "agent version").unwrap();
        finish(pending, true);
        undo(&sandbox, false).unwrap();
        assert_eq!(std::fs::read_to_string(target).unwrap(), "original");
        clear();
    }

    /// B12: a failed mutation must not leave a checkpoint. A phantom entry
    /// hands /undo a no-op that LOOKS like a revert while the real last change
    /// stays applied.
    #[test]
    fn failed_mutations_leave_no_checkpoint() {
        use super::super::tools::{validate, ToolCall};
        use serde_json::json;
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        std::fs::write(d.path().join("a.txt"), "one two one").unwrap();

        // edit_file with a non-unique needle fails; no checkpoint may remain.
        let out = validate(
            &ToolCall {
                name: "edit_file".into(),
                args: json!({"path":"a.txt","old":"one","new":"three"}),
            },
            &sandbox,
        )
        .unwrap()
        .execute(&sandbox);
        assert!(out.is_err());
        assert!(all().is_empty(), "failed edit left a phantom checkpoint");
        assert!(undo(&sandbox, false).is_err(), "nothing to undo");
        clear();
    }

    /// B15: /undo must not silently destroy the user's own hand-edits made
    /// after the agent's write.
    #[test]
    fn undo_refuses_when_the_user_edited_the_file_since() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let f = d.path().join("a.txt");
        std::fs::write(&f, "original").unwrap();

        let pending = prepare(&sandbox, &f, "write_file");
        std::fs::write(&f, "agent version").unwrap();
        finish(pending, true);

        // The user hand-edits afterwards.
        std::fs::write(&f, "user's careful manual fix").unwrap();

        let err = undo(&sandbox, false).unwrap_err();
        assert!(err.contains("changed after the agent wrote it"), "{err}");
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "user's careful manual fix",
            "refused undo must not touch the file"
        );

        // Forced, it proceeds — but parks the overwritten state first.
        undo(&sandbox, true).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "original");
        let parked: Vec<_> = std::fs::read_dir(d.path().join(".camelid/checkpoints"))
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("undone_"))
            .collect();
        assert_eq!(parked.len(), 1, "the overwritten state must be parked");
        let saved = std::fs::read_to_string(parked[0].path()).unwrap();
        assert_eq!(saved, "user's careful manual fix");
        clear();
    }

    #[test]
    fn undo_refuses_when_the_agent_output_disappeared() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let target = d.path().join("a.txt");
        std::fs::write(&target, "original").unwrap();
        let pending = prepare(&sandbox, &target, "write_file").unwrap();
        std::fs::write(&target, "agent version").unwrap();
        finish(Some(pending), true);

        std::fs::remove_file(&target).unwrap();
        let error = undo(&sandbox, false).unwrap_err();
        assert!(error.contains("disappeared"), "{error}");
        assert!(!target.exists(), "non-forced undo must preserve deletion");
        assert_eq!(all().len(), 1, "refused undo remains retryable");

        undo(&sandbox, true).unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "original");
        clear();
    }

    /// B14: a reorder is a real change. The old set-membership diff printed
    /// "(no textual change)" for it — an affirmatively false answer.
    #[test]
    fn diff_sees_reordered_and_duplicated_lines() {
        // LCS anchors on one common line and shows the others moving; the old
        // set-membership diff saw the same multiset and reported no change.
        let d = line_diff("alpha\nbeta\ngamma\n", "gamma\nbeta\nalpha\n");
        assert!(
            d.contains("+ gamma") && d.contains("- gamma"),
            "reorder invisible: {d}"
        );
        assert!(!d.contains("no textual change") && !d.contains("identical"));

        let dup = line_diff("x\ny\n", "x\nx\ny\n");
        assert!(dup.contains("+ x"), "duplicated line invisible: {dup}");

        // Identical inputs are still identified as such.
        assert!(line_diff("same\n", "same\n").contains("identical"));
    }

    /// The hook lives at the execution site, so an edit made by the model is
    /// checkpointed whether or not the model cooperated.
    #[test]
    fn write_and_edit_through_the_tool_path_are_checkpointed() {
        use super::super::tools::{validate, ToolCall};
        use serde_json::json;
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());

        let call = |name: &str, args| ToolCall {
            name: name.to_string(),
            args,
        };

        // Create, then edit, both through validate -> execute.
        validate(
            &call("write_file", json!({"path":"a.txt","content":"first\n"})),
            &sandbox,
        )
        .unwrap()
        .execute(&sandbox);
        validate(
            &call(
                "edit_file",
                json!({"path":"a.txt","old":"first","new":"second"}),
            ),
            &sandbox,
        )
        .unwrap()
        .execute(&sandbox);

        assert_eq!(all().len(), 2, "both mutations should be checkpointed");
        assert_eq!(
            std::fs::read_to_string(d.path().join("a.txt")).unwrap(),
            "second\n"
        );

        undo(&sandbox, false).unwrap();
        assert_eq!(
            std::fs::read_to_string(d.path().join("a.txt")).unwrap(),
            "first\n"
        );
        undo(&sandbox, false).unwrap();
        assert!(!d.path().join("a.txt").exists());
        clear();
    }

    /// Reads must not create checkpoints — only mutations do.
    #[test]
    fn read_only_tools_take_no_checkpoint() {
        use super::super::tools::{validate, ToolCall};
        use serde_json::json;
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("a.txt"), "x").unwrap();
        let sandbox = sb(d.path());
        validate(
            &ToolCall {
                name: "read_file".into(),
                args: json!({"path":"a.txt"}),
            },
            &sandbox,
        )
        .unwrap()
        .execute(&sandbox);
        assert!(all().is_empty());
        clear();
    }

    /// The store is state inside the workspace, so it obeys the same jail as
    /// every other path. A checkpoint that could be written or restored outside
    /// the sandbox root would be a path-traversal hole.
    #[test]
    fn the_store_stays_inside_the_jail() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());

        let victim = outside.path().join("secret.txt");
        std::fs::write(&victim, "not yours").unwrap();

        // Snapshotting a path outside the root records nothing.
        finish(prepare(&sandbox, &victim, "write_file"), true);
        assert!(all().is_empty(), "snapshotted a file outside the workspace");

        // And every snapshot that IS taken lands under the workspace.
        let f = d.path().join("a.txt");
        std::fs::write(&f, "x").unwrap();
        finish(prepare(&sandbox, &f, "write_file"), true);
        for cp in all() {
            if let Some(b) = &cp.backup {
                let canon = std::fs::canonicalize(b).unwrap();
                assert!(
                    canon.starts_with(std::fs::canonicalize(d.path()).unwrap()),
                    "backup escaped the workspace: {}",
                    canon.display()
                );
            }
        }
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "not yours");
        clear();
    }

    #[test]
    fn a_changed_backup_is_refused_and_remains_retryable() {
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let target = d.path().join("a.txt");
        std::fs::write(&target, "original").unwrap();
        let pending = prepare(&sandbox, &target, "write_file").unwrap();
        let backup = pending.backup.clone().unwrap();
        std::fs::write(&target, "agent version").unwrap();
        finish(Some(pending), true);

        std::fs::write(&backup, "tampered").unwrap();
        let error = undo(&sandbox, true).unwrap_err();
        assert!(error.contains("changed"), "{error}");
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "agent version");
        assert_eq!(all().len(), 1, "failed restore must remain retryable");
        clear();
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_state_and_undo_targets_refuse_symlinks() {
        use std::os::unix::fs::symlink;

        let _g = cp_lock();
        clear();
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let sandbox = sb(root.path());
        let target = root.path().join("a.txt");
        std::fs::write(&target, "original").unwrap();

        symlink(outside.path(), root.path().join(".camelid")).unwrap();
        assert!(prepare(&sandbox, &target, "write_file").is_none());
        assert!(!outside.path().join("checkpoints").exists());
        std::fs::remove_file(root.path().join(".camelid")).unwrap();

        let pending = prepare(&sandbox, &target, "write_file").unwrap();
        std::fs::write(&target, "agent version").unwrap();
        finish(Some(pending), true);
        let victim = outside.path().join("victim.txt");
        std::fs::write(&victim, "safe").unwrap();
        std::fs::remove_file(&target).unwrap();
        symlink(&victim, &target).unwrap();

        assert!(undo(&sandbox, true).is_err());
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "safe");
        assert_eq!(all().len(), 1, "refused undo must retain the checkpoint");
        clear();
    }
}
