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

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use super::tools::Sandbox;

const DIR: &str = ".camelid/checkpoints";

/// Append-only record of every committed checkpoint, next to the backups.
///
/// The in-memory log below belongs to one process, but a subagent is a separate
/// process writing into the SAME workspace. Without a shared record its edits
/// are absent from the Changes view and cannot be undone — the session would
/// report a change set that is missing files it really wrote. Journal lines are
/// the durable form; [`sync_from_store`] folds a sibling's entries into this
/// process's log in the order they were actually committed.
const JOURNAL: &str = ".camelid/checkpoints/journal.jsonl";

/// One saved file state, taken immediately before a write.
#[derive(Clone)]
pub struct Checkpoint {
    /// Identity of this entry across processes: `<pid>-<per-process counter>`.
    /// Used to fold a subagent's journal entries in without duplicating our own.
    pub id: String,
    /// Workspace-relative path of the file that was about to change.
    pub rel: String,
    /// Where the previous contents were saved, or `None` if the file did not
    /// exist yet (undo therefore means "delete it again").
    pub backup: Option<PathBuf>,
    pub tool: String,
    /// Hash of the file as the agent left it, recorded when the mutation
    /// committed. If the file no longer matches at undo time, someone else --
    /// usually the user, by hand -- changed it since, and a blind restore
    /// would destroy their work.
    pub post_hash: Option<u64>,
}

/// A snapshot taken before a mutation that has not happened yet. It becomes a
/// [`Checkpoint`] only if the mutation succeeds; a failed tool call must not
/// leave a phantom entry for /undo to "revert".
pub struct Pending {
    rel: String,
    backup: Option<PathBuf>,
    tool: String,
    target: PathBuf,
    /// Resolved store directory, carried so `finish` can journal without
    /// needing the sandbox again.
    dir: PathBuf,
}

/// Serialized form of a [`Checkpoint`] in the shared journal.
#[derive(serde::Serialize, serde::Deserialize)]
struct JournalLine {
    id: String,
    rel: String,
    backup: Option<String>,
    tool: String,
    post_hash: Option<u64>,
}

/// A cheap, dependency-free content hash (FNV-1a). Collision resistance is not
/// the point; detecting "this file changed since the agent wrote it" is.
fn content_hash(path: &Path) -> Option<u64> {
    let bytes = std::fs::read(path).ok()?;
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    Some(h)
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

/// A flattened path safe as a single filename component on every platform:
/// anything outside [A-Za-z0-9._-] becomes '_' (colons would be NTFS stream
/// syntax; separators would be directories).
fn flat_name(rel: &str) -> String {
    rel.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
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
    // Create the store first, then resolve it through the jail. `resolve` with
    // must_exist=false canonicalises the *parent*, so it cannot resolve a
    // two-level path whose first level does not exist yet — the store has to
    // exist before it can be checked, not after.
    std::fs::create_dir_all(sandbox.root().join(DIR)).ok()?;
    let dir = sandbox.resolve(DIR, true).ok()?;

    let backup = if path.exists() {
        // Collision-proof across processes: a subagent shares this store, and
        // a name derived from the process-local log length would let two
        // processes silently clobber each other's backups. Pid + a process
        // atomic counter + the flattened path can collide with nothing.
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let flat = flat_name(&rel);
        let dest = dir.join(format!("{}_{seq:04}_{flat}", std::process::id()));
        std::fs::copy(&target_canon, &dest).ok()?;
        Some(dest)
    } else {
        None
    };

    Some(Pending {
        rel,
        backup,
        tool: tool.to_string(),
        target: target_canon,
        dir,
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
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = format!(
        "{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let cp = Checkpoint {
        id,
        rel: p.rel,
        backup: p.backup,
        tool: p.tool,
        post_hash,
    };
    // Journal before the in-memory push so a crash between the two loses
    // nothing: a duplicate is filtered by id on the next sync, a missing entry
    // would be an un-undoable edit. Best-effort like the rest of this module —
    // a journal that cannot be written must not block the user's edit.
    append_journal(&p.dir, &cp);
    if let Ok(mut g) = log().lock() {
        g.push(cp);
    }
}

fn append_journal(dir: &Path, cp: &Checkpoint) {
    use std::io::Write;
    let line = JournalLine {
        id: cp.id.clone(),
        rel: cp.rel.clone(),
        backup: cp.backup.as_ref().map(|b| b.display().to_string()),
        tool: cp.tool.clone(),
        post_hash: cp.post_hash,
    };
    let Ok(mut json) = serde_json::to_string(&line) else {
        return;
    };
    json.push('\n');
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("journal.jsonl"))
    {
        let _ = f.write_all(json.as_bytes());
    }
}

fn read_journal(root: &Path) -> Vec<Checkpoint> {
    let Ok(text) = std::fs::read_to_string(root.join(JOURNAL)) else {
        return Vec::new();
    };
    // The journal lives INSIDE the workspace, so a sandboxed shell command (or
    // a confused subagent) can write to it. Its `backup` field is later read by
    // the UNSANDBOXED server — `diff` inlines it into /changes on every poll,
    // `undo` copies from it — so an absolute path smuggled into the journal was
    // a read primitive that routed around the kernel sandbox's credential
    // deny-list. Reject any journal line whose backup resolves to a real file
    // OUTSIDE this workspace's checkpoint store. A backup that does not resolve
    // at all (a cleaned-up blob) is kept: the restore copy fails safe, whereas
    // silently nulling it would turn a restore into a delete.
    let store = std::fs::canonicalize(root.join(".camelid/checkpoints")).ok();
    let backup_escapes_store = |path: &Path| -> bool {
        match (std::fs::canonicalize(path).ok(), store.as_deref()) {
            (Some(canonical), Some(store)) => !canonical.starts_with(store),
            // No store dir, or the path resolves but we cannot resolve the
            // store: treat as an escape and refuse.
            (Some(_), None) => true,
            // Unresolvable path: benign (missing blob); the copy will fail safe.
            (None, _) => false,
        }
    };
    text.lines()
        .filter(|line| !line.trim().is_empty())
        // A torn final line from a killed child is skipped, not a hard error.
        .filter_map(|line| serde_json::from_str::<JournalLine>(line).ok())
        .filter(|entry| {
            entry
                .backup
                .as_deref()
                .map(Path::new)
                .is_none_or(|path| !backup_escapes_store(path))
        })
        .map(|entry| Checkpoint {
            id: entry.id,
            rel: entry.rel,
            backup: entry.backup.map(PathBuf::from),
            tool: entry.tool,
            post_hash: entry.post_hash,
        })
        .collect()
}

/// Number of committed, parseable checkpoints in this workspace's shared
/// journal. The agent loop uses a turn-start baseline so a child process's
/// write can satisfy Code's artifact-completion contract without treating an
/// unrelated successful shell probe as a workspace change.
pub fn committed_count(root: &Path) -> usize {
    read_journal(root).len()
}

/// Fold checkpoints committed by other processes (subagents) into this
/// process's log, in the order they were actually committed.
///
/// Called before serving the change set or an undo, so a delegated write is as
/// visible and as revertible as one the server made itself. The journal's
/// append order is the true commit order across processes, so it — not
/// arrival at this process — decides what undo walks back. Idempotent: entries
/// are matched by id, and anything only this process knows (a journal write
/// that failed) is kept at the end rather than dropped.
pub fn sync_from_store(root: &Path) {
    let journaled = read_journal(root);
    if journaled.is_empty() {
        return;
    }
    let Ok(mut g) = log().lock() else { return };
    let ordered: std::collections::HashSet<&str> =
        journaled.iter().map(|cp| cp.id.as_str()).collect();
    let unjournaled: Vec<Checkpoint> = g
        .iter()
        .filter(|cp| !ordered.contains(cp.id.as_str()))
        .cloned()
        .collect();
    *g = journaled;
    g.extend(unjournaled);
}

/// Drop one entry from the shared journal, so an undone change is not folded
/// back in by the next [`sync_from_store`].
fn forget_journaled(root: &Path, id: &str) {
    let journal = root.join(JOURNAL);
    let Ok(text) = std::fs::read_to_string(&journal) else {
        return;
    };
    let kept: Vec<&str> = text
        .lines()
        .filter(|line| {
            serde_json::from_str::<JournalLine>(line)
                .map(|entry| entry.id != id)
                .unwrap_or(true)
        })
        .collect();
    let mut out = kept.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    let _ = std::fs::write(&journal, out);
}

/// Clear the log AND the workspace's shared journal.
///
/// Plain [`clear`] resets only this process, which is right for a terminal
/// session that owns its whole store. A new web Code session must also drop
/// what earlier sessions (and their subagents) journaled, or the first undo
/// would walk back into a previous session's changes.
pub fn clear_for_workspace(root: &Path) {
    clear();
    let _ = std::fs::remove_file(root.join(JOURNAL));
}

/// Undo the most recent checkpoint. Returns what it did.
///
/// Refuses (without `force`) when the file no longer matches the state the
/// agent left it in: that means the user edited it since, and a blind restore
/// would destroy their work to revert the agent's. Before any restore, the
/// current state is parked in the store (`undone_*`), so even a forced undo
/// destroys nothing irrecoverably.
pub fn undo(sandbox: &Sandbox, force: bool) -> Result<String, String> {
    let (cp, depth) = {
        let g = log().lock().map_err(|_| "checkpoint log poisoned")?;
        (g.last().cloned().ok_or("nothing to undo")?, g.len())
    };
    let target = sandbox.resolve(&cp.rel, false)?;
    let target = canonical_target(&target).unwrap_or(target);

    if !force {
        if let (Some(expected), Some(now)) = (cp.post_hash, content_hash(&target)) {
            if expected != now {
                return Err(format!(
                    "{} was changed after the agent wrote it (by you?). /undo would overwrite \
                     those changes — use `/undo force` if that is what you want",
                    cp.rel
                ));
            }
        }
    }

    // Park what is being overwritten, outside the LIFO log (pushing it onto
    // the log would turn walk-back into a toggle).
    if target.exists() {
        if let Ok(dir) = sandbox.resolve(DIR, true) {
            let flat: String = cp
                .rel
                .chars()
                .map(|c| if c == '/' || c == '\\' { '_' } else { c })
                .collect();
            // Sequence the park like `prepare` sequences backups: pid alone
            // made a second undo of the same file overwrite the first park,
            // destroying the only copy of the intermediate state.
            static UNDONE_SEQ: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let seq = UNDONE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let _ = std::fs::copy(
                &target,
                dir.join(format!("undone_{}_{seq:04}_{flat}", std::process::id())),
            );
        }
    }

    // Only pop once the guard has passed — and only if the entry read at the top
    // is still the newest one. The hashing and parking above happen with the log
    // unlocked, and the web lane can run a turn concurrently with an undo, so a
    // blind `pop()` here could discard a checkpoint some other write pushed in
    // between: that write would then be missing from the Changes view and could
    // never be reverted, while the file this call restores is a different one.
    // Refuse instead of corrupting the log; the caller can undo again.
    {
        let mut g = log().lock().map_err(|_| "checkpoint log poisoned")?;
        if g.len() != depth || g.last().map(|last| last.id.as_str()) != Some(cp.id.as_str()) {
            return Err(format!(
                "{} was not undone: the workspace changed while the undo was preparing. Nothing \
                 was modified — try again",
                cp.rel
            ));
        }
        g.pop();
    }
    // RESTORE FIRST, forget after. The old order forgot the journal entry and
    // THEN attempted the copy, so a failed restore (read-only target, ENOSPC,
    // a stale TCC denial on an external volume) left the checkpoint erased
    // from both stores with the target possibly truncated — unrecoverable. A
    // failure now re-pushes the in-memory entry and leaves the journal intact,
    // so the user can fix the cause and undo again.
    // Retryable vs unrecoverable failures diverge below: re-pushing on a
    // MISSING backup blob wedged undo permanently (every attempt pops, fails
    // NotFound, re-pushes the same entry at the top of the stack forever).
    let mut backup_gone = false;
    let restore_result = match &cp.backup {
        Some(b) => {
            // Re-validate at USE time, not just at journal-load time: the blob
            // could have been swapped for a symlink out of the store between
            // the two (the journal is workspace-writable).
            let in_store = std::fs::canonicalize(sandbox.root().join(DIR))
                .ok()
                .zip(std::fs::canonicalize(b).ok())
                .is_some_and(|(store, blob)| blob.starts_with(store));
            if !in_store {
                backup_gone = true;
                Err(format!(
                    "restore failed: the backup for {} is missing or points outside the \
                     checkpoint store; this checkpoint cannot be restored and was dropped",
                    cp.rel
                ))
            } else {
                std::fs::copy(b, &target)
                    .map(|_| format!("restored {}", cp.rel))
                    .map_err(|e| {
                        backup_gone = e.kind() == std::io::ErrorKind::NotFound;
                        format!("restore failed: {e}")
                    })
            }
        }
        None => {
            // The file did not exist before the agent made it.
            let _ = std::fs::remove_file(&target);
            Ok(format!("removed {} (it was newly created)", cp.rel))
        }
    };
    match restore_result {
        Ok(message) => {
            // Keep the durable journal in step, or the next sync would
            // resurrect the entry this call just walked back.
            forget_journaled(sandbox.root(), &cp.id);
            Ok(message)
        }
        Err(error) if backup_gone => {
            // Unrecoverable: drop the entry from the journal too, so undo can
            // proceed to the next checkpoint instead of wedging on this one.
            forget_journaled(sandbox.root(), &cp.id);
            Err(error)
        }
        Err(error) => {
            // Retryable (permissions, disk full): put the entry back so the
            // user can fix the cause and undo again — but only if a concurrent
            // sync has not already restored it.
            if let Ok(mut g) = log().lock() {
                if g.last().map(|last| last.id.as_str()) != Some(cp.id.as_str()) {
                    g.push(cp);
                }
            }
            Err(error)
        }
    }
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
            .ok()
            .and_then(|p| std::fs::read_to_string(p).ok());
        let before = cp
            .backup
            .as_ref()
            .and_then(|b| std::fs::read_to_string(b).ok());
        out.push_str(&format!("--- {} ({})\n", cp.rel, cp.tool));
        match (before, now) {
            (None, Some(after)) => {
                for line in after.lines().take(40) {
                    out.push_str(&format!("+ {line}\n"));
                }
            }
            (Some(b), None) => {
                for line in b.lines().take(40) {
                    out.push_str(&format!("- {line}\n"));
                }
            }
            (Some(b), Some(a)) => out.push_str(&line_diff(&b, &a)),
            (None, None) => out.push_str("(gone)\n"),
        }
    }
    out
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

    /// A failed restore must not destroy the checkpoint. The old order forgot
    /// the journal entry and truncated the target BEFORE the copy, so an
    /// unwritable target lost the entry for good. Now the entry survives a
    /// failure and the file is untouched, so a retry after fixing perms works.
    #[cfg(unix)]
    #[test]
    fn a_failed_restore_preserves_the_checkpoint_and_the_file() {
        use std::os::unix::fs::PermissionsExt;
        let _g = cp_lock();
        clear();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        let f = d.path().join("a.txt");
        std::fs::write(&f, "v1").unwrap();

        let pending = prepare(&sandbox, &f, "edit_file");
        std::fs::write(&f, "v2").unwrap();
        finish(pending, true);

        // Make the file unwritable so std::fs::copy(backup, target) fails.
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o444)).unwrap();
        let result = undo(&sandbox, true);
        // Restore permissions before asserting so a failure still cleans up.
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o644)).unwrap();

        assert!(result.is_err(), "an unwritable target must fail the restore");
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "v2",
            "a failed restore must not truncate the file"
        );
        // The checkpoint survived, so a retry (now writable) succeeds.
        undo(&sandbox, true).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "v1");
        clear();
    }

    /// The journal is workspace-writable, so a hostile absolute `backup` path
    /// would turn the unsandboxed diff/undo reader into a file-exfiltration
    /// primitive. read_journal must drop any backup that resolves outside the
    /// checkpoint store, while keeping the rest of the entry.
    #[test]
    fn a_journal_backup_pointing_outside_the_store_is_rejected() {
        let _g = cp_lock();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        clear_for_workspace(sandbox.root());

        // A real secret outside the workspace.
        let secret = d.path().join("secret_outside.txt");
        std::fs::write(&secret, "TOP SECRET").unwrap();

        let journal = sandbox.root().join(JOURNAL);
        std::fs::create_dir_all(journal.parent().unwrap()).unwrap();
        let line = serde_json::json!({
            "id": "9999-0",
            "rel": "victim.txt",
            "backup": secret.to_string_lossy(),
            "tool": "edit_file",
            "post_hash": null,
        });
        std::fs::write(&journal, format!("{line}\n")).unwrap();

        // The whole line is untrusted and dropped — NOT surfaced with a null
        // backup, which would make undo "delete victim.txt" (None = never
        // existed). A benign line alongside it still loads.
        let benign_backup = sandbox
            .root()
            .join(".camelid/checkpoints/1234_0000_ok.txt");
        std::fs::write(&benign_backup, "prior").unwrap();
        let benign = serde_json::json!({
            "id": "1234-0",
            "rel": "ok.txt",
            "backup": benign_backup.to_string_lossy(),
            "tool": "edit_file",
            "post_hash": null,
        });
        std::fs::write(&journal, format!("{line}\n{benign}\n")).unwrap();

        let loaded = read_journal(sandbox.root());
        assert_eq!(
            loaded.len(),
            1,
            "the escaping line is dropped; only the benign one survives"
        );
        assert_eq!(loaded[0].rel, "ok.txt");
        assert!(
            loaded[0].backup.is_some(),
            "an in-store backup must be preserved"
        );
        clear_for_workspace(sandbox.root());
    }

    #[test]
    fn a_sibling_processs_checkpoint_is_visible_and_undoable_here() {
        // A subagent is a separate process writing into the SAME workspace. Its
        // checkpoints reach this process only through the shared journal; before
        // that existed, a delegated write was missing from the change set and
        // could never be undone.
        let _g = cp_lock();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        clear_for_workspace(sandbox.root());

        let f = d.path().join("child.txt");
        std::fs::write(&f, "before").unwrap();
        let pending = prepare(&sandbox, &f, "write_file");
        std::fs::write(&f, "after").unwrap();
        finish(pending, true);

        // Stand in for the child exiting: its in-memory log dies with it, the
        // journal on disk does not.
        clear();
        assert!(all().is_empty());

        sync_from_store(sandbox.root());
        assert_eq!(
            all().iter().map(|cp| cp.rel.as_str()).collect::<Vec<_>>(),
            vec!["child.txt"],
            "the delegated write must appear in this process's change set"
        );

        undo(&sandbox, false).unwrap();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "before");

        // And it must not come back: the journal entry is dropped with the undo.
        sync_from_store(sandbox.root());
        assert!(all().is_empty(), "an undone change must stay undone");
        clear_for_workspace(sandbox.root());
    }

    #[test]
    fn syncing_orders_by_commit_not_by_arrival() {
        // Undo is LIFO over the whole workspace, so a sibling's later write must
        // walk back before this process's earlier one — arrival order here would
        // revert the wrong file.
        let _g = cp_lock();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        clear_for_workspace(sandbox.root());

        let mine = d.path().join("mine.txt");
        std::fs::write(&mine, "v1").unwrap();
        let p = prepare(&sandbox, &mine, "edit_file");
        std::fs::write(&mine, "v2").unwrap();
        finish(p, true);

        // A sibling commits AFTER us, then exits (its entry lives in the journal
        // only). Simulate by taking our copy out of memory and back in.
        let theirs = d.path().join("theirs.txt");
        std::fs::write(&theirs, "t1").unwrap();
        let p = prepare(&sandbox, &theirs, "edit_file");
        std::fs::write(&theirs, "t2").unwrap();
        finish(p, true);
        clear();

        sync_from_store(sandbox.root());
        undo(&sandbox, false).unwrap();
        assert_eq!(
            std::fs::read_to_string(&theirs).unwrap(),
            "t1",
            "the newest change in the workspace is the one that unwinds first"
        );
        assert_eq!(std::fs::read_to_string(&mine).unwrap(), "v2");
        clear_for_workspace(sandbox.root());
    }

    #[test]
    fn a_torn_journal_line_is_skipped_not_fatal() {
        // A child killed mid-write can leave a half line. Losing that one entry
        // is acceptable; refusing to serve the change set is not.
        let _g = cp_lock();
        let d = tempfile::tempdir().unwrap();
        let sandbox = sb(d.path());
        clear_for_workspace(sandbox.root());

        let f = d.path().join("a.txt");
        std::fs::write(&f, "one").unwrap();
        let p = prepare(&sandbox, &f, "write_file");
        std::fs::write(&f, "two").unwrap();
        finish(p, true);
        clear();

        let journal = d.path().join(JOURNAL);
        let mut text = std::fs::read_to_string(&journal).unwrap();
        text.push_str("{\"id\":\"9-0\",\"rel\":\"trunc");
        std::fs::write(&journal, text).unwrap();

        sync_from_store(sandbox.root());
        assert_eq!(all().len(), 1, "the intact entry survives the torn one");
        clear_for_workspace(sandbox.root());
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
}
