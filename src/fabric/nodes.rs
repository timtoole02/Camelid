//! Which machines this fabric places on, and how that set changes while it runs.
//!
//! The node set used to be fixed at construction: `--node` was read once at
//! startup and never again, so adding a machine or taking one away meant
//! stopping the proxy — which drops the requests everyone else has in flight,
//! the exact thing the graceful stop exists to prevent.
//!
//! A node file is the set, as a file the operator can edit, diff and back up.
//! Its syntax is deliberately the same `label=host[:port]` that `--node`
//! already takes, parsed by [`parse_node_spec`] through [`parse_fabric`], so
//! there is one answer to "what is a node spec" and one place that answers it.

use std::collections::VecDeque;
use std::fs;
use std::io::{Error, ErrorKind, Result, Write};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;

use super::aliases::{parse_model_aliases, AliasParseError, ModelAliases, ALIAS_PREFIX};
use super::capability::capabilities_of;
use super::divergence::sha256_hex;
use super::engine::NodeEngine;
use super::node::{parse_fabric, NodeSpec, NodeSpecParseError};
use super::watch::{Change, WatchedFile};

/// How long a loaded set is trusted before the file is looked at again.
///
/// A node joining or leaving is only noticed this fast. A second is short
/// enough that an operator does not wait on it and long enough that a busy
/// proxy is not stat-ing a file on every placement.
pub(crate) const DEFAULT_NODE_RELOAD_INTERVAL: Duration = Duration::from_secs(1);

/// How many announcements are kept for health. A file edited in a loop must
/// not grow the proxy's memory; the log line carries every one regardless.
const MAX_FOREIGN_ADDITIONS: usize = 32;

/// A node of an engine this fabric does not place on by default, added to the
/// node file while mixed placement was on — and so placed on without anyone
/// having been shown it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForeignAddition {
    pub label: String,
    pub engine: NodeEngine,
    /// What was accepted about it, in the words the operator saw at startup
    /// for the nodes that were there then.
    pub blockers: Vec<&'static str>,
}

enum Source {
    /// Specs given on the command line. They cannot change while the process
    /// runs, so there is nothing to re-read.
    Fixed(Arc<[NodeSpec]>),
    File(WatchedFile<Arc<[NodeSpec]>>),
}

struct Inner {
    source: Source,
    /// Bumped only when the set actually changes, never merely because the
    /// file was re-read. An observation is valid for the generation it was
    /// taken in and no other; see [`NodeSet::current`].
    generation: AtomicU64,
    /// Set while mixed placement is on, because then a foreign node added to
    /// the file is placed on at once: the flag is a standing grant, and a
    /// grant nobody hears being used is not one anybody made.
    announce_foreign_additions: AtomicBool,
    /// The most recent of those, oldest first.
    foreign_additions: Mutex<VecDeque<ForeignAddition>>,
}

impl Inner {
    fn over(source: Source) -> Self {
        Self {
            source,
            generation: AtomicU64::new(0),
            announce_foreign_additions: AtomicBool::new(false),
            foreign_additions: Mutex::new(VecDeque::new()),
        }
    }
}

/// The set of nodes a fabric places on.
///
/// Cloning shares one set, for the same reason the observation and the
/// reservations are shared: the resident proxy hands a `Fabric` to every
/// request, and a set only one clone could see would be re-read by every
/// other one — and worse, they would disagree about which machines exist.
#[derive(Clone)]
pub(crate) struct NodeSet {
    inner: Arc<Inner>,
}

impl NodeSet {
    /// A set that cannot change.
    pub(crate) fn fixed(specs: Vec<NodeSpec>) -> Self {
        Self {
            inner: Arc::new(Inner::over(Source::Fixed(Arc::from(specs)))),
        }
    }

    /// A set read from a file, re-read as it changes.
    pub(crate) fn from_file(path: PathBuf) -> Result<Self> {
        Self::from_file_every(path, DEFAULT_NODE_RELOAD_INTERVAL)
    }

    /// [`Self::from_file`] with the staleness bound supplied.
    ///
    /// A zero interval re-reads on every look, which is what the tests use so
    /// a change does not have to be waited out.
    pub(crate) fn from_file_every(path: PathBuf, interval: Duration) -> Result<Self> {
        // Loaded once here so an unusable file stops the proxy at startup
        // rather than at the first request.
        let specs = load_specs(&path)?;
        Ok(Self {
            inner: Arc::new(Inner::over(Source::File(WatchedFile::new(
                path, interval, specs,
            )))),
        })
    }

    /// Whether the set changes without a restart.
    pub(crate) fn is_reloadable(&self) -> bool {
        matches!(self.inner.source, Source::File(_))
    }

    /// Announce every foreign node added from now on. Nodes already in the set
    /// were in front of the operator when they turned mixed placement on.
    pub(crate) fn announce_foreign_additions(&self, on: bool) {
        self.inner
            .announce_foreign_additions
            .store(on, Ordering::SeqCst);
    }

    /// The foreign nodes announced since this set was built, most recent last.
    pub(crate) fn foreign_added_since_start(&self) -> Vec<ForeignAddition> {
        self.inner
            .foreign_additions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .cloned()
            .collect()
    }

    fn announce(&self, additions: Vec<ForeignAddition>) {
        if additions.is_empty() {
            return;
        }
        let mut kept = self
            .inner
            .foreign_additions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for addition in additions {
            tracing::warn!(
                label = %addition.label,
                engine = %addition.engine,
                blockers = %addition.blockers.join("; "),
                "node of another engine added while placing on other engines; accepted without asking"
            );
            // Printed as well: `RUST_LOG` is unset on a stock proxy.
            eprintln!("{}", foreign_addition_notice(&addition));
            if kept.len() == MAX_FOREIGN_ADDITIONS {
                kept.pop_front();
            }
            kept.push_back(addition);
        }
    }

    /// The set as it stands, and the generation it belongs to.
    ///
    /// The generation is the point of this method. An observation describes
    /// the set it was taken over, so one taken before a machine was added or
    /// removed is not merely stale, it is *about something else*. Callers keep
    /// the generation alongside the observation and reuse neither across a
    /// change. Returning it — rather than a "did it change" flag — means a
    /// caller that only wanted the specs cannot accidentally swallow the
    /// change and leave the next caller reusing an observation of a set that
    /// no longer exists.
    pub(crate) fn current(&self) -> (Arc<[NodeSpec]>, u64) {
        let Source::File(watched) = &self.inner.source else {
            let Source::Fixed(specs) = &self.inner.source else {
                unreachable!("a source is either fixed or a file")
            };
            return (Arc::clone(specs), 0);
        };

        let (specs, change) = watched.look(load_specs);
        match change {
            Change::None => {}
            Change::Loaded {
                previous,
                recovered,
            } => {
                // Compared, not assumed: touching a file without changing what
                // it says must not throw away a usable observation.
                if specs != previous {
                    tracing::info!(
                        nodes = specs.len(),
                        path = %watched.path().display(),
                        "node set reloaded"
                    );
                    self.inner.generation.fetch_add(1, Ordering::SeqCst);
                    if self.inner.announce_foreign_additions.load(Ordering::SeqCst) {
                        self.announce(foreign_additions(&previous, &specs));
                    }
                }
                if recovered {
                    eprintln!(
                        "fabric: node file {} is readable again; placing on {}",
                        watched.path().display(),
                        nodes_phrase(specs.len())
                    );
                }
            }
            Change::Failed { error, first } => {
                // The previous set is kept on purpose; see [`super::watch`].
                // The cost is that a change written into a broken or deleted
                // file does not take effect, so the operator has to hear about
                // it.
                tracing::warn!(
                    path = %watched.path().display(),
                    %error,
                    "could not reload the node set; the previous one is still in force"
                );
                // Printed, not only traced: `RUST_LOG` is unset on a stock
                // proxy, so a machine the operator meant to take out would go
                // on being placed on with nothing said.
                if first {
                    eprintln!("{}", stale_node_set_notice(&error, specs.len()));
                }
            }
        }
        (specs, self.inner.generation.load(Ordering::SeqCst))
    }
}

impl std::fmt::Debug for NodeSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let specs = match &self.inner.source {
            Source::Fixed(specs) => Arc::clone(specs),
            Source::File(watched) => watched.cached(),
        };
        let generation = self.inner.generation.load(Ordering::SeqCst);
        f.debug_struct("NodeSet")
            .field("specs", &specs)
            .field("reloadable", &self.is_reloadable())
            .field("generation", &generation)
            .finish()
    }
}

/// The nodes in `current` that were not in `previous` and whose engine this
/// fabric would not place on by default. Pure.
///
/// A label that changed engine counts as added: the node it now names is one
/// nobody was shown. Decided by the engine's blockers, not its name, and from
/// the engine alone because they are known before its first probe.
fn foreign_additions(previous: &[NodeSpec], current: &[NodeSpec]) -> Vec<ForeignAddition> {
    current
        .iter()
        .filter(|spec| {
            !previous
                .iter()
                .any(|before| before.label == spec.label && before.engine == spec.engine)
        })
        .filter_map(|spec| {
            let blockers = capabilities_of(spec.engine, None).placement_blockers();
            (!blockers.is_empty()).then(|| ForeignAddition {
                label: spec.label.clone(),
                engine: spec.engine,
                blockers,
            })
        })
        .collect()
}

/// The line an operator sees when the standing grant is used. Pure.
fn foreign_addition_notice(addition: &ForeignAddition) -> String {
    format!(
        "fabric: {} ({}) added while placing on other engines; accepted without asking: {}. \
         Tool-calling requests: not until measured.",
        addition.label,
        addition.engine,
        addition.blockers.join("; ")
    )
}

/// Read and validate a node file into the shape the set holds.
fn load_specs(path: &Path) -> Result<Arc<[NodeSpec]>> {
    load_node_file(path).map(Arc::from)
}

fn meaningful_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
}

/// The node lines of a node file's text. The text-level half of
/// [`load_node_file`], factored out so a file and a *would-be* file are read by
/// exactly the same rule.
fn parse_specs_text(text: &str) -> std::result::Result<Vec<NodeSpec>, NodeSpecParseError> {
    let lines: Vec<String> = meaningful_lines(text)
        .filter(|line| !line.starts_with(ALIAS_PREFIX))
        .map(str::to_string)
        .collect();
    parse_fabric(&lines)
}

/// The alias lines of the same text, by the same argument.
fn parse_aliases_text(text: &str) -> std::result::Result<ModelAliases, AliasParseError> {
    let lines: Vec<String> = meaningful_lines(text)
        .filter(|line| line.starts_with(ALIAS_PREFIX))
        .map(str::to_string)
        .collect();
    parse_model_aliases(&lines)
}

/// The SHA-256 of a node file as it stands, or `None` if there is no file.
///
/// What a caller shows a person is what they are agreeing to add a line to, so
/// the write checks the file is still that one.
pub(crate) fn file_sha256(path: &Path) -> Option<String> {
    fs::read(path).ok().map(|bytes| sha256_hex(&bytes))
}

/// What the caller believed the file was when a person confirmed the addition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Base {
    Sha256(String),
    /// The caller believes there is no file yet. CLI only: the proxy's own
    /// startup already requires a loadable one.
    Absent,
}

/// A node file that gained exactly one node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct Appended {
    pub(crate) path: String,
    /// The exact text added, so a person can be shown what was written rather
    /// than a description of it.
    pub(crate) appended: String,
    pub(crate) line: String,
    pub(crate) sha256_before: Option<String>,
    pub(crate) sha256_after: String,
}

/// Why nothing was written. Every one of these leaves the file untouched.
///
/// Public because a join carries one outward: the code is what a page branches
/// on, and the message is what a person is shown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppendRefusal {
    /// The file is not what it was when the person was shown it.
    FileChanged,
    FileDoesNotParse(String),
    DuplicateLabel(String),
    /// The bytes we composed would have meant more than the one node asked for.
    /// The backstop: even a sanitizer bug cannot get past this.
    WouldChangeMoreThanTheNode(String),
    WriteFailed(String),
}

impl std::fmt::Display for AppendRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileChanged => write!(
                f,
                "the nodes file changed since it was read; nothing was written. Scan again so \
                 the line is added to the file as it stands now"
            ),
            Self::FileDoesNotParse(detail) => write!(
                f,
                "{detail}. A line added to a file the proxy cannot read would never take effect, \
                 so nothing was written"
            ),
            Self::DuplicateLabel(label) => {
                write!(f, "node label `{label}` is used more than once")
            }
            Self::WouldChangeMoreThanTheNode(detail) => write!(
                f,
                "writing that would have changed more than the one node asked for ({detail}); \
                 nothing was written"
            ),
            Self::WriteFailed(detail) => write!(f, "the nodes file could not be written: {detail}"),
        }
    }
}

impl AppendRefusal {
    pub fn code(&self) -> &'static str {
        match self {
            Self::FileChanged => "file_changed",
            Self::FileDoesNotParse(_) => "file_does_not_parse",
            Self::DuplicateLabel(_) => "duplicate_label",
            Self::WouldChangeMoreThanTheNode(_) => "invalid_spec",
            Self::WriteFailed(_) => "write_failed",
        }
    }
}

/// Serializes every write to a node file from inside this process.
///
/// A re-read of the hash cannot close a lost update between two threads of one
/// process: both read the same bytes, both find them unchanged, and one line
/// wins. Only a lock held across read, compose and rename can.
static NODES_FILE_WRITE: Mutex<()> = Mutex::new(());

/// Add exactly one node to a node file, preserving every existing byte.
///
/// The file is the operator's: their comments, their commented-out machines,
/// their alias lines, their line endings. So this appends to the bytes rather
/// than re-serializing what was parsed out of them — a round trip through
/// `NodeSpec` would silently delete everything that is not a node.
///
/// Six things have to hold, and any one of them failing writes nothing:
///
/// 1. the file is still the one whose hash the caller was shown;
/// 2. what is there now parses, allowing zero nodes so a comments-only file can
///    take its first;
/// 3. the label is not already used;
/// 4. the would-be text parses under the *loader's* own rules;
/// 5. its node list is the old list plus exactly this spec, and its alias list
///    is unchanged — the backstop against anything smuggled in through a
///    string, however it got there;
/// 6. the write lands whole, via a temp file in the same directory and a
///    rename, so a reader sees the old file or the new one and never half a
///    line.
pub(crate) fn append_node(
    path: &Path,
    spec: &NodeSpec,
    base: &Base,
    comment: &str,
) -> std::result::Result<Appended, AppendRefusal> {
    if !comment.starts_with('#') || comment.chars().any(char::is_control) {
        return Err(AppendRefusal::WouldChangeMoreThanTheNode(
            "the provenance comment is not a single comment line".to_string(),
        ));
    }

    let _serialized = NODES_FILE_WRITE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let real = canonical_target(path)?;
    let directory = real
        .parent()
        .ok_or_else(|| AppendRefusal::WriteFailed("the nodes file has no directory".to_string()))?
        .to_path_buf();

    // Held for the rest of this function on unix, so another process editing
    // the same file waits rather than racing. No lock file is ever created:
    // the lock is on the target itself.
    let _across_processes = lock_target(&real)?;

    let existing = match fs::read(&real) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == ErrorKind::NotFound => None,
        Err(error) => return Err(AppendRefusal::WriteFailed(error.to_string())),
    };
    let sha256_before = existing.as_deref().map(sha256_hex);
    match (base, &sha256_before) {
        (Base::Sha256(expected), Some(found)) if expected == found => {}
        (Base::Absent, None) => {}
        _ => return Err(AppendRefusal::FileChanged),
    }

    let old = match &existing {
        Some(bytes) => String::from_utf8(bytes.clone()).map_err(|_| {
            AppendRefusal::FileDoesNotParse(format!(
                "node file {} is not UTF-8",
                real.display()
            ))
        })?,
        None => String::new(),
    };

    // Pre-state: zero nodes is allowed on purpose. The loader refuses an empty
    // fabric, which is right for serving and wrong here — it would make a file
    // of nothing but comments the one file a person could never add to.
    let before_specs = parse_specs_text(&old).map_err(|error| {
        AppendRefusal::FileDoesNotParse(format!("node file {}: {error}", real.display()))
    })?;
    let before_aliases = parse_aliases_text(&old).map_err(|error| {
        AppendRefusal::FileDoesNotParse(format!("node file {}: {error}", real.display()))
    })?;
    if before_specs
        .iter()
        .any(|existing| existing.label == spec.label)
    {
        return Err(AppendRefusal::DuplicateLabel(spec.label.clone()));
    }

    let eol = if old.contains("\r\n") { "\r\n" } else { "\n" };
    let line = spec.to_line();
    let mut appended = String::new();
    if !old.is_empty() && !old.ends_with('\n') {
        appended.push_str(eol);
    }
    appended.push_str(comment);
    appended.push_str(eol);
    appended.push_str(&line);
    appended.push_str(eol);
    let new = format!("{old}{appended}");

    // Post-state, under the loader's own rules, including its non-empty one.
    let after_specs = parse_specs_text(&new).map_err(|error| {
        AppendRefusal::WouldChangeMoreThanTheNode(format!("the result would not load: {error}"))
    })?;
    let after_aliases = parse_aliases_text(&new).map_err(|error| {
        AppendRefusal::WouldChangeMoreThanTheNode(format!("the result would not load: {error}"))
    })?;
    if after_specs.is_empty() {
        return Err(AppendRefusal::WouldChangeMoreThanTheNode(
            "the result would name no nodes".to_string(),
        ));
    }
    let mut expected = before_specs.clone();
    expected.push(spec.clone());
    if after_specs != expected {
        return Err(AppendRefusal::WouldChangeMoreThanTheNode(format!(
            "it would leave {} nodes where {} were asked for",
            after_specs.len(),
            expected.len()
        )));
    }
    if after_aliases != before_aliases {
        return Err(AppendRefusal::WouldChangeMoreThanTheNode(
            "it would change the model aliases".to_string(),
        ));
    }

    write_atomically(&directory, &real, new.as_bytes(), existing.is_none())?;

    Ok(Appended {
        path: real.display().to_string(),
        appended,
        line,
        sha256_before,
        sha256_after: sha256_hex(new.as_bytes()),
    })
}

/// Resolve the path a write should land on, following a symlink to its target.
///
/// A renamed file replaces whatever the name pointed at, so writing to the link
/// would turn an operator's symlink into a regular file and orphan the thing it
/// pointed at.
fn canonical_target(path: &Path) -> std::result::Result<PathBuf, AppendRefusal> {
    match path.canonicalize() {
        Ok(real) => Ok(real),
        Err(error) if error.kind() == ErrorKind::NotFound => {
            let parent = path.parent().filter(|parent| !parent.as_os_str().is_empty());
            let directory = match parent {
                Some(parent) => parent
                    .canonicalize()
                    .map_err(|error| AppendRefusal::WriteFailed(error.to_string()))?,
                None => std::env::current_dir()
                    .map_err(|error| AppendRefusal::WriteFailed(error.to_string()))?,
            };
            let name = path.file_name().ok_or_else(|| {
                AppendRefusal::WriteFailed("the nodes file has no name".to_string())
            })?;
            Ok(directory.join(name))
        }
        Err(error) => Err(AppendRefusal::WriteFailed(error.to_string())),
    }
}

/// Take an exclusive lock on the node file itself, for as long as the returned
/// handle lives.
///
/// The inode is re-checked after locking: another process could have renamed a
/// new file into place between the open and the lock, and a lock on the file
/// that used to be there protects nothing.
#[cfg(unix)]
fn lock_target(path: &Path) -> std::result::Result<Option<fs::File>, AppendRefusal> {
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::io::AsRawFd;

    for _ in 0..3 {
        let file = match fs::OpenOptions::new().read(true).write(true).open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(AppendRefusal::WriteFailed(error.to_string())),
        };
        // SAFETY: the descriptor is owned by `file` and outlives the call.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(AppendRefusal::WriteFailed(
                std::io::Error::last_os_error().to_string(),
            ));
        }
        let locked = file
            .metadata()
            .map_err(|error| AppendRefusal::WriteFailed(error.to_string()))?;
        match fs::metadata(path) {
            Ok(current) if current.ino() == locked.ino() => return Ok(Some(file)),
            // Somebody renamed a new file in. Drop this lock and take one on
            // the file that is actually there now.
            Ok(_) => continue,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(AppendRefusal::WriteFailed(error.to_string())),
        }
    }
    Err(AppendRefusal::WriteFailed(
        "the nodes file was replaced repeatedly while waiting for a lock".to_string(),
    ))
}

/// No cross-process file lock is taken here, so concurrent safety on this
/// platform rests on the hash check alone. Documented rather than implied.
#[cfg(not(unix))]
fn lock_target(_path: &Path) -> std::result::Result<Option<fs::File>, AppendRefusal> {
    Ok(None)
}

/// Replace the file's contents whole: temp file in the same directory, then
/// rename. A reader sees one or the other, never half a line.
fn write_atomically(
    directory: &Path,
    real: &Path,
    bytes: &[u8],
    creating: bool,
) -> std::result::Result<(), AppendRefusal> {
    let mut temp = tempfile::Builder::new()
        .prefix(".nodes.discover-")
        .tempfile_in(directory)
        .map_err(|error| AppendRefusal::WriteFailed(error.to_string()))?;
    temp.write_all(bytes)
        .and_then(|()| temp.as_file().sync_all())
        .map_err(|error| AppendRefusal::WriteFailed(error.to_string()))?;
    if let Ok(existing) = fs::metadata(real) {
        // The operator's own mode, not the temp file's private default.
        let _ = fs::set_permissions(temp.path(), existing.permissions());
    }

    // Creating uses no-clobber, so a file that appeared while we were composing
    // is reported rather than overwritten.
    let persisted = if creating {
        temp.persist_noclobber(real).map_err(|error| {
            if error.error.kind() == ErrorKind::AlreadyExists {
                AppendRefusal::FileChanged
            } else {
                AppendRefusal::WriteFailed(error.error.to_string())
            }
        })
    } else {
        temp.persist(real)
            .map_err(|error| AppendRefusal::WriteFailed(error.error.to_string()))
    };
    persisted?;

    // The rename itself has to reach the disk, or a crash leaves the directory
    // pointing at a file that is no longer there.
    #[cfg(unix)]
    if let Ok(handle) = fs::File::open(directory) {
        let _ = handle.sync_all();
    }
    Ok(())
}

/// The label of an existing node that already occupies this endpoint, if one
/// does. Pure, over addresses that have already been resolved.
///
/// Matching on resolved addresses rather than on how a host was spelled is the
/// point: `localhost:11434` and `127.0.0.1:11434` are one server, and adding it
/// twice makes the fabric believe it has two machines and re-place a failed
/// request onto the one that just failed.
pub(crate) fn endpoint_conflict(
    existing: &[(NodeSpec, Vec<IpAddr>)],
    port: u16,
    addresses: &[IpAddr],
) -> Option<String> {
    existing
        .iter()
        .find(|(spec, resolved)| {
            spec.port == port
                && resolved
                    .iter()
                    .any(|address| addresses.contains(address))
        })
        .map(|(spec, _)| spec.label.clone())
}

/// Read the `alias` lines from a node file.
///
/// Read once, when the fabric is built, rather than on the node file's reload
/// schedule: an alias is a claim about what weights are, and a claim that
/// changed under a running comparison would make its receipt unreproducible.
pub(crate) fn load_model_aliases(path: &Path) -> Result<ModelAliases> {
    let text = fs::read_to_string(path).map_err(|error| {
        Error::new(
            error.kind(),
            format!("could not read node file {}: {error}", path.display()),
        )
    })?;

    let lines: Vec<String> = meaningful_lines(&text)
        .filter(|line| line.starts_with(ALIAS_PREFIX))
        .map(str::to_string)
        .collect();

    parse_model_aliases(&lines).map_err(|error| {
        Error::new(
            ErrorKind::InvalidData,
            format!("node file {}: {error}", path.display()),
        )
    })
}

/// Read and validate a node file.
///
/// Blank lines and whole-line `#` comments are skipped, so an operator can
/// take a machine out by commenting it rather than deleting what they wrote.
/// `alias` lines declare model identity and are read separately by
/// [`load_model_aliases`]. Everything else is a spec in `--node` syntax, and
/// duplicate labels are refused by [`parse_fabric`] exactly as they are on the
/// command line.
fn load_node_file(path: &Path) -> Result<Vec<NodeSpec>> {
    let text = fs::read_to_string(path).map_err(|error| {
        Error::new(
            error.kind(),
            format!("could not read node file {}: {error}", path.display()),
        )
    })?;

    let lines: Vec<String> = meaningful_lines(&text)
        .filter(|line| !line.starts_with(ALIAS_PREFIX))
        .map(str::to_string)
        .collect();

    let specs = parse_fabric(&lines).map_err(|error| {
        Error::new(
            ErrorKind::InvalidData,
            format!("node file {}: {error}", path.display()),
        )
    })?;

    // An empty set is refused rather than served, at startup and on every
    // reload. A fabric with no nodes answers every request with 503, and the
    // realistic way to get an empty file is a truncated write rather than a
    // deliberate one. An operator who wants to serve nothing stops the proxy.
    if specs.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "node file {} names no nodes; a fabric with no nodes can serve \
                 nothing, so the proxy will not run on one",
                path.display()
            ),
        ));
    }
    Ok(specs)
}

/// What an operator is told when a re-read fails. Pure, so it is tested rather
/// than eyeballed.
///
/// It has to say the change did not happen: the file and the set being placed
/// on no longer agree, and the file is the one the operator is looking at. A
/// machine they meant to take out is still taking requests.
fn stale_node_set_notice(error: &Error, nodes: usize) -> String {
    format!(
        "fabric: could not reload the node file: {error}. The previous set of {} \
         is still being placed on, so a change written here has NOT taken effect.",
        nodes_phrase(nodes)
    )
}

fn nodes_phrase(count: usize) -> String {
    if count == 1 {
        "1 machine".to_string()
    } else {
        format!("{count} machines")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &tempfile::TempDir, body: &str) -> PathBuf {
        let path = dir.path().join("nodes");
        fs::write(&path, body).expect("write node file");
        path
    }

    const TWO: &str = "a=127.0.0.1:8181\nb=127.0.0.1:8182\n";

    fn labels(specs: &[NodeSpec]) -> Vec<String> {
        specs.iter().map(|spec| spec.label.clone()).collect()
    }

    #[test]
    fn alias_lines_share_the_file_with_node_lines_without_either_reading_the_other() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(
            &dir,
            "# the machines\n\
             studio=ollama://127.0.0.1:11434\n\
             desk=lmstudio://127.0.0.1:1234\n\
             \n\
             # what each of them calls the same weights\n\
             alias llama-3.2-1b-instruct=studio:llama-3.2-1b-instruct:latest\n\
             alias llama-3.2-1b-instruct=desk:llama-3.2-1b-instruct\n",
        );

        let specs = load_node_file(&path).expect("nodes parse, ignoring alias lines");
        assert_eq!(labels(&specs), ["studio", "desk"]);

        let aliases = load_model_aliases(&path).expect("aliases parse, ignoring node lines");
        assert_eq!(
            aliases.resolve("studio", "llama-3.2-1b-instruct"),
            "llama-3.2-1b-instruct:latest"
        );
        assert_eq!(
            aliases.resolve("desk", "llama-3.2-1b-instruct"),
            "llama-3.2-1b-instruct"
        );
    }

    #[test]
    fn a_file_with_no_alias_lines_yields_no_declarations_rather_than_failing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, TWO);
        assert!(load_model_aliases(&path).expect("parses").is_empty());
    }

    #[test]
    fn a_malformed_alias_line_names_the_file_it_is_in() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write(&dir, "a=127.0.0.1:8181\nalias nonsense\n");
        let message = load_model_aliases(&path).expect_err("refused").to_string();
        assert!(message.contains("node file"), "{message}");
        assert!(message.contains("alias CANONICAL=LABEL:LOCAL"), "{message}");
    }

    #[test]
    fn a_fixed_set_never_changes() {
        let set = NodeSet::fixed(parse_fabric(&["a=host".to_string()]).expect("parse"));
        assert!(!set.is_reloadable());
        let (first, generation) = set.current();
        let (again, same) = set.current();
        assert_eq!(labels(&first), labels(&again));
        assert_eq!(generation, same);
    }

    /// The point of the file: a machine joins and starts being placed on,
    /// without the proxy stopping.
    #[test]
    fn a_node_added_to_the_file_joins_the_set() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "a=127.0.0.1:8181\n");
        let set = NodeSet::from_file_every(path.clone(), Duration::ZERO).expect("load");

        let (before, first_generation) = set.current();
        assert_eq!(labels(&before), vec!["a"]);

        fs::write(&path, TWO).expect("rewrite");

        let (after, second_generation) = set.current();
        assert_eq!(labels(&after), vec!["a", "b"]);
        assert_ne!(
            first_generation, second_generation,
            "a changed set must not share a generation with the set it replaced"
        );
    }

    #[test]
    fn a_node_removed_from_the_file_leaves_the_set() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, TWO);
        let set = NodeSet::from_file_every(path.clone(), Duration::ZERO).expect("load");
        assert_eq!(labels(&set.current().0), vec!["a", "b"]);

        fs::write(&path, "b=127.0.0.1:8182\n").expect("rewrite");
        assert_eq!(labels(&set.current().0), vec!["b"]);
    }

    /// Commenting a machine out is how an operator takes it away for an hour
    /// without losing what they had typed.
    #[test]
    fn blank_lines_and_comments_are_not_nodes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(
            &dir,
            "# my fabric\n\n  a=127.0.0.1:8181  \n\n#b=127.0.0.1:8182\n",
        );
        let set = NodeSet::from_file(path).expect("load");
        assert_eq!(labels(&set.current().0), vec!["a"]);
    }

    /// A file being replaced is briefly unreadable, and an empty one is very
    /// likely a truncated write. Neither may empty the fabric.
    #[test]
    fn a_broken_or_empty_file_leaves_the_previous_set_in_force() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, TWO);
        let set = NodeSet::from_file_every(path.clone(), Duration::ZERO).expect("load");

        fs::write(&path, "not a spec").expect("corrupt");
        assert_eq!(labels(&set.current().0), vec!["a", "b"]);

        fs::write(&path, "").expect("truncate");
        assert_eq!(labels(&set.current().0), vec!["a", "b"]);

        fs::remove_file(&path).expect("remove");
        assert_eq!(labels(&set.current().0), vec!["a", "b"]);

        // ...and a file that becomes usable again is picked up.
        fs::write(&path, "c=127.0.0.1:8183\n").expect("restore");
        assert_eq!(labels(&set.current().0), vec!["c"]);
    }

    /// Rewriting a file with the same content must not invalidate the
    /// observation taken over it, or an operator's editor could cost a probe
    /// of every node for nothing.
    ///
    /// Only the identical rewrite is asserted here. A *reordering* is a change
    /// the generation should also catch, but it cannot be tested this way: the
    /// reordered text is the same length as the original, so a write landing in
    /// the same modification-time tick is indistinguishable from no write at
    /// all, and the test would fail on timing rather than on behaviour. Adding
    /// and removing are covered above and change the length.
    #[test]
    fn rewriting_a_file_without_changing_it_does_not_change_the_generation() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, TWO);
        let set = NodeSet::from_file_every(path.clone(), Duration::ZERO).expect("load");
        let (_, before) = set.current();

        fs::write(&path, TWO).expect("identical rewrite");
        let (specs, after) = set.current();

        assert_eq!(labels(&specs), vec!["a", "b"]);
        assert_eq!(before, after, "an identical rewrite is not a change");
    }

    #[test]
    fn formatting_a_set_does_not_reload_its_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "a=127.0.0.1:8181\n");
        let set = NodeSet::from_file_every(path.clone(), Duration::ZERO).expect("load");
        let (_, before) = set.current();

        fs::write(&path, TWO).expect("rewrite");
        let rendered = format!("{set:?}");

        assert!(rendered.contains("a"), "{rendered}");
        assert_eq!(
            set.inner.generation.load(Ordering::SeqCst),
            before,
            "formatting must not perform I/O or change placement state"
        );

        let (specs, after) = set.current();
        assert_eq!(labels(&specs), vec!["a", "b"]);
        assert_ne!(before, after, "an explicit lookup still reloads the file");
    }

    /// Keeping the previous set is only safe if the operator is told, and
    /// `RUST_LOG` is unset on a stock proxy, so the notice is printed. It has
    /// to say the change did not happen: a machine they meant to take out is
    /// still taking requests, and "could not reload" alone reads like a retry
    /// that will sort itself out.
    #[test]
    fn the_notice_says_the_change_has_not_taken_effect() {
        let error = Error::new(ErrorKind::InvalidData, "nodes: needs a label");
        let notice = stale_node_set_notice(&error, 2);
        assert!(notice.contains("nodes: needs a label"), "{notice}");
        assert!(notice.contains("previous set of 2 machines"), "{notice}");
        assert!(notice.contains("NOT taken effect"), "{notice}");
        assert!(
            stale_node_set_notice(&error, 1).contains("previous set of 1 machine "),
            "one machine is not '1 machines'"
        );
    }

    #[test]
    fn a_set_inside_the_bound_is_not_re_read() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "a=127.0.0.1:8181\n");
        let set = NodeSet::from_file_every(path.clone(), Duration::from_secs(3600)).expect("load");

        fs::write(&path, TWO).expect("rewrite");

        assert_eq!(
            labels(&set.current().0),
            vec!["a"],
            "the file was re-read before its staleness bound had passed"
        );
    }

    #[test]
    fn an_unusable_file_stops_the_proxy_rather_than_emptying_the_fabric() {
        let dir = tempfile::tempdir().expect("temp dir");

        for (label, body) in [
            ("no nodes", ""),
            ("only comments", "# nothing here\n\n"),
            ("not a spec", "this is not a node\n"),
            ("duplicate label", "a=127.0.0.1:8181\na=127.0.0.1:8182\n"),
            ("unbracketed ipv6", "a=::1:8181\n"),
        ] {
            let path = write(&dir, body);
            NodeSet::from_file(path)
                .err()
                .unwrap_or_else(|| panic!("{label} was accepted"));
        }

        NodeSet::from_file(dir.path().join("absent"))
            .expect_err("a missing file is not an empty fabric");
    }

    /// The flag is a standing grant over a file that is re-read. A node added
    /// to it is placed on at once, so it has to be announced, and remembered
    /// for health, the moment the set takes it in.
    #[test]
    fn a_foreign_node_added_under_mixed_mode_is_announced() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "a=127.0.0.1:8181\n");
        let set = NodeSet::from_file_every(path.clone(), Duration::ZERO).expect("load");
        set.announce_foreign_additions(true);

        fs::write(&path, "a=127.0.0.1:8181\nb=ollama://127.0.0.1:1\n").expect("add b");
        let (specs, _) = set.current();
        assert_eq!(labels(&specs), ["a", "b"]);

        let announced = set.foreign_added_since_start();
        assert_eq!(announced.len(), 1, "{announced:?}");
        assert_eq!(announced[0].label, "b");
        assert_eq!(announced[0].engine, NodeEngine::Ollama);
        assert_eq!(announced[0].blockers.len(), 3, "{announced:?}");
        let notice = foreign_addition_notice(&announced[0]);
        assert!(notice.contains("b (ollama)"), "{notice}");
        assert!(notice.contains("publishes no load to rank on"), "{notice}");

        // The same edit with the flag off is no grant, so nothing to announce.
        let quiet_dir = tempfile::tempdir().expect("temp dir");
        let quiet_path = write(&quiet_dir, "a=127.0.0.1:8181\n");
        let quiet = NodeSet::from_file_every(quiet_path.clone(), Duration::ZERO).expect("load");
        fs::write(&quiet_path, "a=127.0.0.1:8181\nb=ollama://127.0.0.1:1\n").expect("add b");
        assert_eq!(labels(&quiet.current().0), ["a", "b"]);
        assert!(quiet.foreign_added_since_start().is_empty());
    }

    #[test]
    fn a_node_of_our_own_engine_added_later_is_not_announced() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "a=127.0.0.1:8181\n");
        let set = NodeSet::from_file_every(path.clone(), Duration::ZERO).expect("load");
        set.announce_foreign_additions(true);
        fs::write(&path, TWO).expect("add a Camelid node");
        assert_eq!(labels(&set.current().0), ["a", "b"]);
        assert!(set.foreign_added_since_start().is_empty());

        // A label that changes engine names a node nobody was shown.
        fs::write(&path, "a=127.0.0.1:8181\nb=lmstudio://127.0.0.1:8182\n").expect("re-declare b");
        set.current();
        let announced = set.foreign_added_since_start();
        assert_eq!(announced.len(), 1, "{announced:?}");
        assert_eq!(announced[0].engine, NodeEngine::LmStudio);
    }

    // ---- adding one node to an operator's file -------------------------------

    fn found(label: &str, host: &str, port: u16, engine: NodeEngine) -> NodeSpec {
        NodeSpec {
            label: label.to_string(),
            host: host.to_string(),
            port,
            engine,
        }
    }

    const COMMENT: &str = "# joined by fabric discover 2026-09-12T10:04:11Z: \
                           127.0.0.1:11434 answered like ollama 0.33.2";

    fn base_of(path: &Path) -> Base {
        Base::Sha256(file_sha256(path).expect("the file exists"))
    }

    fn temp_files(dir: &tempfile::TempDir) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir.path())
            .expect("read dir")
            .map(|entry| entry.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// The file belongs to the operator. Everything in it that is not a node —
    /// comments, a machine they commented out, blank lines, aliases — has to
    /// come back byte for byte.
    #[test]
    fn joining_appends_and_leaves_every_existing_byte_alone() {
        let dir = tempfile::tempdir().expect("temp dir");
        // Deliberately no trailing newline: the append has to add one.
        let original = "# my fabric\n\
                        \n\
                        local=127.0.0.1:8181\n\
                        #retired=192.0.2.9:8181\n\
                        alias llama=local:llama-3.2-1b";
        let path = write(&dir, original);
        let before = base_of(&path);

        let spec = found("studio", "127.0.0.1", 11434, NodeEngine::Ollama);
        let appended = append_node(&path, &spec, &before, COMMENT).expect("appends");

        let after = fs::read_to_string(&path).expect("read back");
        assert!(
            after.starts_with(original),
            "every existing byte has to survive verbatim:\n{after}"
        );
        assert_eq!(
            after,
            format!("{original}\n{COMMENT}\nstudio=ollama://127.0.0.1:11434\n")
        );
        assert_eq!(appended.line, "studio=ollama://127.0.0.1:11434");
        assert_eq!(
            appended.appended,
            format!("\n{COMMENT}\nstudio=ollama://127.0.0.1:11434\n")
        );
        assert_eq!(appended.sha256_after, file_sha256(&path).expect("hashes"));

        assert_eq!(labels(&load_node_file(&path).expect("loads")), ["local", "studio"]);
        assert_eq!(
            load_model_aliases(&path).expect("aliases").resolve("local", "llama"),
            "llama-3.2-1b",
            "the alias lines have to mean exactly what they meant"
        );
    }

    #[test]
    fn a_crlf_file_gets_crlf_lines() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "# windows\r\nlocal=127.0.0.1:8181\r\n");
        let spec = found("studio", "127.0.0.1", 11434, NodeEngine::Ollama);
        append_node(&path, &spec, &base_of(&path), COMMENT).expect("appends");
        let after = fs::read_to_string(&path).expect("read back");
        assert!(after.ends_with("\r\nstudio=ollama://127.0.0.1:11434\r\n"), "{after:?}");
        assert!(!after.contains("\n\n"), "no bare LF may be introduced: {after:?}");

        let lf_dir = tempfile::tempdir().expect("temp dir");
        let lf = write(&lf_dir, "local=127.0.0.1:8181\n");
        append_node(&lf, &spec, &base_of(&lf), COMMENT).expect("appends");
        let after = fs::read_to_string(&lf).expect("read back");
        assert!(!after.contains('\r'), "{after:?}");
    }

    /// The person agreed to add a line to the file they were shown. If it is
    /// not that file any more, what they agreed to is not what would be written.
    #[test]
    fn a_file_changed_since_the_scan_is_refused_and_untouched() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "local=127.0.0.1:8181\n");
        let stale = base_of(&path);
        fs::write(&path, "local=127.0.0.1:8181\nother=127.0.0.1:8182\n").expect("edit");
        let hash = file_sha256(&path).expect("hashes");

        let spec = found("studio", "127.0.0.1", 11434, NodeEngine::Ollama);
        assert_eq!(
            append_node(&path, &spec, &stale, COMMENT).expect_err("refused"),
            AppendRefusal::FileChanged
        );
        assert_eq!(file_sha256(&path).expect("hashes"), hash, "nothing was written");
        assert_eq!(temp_files(&dir), ["nodes"], "no temp file may be left behind");
    }

    /// A line added to a file the proxy cannot read would never take effect.
    #[test]
    fn a_file_that_does_not_parse_now_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "not a spec\n");
        let spec = found("studio", "127.0.0.1", 11434, NodeEngine::Ollama);
        let refusal = append_node(&path, &spec, &base_of(&path), COMMENT).expect_err("refused");
        assert_eq!(refusal.code(), "file_does_not_parse");
        assert!(refusal.to_string().contains("needs a label"), "{refusal}");
        assert_eq!(fs::read_to_string(&path).expect("read"), "not a spec\n");
    }

    /// The loader refuses a file naming no nodes, which is right for serving
    /// and wrong for adding the first one. Reusing that rule here would make a
    /// comments-only file the one file nothing could ever be added to.
    #[test]
    fn the_first_node_can_be_joined_to_a_comments_only_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "# machines go here\n\n");
        assert!(
            load_node_file(&path).is_err(),
            "the loader refuses this file, which is the point of the test"
        );
        let spec = found("studio", "127.0.0.1", 11434, NodeEngine::Ollama);
        append_node(&path, &spec, &base_of(&path), COMMENT).expect("takes its first node");
        assert_eq!(labels(&load_node_file(&path).expect("loads now")), ["studio"]);
    }

    #[test]
    fn the_cli_creates_a_missing_nodes_file_only_when_absent() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("nodes");
        let spec = found("studio", "127.0.0.1", 11434, NodeEngine::Ollama);
        let appended = append_node(&path, &spec, &Base::Absent, COMMENT).expect("creates");
        assert_eq!(appended.sha256_before, None);
        assert_eq!(
            fs::read_to_string(&path).expect("read"),
            format!("{COMMENT}\nstudio=ollama://127.0.0.1:11434\n")
        );

        // A file that appeared in the meantime is reported, never overwritten.
        let raced = dir.path().join("raced");
        fs::write(&raced, "local=127.0.0.1:8181\n").expect("someone else wrote it");
        assert_eq!(
            append_node(&raced, &spec, &Base::Absent, COMMENT).expect_err("refused"),
            AppendRefusal::FileChanged
        );
        assert_eq!(fs::read_to_string(&raced).expect("read"), "local=127.0.0.1:8181\n");
    }

    #[test]
    fn a_duplicate_label_is_refused_with_the_loaders_own_message() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "studio=ollama://127.0.0.1:11434\n");
        let spec = found("studio", "127.0.0.1", 1234, NodeEngine::LmStudio);
        let refusal = append_node(&path, &spec, &base_of(&path), COMMENT).expect_err("refused");
        assert_eq!(refusal.code(), "duplicate_label");
        assert!(refusal.to_string().contains("is used more than once"), "{refusal}");
    }

    /// The backstop. Even if every grammar above it were removed, bytes that
    /// would mean more than the one node asked for are not written.
    #[test]
    fn an_append_that_would_add_anything_but_the_requested_node_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "local=127.0.0.1:8181\n");
        let hash = file_sha256(&path).expect("hashes");
        let spec = found("studio", "127.0.0.1", 11434, NodeEngine::Ollama);

        for smuggled in [
            "# answered like ollama 0.1\nx=camelid://169.254.0.9:8181",
            "# answered like ollama 0.1\r\nx=camelid://169.254.0.9:8181",
            "# ok\nalias llama=local:other",
            "not a comment at all",
        ] {
            let refusal =
                append_node(&path, &spec, &Base::Sha256(hash.clone()), smuggled).expect_err(
                    "a comment that carries a second line must never be written",
                );
            assert_eq!(refusal.code(), "invalid_spec", "{smuggled:?}: {refusal}");
            assert_eq!(
                file_sha256(&path).expect("hashes"),
                hash,
                "{smuggled:?} changed the file"
            );
        }
        assert_eq!(temp_files(&dir), ["nodes"], "no temp file may be left behind");
    }

    /// The exit criterion: the nodes file is the only thing written.
    #[test]
    fn joining_writes_nothing_but_the_nodes_file() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "local=127.0.0.1:8181\n");
        let before = temp_files(&dir);
        let spec = found("studio", "127.0.0.1", 11434, NodeEngine::Ollama);
        append_node(&path, &spec, &base_of(&path), COMMENT).expect("appends");
        assert_eq!(
            temp_files(&dir),
            before,
            "a join may add no lock file, no backup and no leftover temp"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_nodes_file_stays_a_symlink() {
        let dir = tempfile::tempdir().expect("temp dir");
        let target = write(&dir, "local=127.0.0.1:8181\n");
        let link = dir.path().join("nodes-link");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");

        let spec = found("studio", "127.0.0.1", 11434, NodeEngine::Ollama);
        let base = Base::Sha256(file_sha256(&link).expect("hashes through the link"));
        append_node(&link, &spec, &base, COMMENT).expect("appends");

        assert!(
            fs::symlink_metadata(&link).expect("stat").file_type().is_symlink(),
            "the rename replaced the operator's symlink with a regular file"
        );
        assert!(
            fs::read_to_string(&target).expect("read target").contains("studio="),
            "the target is where the line belongs"
        );
    }

    /// A write the running proxy does not pick up is a write that did nothing.
    #[test]
    fn a_joined_node_is_picked_up_by_a_running_node_set() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "local=127.0.0.1:8181\n");
        let set = NodeSet::from_file_every(path.clone(), Duration::ZERO).expect("load");
        let (_, before) = set.current();

        let spec = found("studio", "127.0.0.1", 11434, NodeEngine::Ollama);
        append_node(&path, &spec, &base_of(&path), COMMENT).expect("appends");

        let (specs, after) = set.current();
        assert_eq!(labels(&specs), ["local", "studio"]);
        assert_ne!(before, after, "the set has to know it changed");
    }

    /// Without the cross-process lock, an editor saving at the same instant and
    /// this write would each overwrite the other's whole file.
    #[cfg(unix)]
    #[test]
    fn a_join_waits_for_a_held_file_lock() {
        use std::os::unix::io::AsRawFd;
        use std::sync::mpsc;

        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "local=127.0.0.1:8181\n");
        let base = base_of(&path);

        let held = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("open for locking");
        // SAFETY: the descriptor is owned by `held` and outlives the call.
        assert_eq!(unsafe { libc::flock(held.as_raw_fd(), libc::LOCK_EX) }, 0);

        let (done, waiting) = mpsc::channel();
        let joining = {
            let path = path.clone();
            std::thread::spawn(move || {
                let spec = found("studio", "127.0.0.1", 11434, NodeEngine::Ollama);
                let outcome = append_node(&path, &spec, &base, COMMENT);
                let _ = done.send(());
                outcome
            })
        };

        assert!(
            waiting.recv_timeout(Duration::from_millis(300)).is_err(),
            "the append got past a lock another process was holding"
        );
        drop(held);
        joining.join().expect("thread").expect("appends once released");
        assert_eq!(labels(&load_node_file(&path).expect("loads")), ["local", "studio"]);
    }

    /// Two labels for one server is a legitimate hand-written arrangement — it
    /// is how one engine is compared with itself — so the loader keeps taking
    /// it. The rule against adding a second one lives in the join path.
    #[test]
    fn two_labels_for_one_server_written_by_hand_still_load() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(
            &dir,
            "a=ollama://127.0.0.1:11434\nb=ollama://127.0.0.1:11434\n",
        );
        assert_eq!(labels(&load_node_file(&path).expect("loads")), ["a", "b"]);
        assert_eq!(labels(&NodeSet::from_file(path).expect("load").current().0), ["a", "b"]);
    }

    #[test]
    fn a_second_label_for_an_existing_endpoint_is_refused() {
        let loopback: IpAddr = "127.0.0.1".parse().expect("ip");
        let other: IpAddr = "100.64.0.37".parse().expect("ip");
        let existing = vec![(
            found("local", "localhost", 11434, NodeEngine::Ollama),
            vec![loopback],
        )];

        // Spelled differently, resolving to the same socket: one server.
        assert_eq!(
            endpoint_conflict(&existing, 11434, &[loopback]),
            Some("local".to_string())
        );
        // A different port on the same machine is a different node.
        assert_eq!(endpoint_conflict(&existing, 1234, &[loopback]), None);
        // A different machine on the same port is a different node.
        assert_eq!(endpoint_conflict(&existing, 11434, &[other]), None);
    }

    /// Clones share one set, or two requests would disagree about which
    /// machines exist.
    #[test]
    fn a_clone_sees_the_same_set() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write(&dir, "a=127.0.0.1:8181\n");
        let set = NodeSet::from_file_every(path.clone(), Duration::ZERO).expect("load");
        let clone = set.clone();

        fs::write(&path, TWO).expect("rewrite");
        let (seen_by_clone, clone_generation) = clone.current();
        let (seen_by_original, original_generation) = set.current();

        assert_eq!(labels(&seen_by_clone), vec!["a", "b"]);
        assert_eq!(labels(&seen_by_original), vec!["a", "b"]);
        assert_eq!(clone_generation, original_generation);
    }
}
