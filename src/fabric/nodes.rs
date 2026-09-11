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

use std::fs;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::aliases::{parse_model_aliases, ModelAliases, ALIAS_PREFIX};
use super::node::{parse_fabric, NodeSpec};
use super::watch::{Change, WatchedFile};

/// How long a loaded set is trusted before the file is looked at again.
///
/// A node joining or leaving is only noticed this fast. A second is short
/// enough that an operator does not wait on it and long enough that a busy
/// proxy is not stat-ing a file on every placement.
pub(crate) const DEFAULT_NODE_RELOAD_INTERVAL: Duration = Duration::from_secs(1);

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
            inner: Arc::new(Inner {
                source: Source::Fixed(Arc::from(specs)),
                generation: AtomicU64::new(0),
            }),
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
            inner: Arc::new(Inner {
                source: Source::File(WatchedFile::new(path, interval, specs)),
                generation: AtomicU64::new(0),
            }),
        })
    }

    /// Whether the set changes without a restart.
    pub(crate) fn is_reloadable(&self) -> bool {
        matches!(self.inner.source, Source::File(_))
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

/// Read and validate a node file into the shape the set holds.
fn load_specs(path: &Path) -> Result<Arc<[NodeSpec]>> {
    load_node_file(path).map(Arc::from)
}

fn meaningful_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
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
