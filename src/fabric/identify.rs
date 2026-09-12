//! What a stranger's address is running, judged only from what it answered.
//!
//! This is deliberately *not* [`super::probe`]. A probe reads a node an operator
//! already declared, so it is lenient on purpose: every field defaults, and a
//! renamed field cannot drop a declared machine out of the fabric. Asked to
//! judge a stranger, that same leniency would read `{}` as a match — an empty
//! object satisfies every defaulted field of every engine at once.
//!
//! So the two jobs get opposite defaults. Identification requires each field it
//! names to be present and of the right type, and an engine that does not
//! produce its whole signature is **ruled out**, never assumed.
//!
//! The other rule this module exists to hold: *a check that never finished is
//! not a non-match*. If one engine's path timed out while another engine
//! matched, what we have is a match and an unfinished rival check, which is
//! [`Classification::Incomplete`] — because the timed-out rival could have
//! matched too, and acting on the lucky one is how a confirmed machine ends up
//! being something else entirely.

use std::collections::{BTreeMap, BTreeSet};

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use serde_json::Value;

use super::engine::NodeEngine;

/// Written wherever an engine reported a version this build will not repeat.
pub(crate) const VERSION_NOT_RECORDED: &str = "version not recorded";

/// Longest version string this build will repeat.
const MAX_VERSION_BYTES: usize = 64;

/// The version, only if it is one this build is willing to write into an
/// operator's file and print on their terminal.
///
/// A version arrives from a machine nobody has identified yet, and it is
/// repeated in two places that parse: the provenance comment in the nodes file,
/// which is line-oriented, and a terminal, which acts on escape sequences. A
/// grammar of "the characters versions are actually made of" costs nothing real
/// — `0.33.2`, `v0.7.2-551`, `1.0.0+build.4` all pass — and closes both.
pub(crate) fn recorded_version(raw: &str) -> Option<&str> {
    if raw.is_empty() || raw.len() > MAX_VERSION_BYTES {
        return None;
    }
    let mut characters = raw.chars();
    let first = characters.next()?;
    if !first.is_ascii_alphanumeric() {
        return None;
    }
    characters
        .all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '+' | '_' | '-')
        })
        .then_some(raw)
}

/// What happened when one path was asked, in the terms a verdict turns on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Outcome {
    /// A complete HTTP answer. `json` is `Some` only for a body that parsed as
    /// JSON; the body itself is never kept beyond that.
    Http {
        status: u16,
        content_type: Option<String>,
        json: Option<Value>,
        bytes_len: usize,
    },
    /// The request never finished: no connection, no answer within the budget,
    /// a truncated frame. This is the one outcome that leaves a verdict open.
    Unanswered(String),
    /// Something answered, and it was not HTTP.
    NotHttp(String),
    /// The connection was accepted and nothing was ever written back.
    Silent,
    /// TLS did not authenticate, so nothing above it was ever read.
    TlsRefused(String),
}

/// One path, and what came back from it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Answer {
    pub(crate) path: String,
    pub(crate) outcome: Outcome,
}

impl Answer {
    pub(crate) fn new(path: impl Into<String>, outcome: Outcome) -> Self {
        Self {
            path: path.into(),
            outcome,
        }
    }
}

/// Everything one address answered, indexed by path for the engine adapters.
pub(crate) struct Answers<'a> {
    answers: &'a [Answer],
}

/// What one engine's adapter sees when it reads one of its own paths.
pub(crate) enum PathRead<'a> {
    /// A 200 whose body parsed as JSON. The only shape a signature can match.
    Json(&'a Value),
    /// A finished answer that no signature can match. Carries what to say.
    Finished(String),
    /// 401 or 403: a finished answer that withholds what was asked for.
    Withheld,
    /// Nothing came back, so this path settles nothing either way.
    Unanswered,
}

impl<'a> Answers<'a> {
    pub(crate) fn of(answers: &'a [Answer]) -> Self {
        Self { answers }
    }

    pub(crate) fn read(&self, path: &str) -> PathRead<'a> {
        let Some(answer) = self.answers.iter().find(|answer| answer.path == path) else {
            return PathRead::Unanswered;
        };
        match &answer.outcome {
            Outcome::Http { status: 401 | 403, .. } => PathRead::Withheld,
            Outcome::Http {
                status: 200,
                json: Some(value),
                ..
            } => PathRead::Json(value),
            Outcome::Http {
                status: 200,
                content_type,
                ..
            } => PathRead::Finished(format!(
                "{path} answered 200 with {}",
                content_type.as_deref().unwrap_or("a body that is not JSON")
            )),
            Outcome::Http { status, .. } => {
                PathRead::Finished(format!("{path} answered HTTP {status}"))
            }
            Outcome::NotHttp(detail) => PathRead::Finished(format!("{path}: {detail}")),
            Outcome::Silent => {
                PathRead::Finished(format!("{path} was accepted and never answered"))
            }
            // A refused handshake and a dead socket settle the same amount:
            // nothing above the transport was ever read.
            Outcome::TlsRefused(_) | Outcome::Unanswered(_) => PathRead::Unanswered,
        }
    }
}

/// One engine's reading of one of its own paths, before the paths are combined.
pub(crate) enum PathVerdict {
    /// This path matched the engine's signature, with the fact we wrote for it.
    Signature(String),
    RuledOut(String),
    Withheld,
    Unanswered,
}

/// One path of an engine's signature, and what we are willing to say about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PathFact {
    pub(crate) path: String,
    pub(crate) fact: String,
}

/// What one engine concluded about one address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EngineVerdict {
    /// Every path of this engine's signature matched.
    Matched {
        /// As the engine reported it. Repeated only through
        /// [`recorded_version`], never raw.
        version: Option<String>,
        facts: Vec<PathFact>,
    },
    /// A finished answer that this engine's signature cannot match.
    RuledOut { by: String },
    /// One of this engine's paths never answered, so it is neither.
    Undecided { unanswered: Vec<String> },
    /// A path this engine needs answered 401 or 403.
    Withheld { paths: Vec<String> },
    /// It answered as something this fabric knows and does not place on.
    NotANode { reason: String },
}

impl EngineVerdict {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Matched { .. } => "matched",
            Self::RuledOut { .. } => "ruled_out",
            Self::Undecided { .. } => "undecided",
            Self::Withheld { .. } => "withheld",
            Self::NotANode { .. } => "not_a_node",
        }
    }

    /// The version, only where it passes the grammar this build will repeat.
    pub(crate) fn recorded_version(&self) -> Option<&str> {
        match self {
            Self::Matched { version, .. } => version.as_deref().and_then(recorded_version),
            _ => None,
        }
    }

    /// The version exactly as the engine reported it. Only the comment builder
    /// and the confirm preview see this, and both pass it through
    /// [`recorded_version`] before it reaches a file or a terminal.
    pub(crate) fn reported_version(&self) -> Option<&str> {
        match self {
            Self::Matched { version, .. } => version.as_deref(),
            _ => None,
        }
    }

    fn detail(&self) -> Option<&str> {
        match self {
            Self::RuledOut { by } => Some(by),
            Self::NotANode { reason } => Some(reason),
            _ => None,
        }
    }
}

/// Combine one engine's per-path readings into its verdict.
///
/// The order is the whole rule. A finished answer that rules the engine out
/// wins over a path that never answered, because no answer to *that* request
/// could have made this engine match. Without that, one slow port would make
/// every scan permanently inconclusive.
pub(crate) fn from_paths(reads: &[(&str, PathVerdict)], version: Option<&str>) -> EngineVerdict {
    for (_, read) in reads {
        if let PathVerdict::RuledOut(by) = read {
            return EngineVerdict::RuledOut { by: by.clone() };
        }
    }
    let unanswered: Vec<String> = reads
        .iter()
        .filter(|(_, read)| matches!(read, PathVerdict::Unanswered))
        .map(|(path, _)| (*path).to_string())
        .collect();
    if !unanswered.is_empty() {
        return EngineVerdict::Undecided { unanswered };
    }
    let withheld: Vec<String> = reads
        .iter()
        .filter(|(_, read)| matches!(read, PathVerdict::Withheld))
        .map(|(path, _)| (*path).to_string())
        .collect();
    if !withheld.is_empty() {
        return EngineVerdict::Withheld { paths: withheld };
    }
    EngineVerdict::Matched {
        version: version.map(str::to_string),
        facts: reads
            .iter()
            .filter_map(|(path, read)| match read {
                PathVerdict::Signature(fact) => Some(PathFact {
                    path: (*path).to_string(),
                    fact: fact.clone(),
                }),
                _ => None,
            })
            .collect(),
    }
}

/// One engine's verdict, as it is reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineReport {
    pub engine: NodeEngine,
    pub(crate) verdict: EngineVerdict,
}

impl Serialize for EngineReport {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("engine", &self.engine)?;
        map.serialize_entry("verdict", self.verdict.as_str())?;
        // Only ever the grammar-checked form: this value is read by a UI and
        // echoed back to a person.
        if let Some(version) = self.verdict.recorded_version() {
            map.serialize_entry("version", version)?;
        }
        if let Some(detail) = self.verdict.detail() {
            map.serialize_entry("detail", detail)?;
        }
        map.end()
    }
}

/// One request, and what it established. Every `fact` is a sentence this build
/// wrote; no third-party body is echoed into it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Evidence {
    pub request: String,
    pub status: Option<u16>,
    pub content_type: Option<String>,
    pub fact: String,
    pub matched: Vec<NodeEngine>,
}

/// What this build is willing to say one address is.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Classification {
    /// Exactly one engine matched, and no rival check was left unfinished.
    AnswersLike {
        engine: NodeEngine,
        version: Option<String>,
        /// Paths another engine needed that answered 401 or 403. A finished
        /// answer, so it does not block this match — it is reported because it
        /// is the reason those engines were not ruled out on their merits.
        withheld_elsewhere: Vec<String>,
    },
    /// Two engines matched. A person picks; the first is never taken.
    Ambiguous { engines: Vec<NodeEngine> },
    /// This fabric's own proxy. Not a node, and adding it would place work on
    /// a machine twice over.
    FabricProxy { reason: String },
    /// A check did not finish, so nothing is concluded — even beside a match.
    Incomplete {
        unanswered: Vec<String>,
        matched_so_far: Vec<NodeEngine>,
    },
    /// Nothing matched, and something asked for a credential.
    RequiresCredentials { paths: Vec<String> },
    /// It speaks HTTP, and no signature this build knows matched. That is all
    /// this says. It is never called an unknown engine.
    OtherHttp { statuses: BTreeMap<String, u16> },
    NotHttp { detail: String },
    SilentAfterConnect,
    TlsNotAuthenticated { detail: String },
}

impl Classification {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::AnswersLike { .. } => "answers_like",
            Self::Ambiguous { .. } => "ambiguous",
            Self::FabricProxy { .. } => "fabric_proxy",
            Self::Incomplete { .. } => "incomplete",
            Self::RequiresCredentials { .. } => "requires_credentials",
            Self::OtherHttp { .. } => "other_http",
            Self::NotHttp { .. } => "not_http",
            Self::SilentAfterConnect => "silent_after_connect",
            Self::TlsNotAuthenticated { .. } => "tls_not_authenticated",
        }
    }

    /// The single engine this address answered like, where there is one.
    pub(crate) fn matched_engine(&self) -> Option<NodeEngine> {
        match self {
            Self::AnswersLike { engine, .. } => Some(*engine),
            _ => None,
        }
    }
}

/// Every engine's verdict, the conclusion, and what each request established.
#[derive(Debug, Clone, PartialEq)]
pub struct Identification {
    pub engines: Vec<EngineReport>,
    pub classification: Classification,
    pub evidence: Vec<Evidence>,
}

/// Judge one address from its answers. Pure.
pub(crate) fn classify(answers: &[Answer]) -> Identification {
    let reads = Answers::of(answers);
    let engines: Vec<EngineReport> = NodeEngine::ALL
        .iter()
        .map(|engine| EngineReport {
            engine: *engine,
            verdict: engine.identify(&reads),
        })
        .collect();
    let classification = conclude(answers, &engines);
    let evidence = evidence_of(answers, &engines);
    Identification {
        engines,
        classification,
        evidence,
    }
}

fn conclude(answers: &[Answer], engines: &[EngineReport]) -> Classification {
    // Connection-level first. Nothing above the transport was ever read, so no
    // engine verdict over these answers means anything.
    if let Some(kind) = connection_level(answers) {
        return kind;
    }

    if let Some(reason) = engines.iter().find_map(|report| match &report.verdict {
        EngineVerdict::NotANode { reason } => Some(reason.clone()),
        _ => None,
    }) {
        return Classification::FabricProxy { reason };
    }

    let matched: Vec<NodeEngine> = engines
        .iter()
        .filter(|report| matches!(report.verdict, EngineVerdict::Matched { .. }))
        .map(|report| report.engine)
        .collect();

    // Before any match is acted on: an unfinished rival check means the match
    // is one of possibly several, and this build will not pick between a
    // signature it saw and one it never got to look at.
    let unanswered: Vec<String> = sorted_unique(engines.iter().flat_map(|report| {
        match &report.verdict {
            EngineVerdict::Undecided { unanswered } => unanswered.clone(),
            _ => Vec::new(),
        }
    }));
    if !unanswered.is_empty() {
        return Classification::Incomplete {
            unanswered,
            matched_so_far: matched,
        };
    }

    let withheld: Vec<String> = sorted_unique(engines.iter().flat_map(|report| {
        match &report.verdict {
            EngineVerdict::Withheld { paths } => paths.clone(),
            _ => Vec::new(),
        }
    }));

    match matched.as_slice() {
        [engine] => {
            let version = engines
                .iter()
                .find(|report| report.engine == *engine)
                .and_then(|report| report.verdict.recorded_version())
                .map(str::to_string);
            Classification::AnswersLike {
                engine: *engine,
                version,
                withheld_elsewhere: withheld,
            }
        }
        [] => {
            if !withheld.is_empty() {
                return Classification::RequiresCredentials { paths: withheld };
            }
            Classification::OtherHttp {
                statuses: answers
                    .iter()
                    .filter_map(|answer| match &answer.outcome {
                        Outcome::Http { status, .. } => Some((answer.path.clone(), *status)),
                        _ => None,
                    })
                    .collect(),
            }
        }
        _ => Classification::Ambiguous { engines: matched },
    }
}

/// The kinds that are decided by the connection rather than by a body.
fn connection_level(answers: &[Answer]) -> Option<Classification> {
    if answers.is_empty() {
        return Some(Classification::SilentAfterConnect);
    }
    if answers
        .iter()
        .any(|answer| matches!(answer.outcome, Outcome::Http { .. }))
    {
        return None;
    }
    if let Some(detail) = answers.iter().find_map(|answer| match &answer.outcome {
        Outcome::TlsRefused(detail) => Some(detail.clone()),
        _ => None,
    }) {
        return Some(Classification::TlsNotAuthenticated { detail });
    }
    if let Some(detail) = answers.iter().find_map(|answer| match &answer.outcome {
        Outcome::NotHttp(detail) => Some(detail.clone()),
        _ => None,
    }) {
        return Some(Classification::NotHttp { detail });
    }
    if answers
        .iter()
        .all(|answer| matches!(answer.outcome, Outcome::Silent))
    {
        return Some(Classification::SilentAfterConnect);
    }
    None
}

fn evidence_of(answers: &[Answer], engines: &[EngineReport]) -> Vec<Evidence> {
    answers
        .iter()
        .map(|answer| {
            let mut fact = None;
            let mut matched = Vec::new();
            for report in engines {
                let EngineVerdict::Matched { facts, .. } = &report.verdict else {
                    continue;
                };
                if let Some(found) = facts.iter().find(|entry| entry.path == answer.path) {
                    fact.get_or_insert_with(|| found.fact.clone());
                    matched.push(report.engine);
                }
            }
            let (status, content_type) = match &answer.outcome {
                Outcome::Http {
                    status,
                    content_type,
                    ..
                } => (Some(*status), content_type.clone()),
                _ => (None, None),
            };
            Evidence {
                request: format!("GET {}", answer.path),
                status,
                content_type,
                fact: fact.unwrap_or_else(|| plain_fact(answer)),
                matched,
            }
        })
        .collect()
}

/// What we are willing to say about an answer nothing matched. Written here, so
/// it can never become an echo of somebody else's body.
fn plain_fact(answer: &Answer) -> String {
    match &answer.outcome {
        Outcome::Http {
            status: 401 | 403, ..
        } => "a credential was asked for".to_string(),
        Outcome::Http {
            status,
            json: Some(_),
            bytes_len,
            ..
        } => format!("HTTP {status} with {bytes_len} bytes of JSON no signature here matched"),
        Outcome::Http {
            status,
            content_type,
            bytes_len,
            ..
        } => format!(
            "HTTP {status} with {bytes_len} bytes of {}",
            content_type.as_deref().unwrap_or("an unnamed media type")
        ),
        Outcome::Unanswered(detail) => format!("no answer: {detail}"),
        Outcome::NotHttp(detail) => format!("not HTTP: {detail}"),
        Outcome::Silent => "accepted the connection and sent nothing".to_string(),
        Outcome::TlsRefused(detail) => format!("TLS did not authenticate: {detail}"),
    }
}

fn sorted_unique(values: impl Iterator<Item = String>) -> Vec<String> {
    values.collect::<BTreeSet<String>>().into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const HEALTH: &str = "/v1/health";
    const VERSION: &str = "/api/version";
    const TAGS: &str = "/api/tags";
    const MODELS: &str = "/api/v0/models";
    const EVERY_PATH: [&str; 4] = [HEALTH, VERSION, TAGS, MODELS];

    fn json_answer(path: &str, body: Value) -> Answer {
        let bytes_len = body.to_string().len();
        Answer::new(
            path,
            Outcome::Http {
                status: 200,
                content_type: Some("application/json".to_string()),
                json: Some(body),
                bytes_len,
            },
        )
    }

    fn status_answer(path: &str, status: u16) -> Answer {
        Answer::new(
            path,
            Outcome::Http {
                status,
                content_type: Some("application/json".to_string()),
                json: None,
                bytes_len: 0,
            },
        )
    }

    fn timed_out(path: &str) -> Answer {
        Answer::new(path, Outcome::Unanswered("timed out".to_string()))
    }

    /// The body a real `camelid serve` answers `/v1/health` with, reduced to
    /// the fields the signature names.
    fn camelid_health() -> Value {
        json!({
            "ok": true,
            "engine": "camelid",
            "generation_ready": true,
            "active_model_id": "llama-3.2-1b-instruct",
            "version": "v0.7.2-551",
            "backend": "metal",
            "engine_queue_depth": 0,
            "engine_queued_tasks": 0
        })
    }

    /// Every path answered `status`, except the ones given.
    fn everything_else(status: u16, given: Vec<Answer>) -> Vec<Answer> {
        let mut answers = given;
        for path in EVERY_PATH {
            if !answers.iter().any(|answer| answer.path == path) {
                answers.push(status_answer(path, status));
            }
        }
        answers
    }

    fn verdict_of(identification: &Identification, engine: NodeEngine) -> &EngineVerdict {
        &identification
            .engines
            .iter()
            .find(|report| report.engine == engine)
            .expect("every engine is reported")
            .verdict
    }

    #[test]
    fn a_camelid_health_answer_matches_only_camelid() {
        let identification = classify(&everything_else(
            404,
            vec![json_answer(HEALTH, camelid_health())],
        ));
        assert_eq!(
            identification.classification,
            Classification::AnswersLike {
                engine: NodeEngine::Camelid,
                version: Some("v0.7.2-551".to_string()),
                withheld_elsewhere: Vec::new(),
            }
        );
        for other in [NodeEngine::Ollama, NodeEngine::LmStudio] {
            assert!(
                matches!(verdict_of(&identification, other), EngineVerdict::RuledOut { .. }),
                "{other} must be ruled out by its own 404s"
            );
        }
    }

    /// A node started with an API key answers 401 on every route but its
    /// health. Reading that as "unfinished" would make every keyed Camelid in
    /// existence unclassifiable.
    #[test]
    fn a_keyed_camelid_is_answers_like_camelid_despite_401s_elsewhere() {
        let identification = classify(&everything_else(
            401,
            vec![json_answer(HEALTH, camelid_health())],
        ));
        assert_eq!(
            identification.classification,
            Classification::AnswersLike {
                engine: NodeEngine::Camelid,
                version: Some("v0.7.2-551".to_string()),
                withheld_elsewhere: vec![
                    TAGS.to_string(),
                    MODELS.to_string(),
                    VERSION.to_string(),
                ],
            }
        );
    }

    #[test]
    fn a_fabric_proxy_is_classified_as_a_proxy_and_never_proposed() {
        let identification = classify(&everything_else(
            404,
            vec![json_answer(
                HEALTH,
                json!({"ok": true, "service": "camelid-fabric", "version": "0.7.2", "ready": true}),
            )],
        ));
        assert_eq!(identification.classification.kind(), "fabric_proxy");
        assert_eq!(identification.classification.matched_engine(), None);
    }

    #[test]
    fn ollama_needs_both_version_and_tags() {
        let both = classify(&everything_else(
            404,
            vec![
                json_answer(VERSION, json!({"version": "0.33.2"})),
                json_answer(TAGS, json!({"models": [{"name": "llama3.2:latest"}]})),
            ],
        ));
        assert_eq!(
            both.classification,
            Classification::AnswersLike {
                engine: NodeEngine::Ollama,
                version: Some("0.33.2".to_string()),
                withheld_elsewhere: Vec::new(),
            }
        );

        let only_version = classify(&everything_else(
            404,
            vec![json_answer(VERSION, json!({"version": "0.33.2"}))],
        ));
        assert_eq!(only_version.classification.kind(), "other_http");
    }

    /// The probe reads `{}` as "an Ollama with no models". Identification must
    /// read it as nothing at all.
    #[test]
    fn a_lenient_probe_payload_is_not_an_identification() {
        let identification = classify(&everything_else(
            404,
            vec![json_answer(VERSION, json!({})), json_answer(TAGS, json!({}))],
        ));
        assert_eq!(identification.classification.kind(), "other_http");
        assert_eq!(identification.classification.matched_engine(), None);
    }

    #[test]
    fn an_lm_studio_style_error_body_is_not_a_match() {
        let body = json!({"error": "Unexpected endpoint or method."});
        let answers: Vec<Answer> = EVERY_PATH
            .iter()
            .map(|path| json_answer(path, body.clone()))
            .collect();
        assert_eq!(classify(&answers).classification.kind(), "other_http");
    }

    #[test]
    fn two_matching_signatures_are_ambiguous_not_the_first() {
        let identification = classify(&everything_else(
            404,
            vec![
                json_answer(HEALTH, camelid_health()),
                json_answer(VERSION, json!({"version": "0.33.2"})),
                json_answer(TAGS, json!({"models": [{"name": "a"}]})),
            ],
        ));
        assert_eq!(
            identification.classification,
            Classification::Ambiguous {
                engines: vec![NodeEngine::Camelid, NodeEngine::Ollama],
            }
        );
        assert_eq!(
            identification.classification.matched_engine(),
            None,
            "an ambiguous address names no single engine, so nothing can be proposed from it"
        );
    }

    #[test]
    fn a_timeout_on_one_path_is_incomplete_not_other_http() {
        let identification = classify(&everything_else(
            404,
            vec![timed_out(VERSION), timed_out(TAGS)],
        ));
        assert_eq!(
            identification.classification,
            Classification::Incomplete {
                unanswered: vec![TAGS.to_string(), VERSION.to_string()],
                matched_so_far: Vec::new(),
            }
        );
    }

    #[test]
    fn a_match_with_an_unfinished_rival_check_is_incomplete_not_answers_like() {
        let identification = classify(&everything_else(
            404,
            vec![
                json_answer(HEALTH, camelid_health()),
                timed_out(VERSION),
                timed_out(TAGS),
            ],
        ));
        assert_eq!(
            identification.classification,
            Classification::Incomplete {
                unanswered: vec![TAGS.to_string(), VERSION.to_string()],
                matched_so_far: vec![NodeEngine::Camelid],
            },
            "a rival whose check never finished is not a rival that was ruled out"
        );
    }

    /// The paired limit on the rule above: a rival already ruled out by a
    /// finished answer stays ruled out, because no answer to its other request
    /// could have changed that.
    #[test]
    fn a_rival_ruled_out_by_a_finished_answer_does_not_block_a_match() {
        let identification = classify(&everything_else(
            404,
            vec![json_answer(HEALTH, camelid_health()), timed_out(TAGS)],
        ));
        assert_eq!(
            identification.classification,
            Classification::AnswersLike {
                engine: NodeEngine::Camelid,
                version: Some("v0.7.2-551".to_string()),
                withheld_elsewhere: Vec::new(),
            }
        );
    }

    #[test]
    fn a_401_with_no_match_requires_credentials() {
        let identification =
            classify(&everything_else(404, vec![status_answer(MODELS, 401)]));
        assert_eq!(
            identification.classification,
            Classification::RequiresCredentials {
                paths: vec![MODELS.to_string()],
            }
        );
        assert_eq!(identification.classification.matched_engine(), None);
    }

    #[test]
    fn an_html_page_is_other_http_with_no_proposal() {
        let page = "<!doctype html><title>Directory listing</title>";
        let answers: Vec<Answer> = EVERY_PATH
            .iter()
            .map(|path| {
                Answer::new(
                    *path,
                    Outcome::Http {
                        status: 200,
                        content_type: Some("text/html".to_string()),
                        json: None,
                        bytes_len: page.len(),
                    },
                )
            })
            .collect();
        let identification = classify(&answers);
        assert_eq!(identification.classification.kind(), "other_http");
        assert_eq!(identification.classification.matched_engine(), None);
        for evidence in &identification.evidence {
            assert!(
                !evidence.fact.contains("Directory listing"),
                "a stranger's body must never be echoed back: {}",
                evidence.fact
            );
        }
    }

    #[test]
    fn a_non_http_reply_is_not_http_and_a_silent_accept_is_silent() {
        let banner: Vec<Answer> = EVERY_PATH
            .iter()
            .map(|path| {
                Answer::new(
                    *path,
                    Outcome::NotHttp("answered SSH-2.0-OpenSSH_9.6".to_string()),
                )
            })
            .collect();
        assert_eq!(classify(&banner).classification.kind(), "not_http");

        let silent: Vec<Answer> = EVERY_PATH
            .iter()
            .map(|path| Answer::new(*path, Outcome::Silent))
            .collect();
        assert_eq!(
            classify(&silent).classification,
            Classification::SilentAfterConnect
        );
    }

    /// The grammar is what stands between a stranger's string and two parsers:
    /// the nodes file, which is line-oriented, and a terminal.
    #[test]
    fn a_version_this_build_will_not_repeat_is_not_recorded() {
        for good in ["0.33.2", "v0.7.2-551", "1.0.0+build.4", "a", "A_1"] {
            assert_eq!(recorded_version(good), Some(good), "{good}");
        }
        for bad in [
            "",
            "0.1\nx=camelid://169.254.0.9:8181",
            "0.1\r\nx=y",
            "\u{1b}[2J0.1",
            "-1.0",
            ".1",
            "0.1 (build 4)",
            "0.1#",
        ] {
            assert_eq!(recorded_version(bad), None, "{bad:?}");
        }
        assert_eq!(recorded_version(&"a".repeat(64)), Some("a".repeat(64).as_str()));
        assert_eq!(recorded_version(&"a".repeat(65)), None);
    }

    /// A version that fails the grammar leaves the finding rather than the
    /// whole match: what it is stays established, what it calls itself does not.
    #[test]
    fn a_hostile_version_is_dropped_without_dropping_the_match() {
        let identification = classify(&everything_else(
            404,
            vec![
                json_answer(VERSION, json!({"version": "0.1\nx=camelid://169.254.0.9:8181"})),
                json_answer(TAGS, json!({"models": [{"name": "a"}]})),
            ],
        ));
        assert_eq!(
            identification.classification,
            Classification::AnswersLike {
                engine: NodeEngine::Ollama,
                version: None,
                withheld_elsewhere: Vec::new(),
            }
        );
        let serialized = serde_json::to_value(
            identification
                .engines
                .iter()
                .find(|report| report.engine == NodeEngine::Ollama)
                .expect("ollama is reported"),
        )
        .expect("serializes");
        assert!(
            serialized.get("version").is_none(),
            "an unrecordable version is absent, never a half-escaped string: {serialized}"
        );
    }
}
