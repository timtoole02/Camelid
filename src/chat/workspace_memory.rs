use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: i64 = 6;
const MAX_THREAD_TITLE_CHARS: usize = 80;
const MAX_EVIDENCE_PER_TURN: usize = 32 * 8;
static INITIALIZE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct StoredTurn {
    pub id: i64,
    pub turn_index: u32,
    pub user_text: String,
    pub assistant_text: String,
    pub terminal_outcome: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MemoryContext {
    pub relevant: Vec<StoredTurn>,
    pub recent: Vec<StoredTurn>,
    pub evidence: Vec<StoredEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct StoredThread {
    pub id: String,
    pub title: String,
    pub canonical_root: String,
    pub model_id: String,
    pub model_sha256: String,
    pub compacted_through_turn: Option<u32>,
    pub compaction_count: u32,
    pub updated_at: i64,
    pub turn_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct CompactionResult {
    pub compacted_through_turn: Option<u32>,
    pub archived_turns: u32,
    pub compaction_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EvidenceInput {
    pub tool: String,
    pub detail: String,
    pub observation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(crate) struct StoredEvidence {
    pub tool: String,
    pub detail: String,
    pub observation: String,
    pub observation_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AppendTurn {
    Inserted(i64),
    Duplicate(i64),
}

#[derive(Clone)]
pub(crate) struct WorkspaceMemoryStore {
    path: PathBuf,
}

impl WorkspaceMemoryStore {
    pub(crate) fn open(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let _initializing = INITIALIZE_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let store = Self { path: path.into() };
        if let Some(parent) = store.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut connection = store.connect()?;
        let migrated = migrate(&mut connection)?;
        verify_schema(&connection)?;
        if migrated {
            verify_foreign_keys(&connection)?;
        }
        Ok(store)
    }

    pub(crate) fn create_thread_for_model(
        &self,
        thread_id: &str,
        canonical_root: &str,
        model_id: &str,
        model_sha256: &str,
        initial_goal: &str,
    ) -> anyhow::Result<()> {
        let connection = self.connect()?;
        let now = now_epoch_seconds();
        connection.execute(
            "INSERT INTO workspace_threads
             (id, title, canonical_root, model_id, model_sha256, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
            params![
                thread_id,
                workspace_thread_title(initial_goal),
                canonical_root,
                model_id,
                model_sha256,
                now
            ],
        )?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn create_thread(
        &self,
        thread_id: &str,
        canonical_root: &str,
        model_id: &str,
    ) -> anyhow::Result<()> {
        self.create_thread_for_model(
            thread_id,
            canonical_root,
            model_id,
            "test-model-sha256",
            "Test conversation",
        )
    }

    pub(crate) fn thread(&self, thread_id: &str) -> anyhow::Result<Option<StoredThread>> {
        let connection = self.connect()?;
        Ok(connection
            .query_row(
                "SELECT th.id, th.canonical_root, th.model_id, th.model_sha256,
                    th.compacted_through_turn,
                    (SELECT COUNT(*) FROM workspace_compactions c WHERE c.thread_id = th.id),
                    th.updated_at, COUNT(t.id), th.title
                 FROM workspace_threads AS th
                 LEFT JOIN workspace_turns AS t ON t.thread_id = th.id
                 WHERE th.id = ?1 GROUP BY th.id",
                [thread_id],
                stored_thread_from_row,
            )
            .optional()?)
    }

    pub(crate) fn threads_for_root(
        &self,
        canonical_root: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<StoredThread>> {
        let connection = self.connect()?;
        let mut statement = connection.prepare(
            "SELECT th.id, th.canonical_root, th.model_id, th.model_sha256,
                    th.compacted_through_turn,
                    (SELECT COUNT(*) FROM workspace_compactions c WHERE c.thread_id = th.id),
                    th.updated_at, COUNT(t.id), th.title
             FROM workspace_threads AS th
             LEFT JOIN workspace_turns AS t ON t.thread_id = th.id
             WHERE th.canonical_root = ?1
             GROUP BY th.id ORDER BY th.updated_at DESC LIMIT ?2",
        )?;
        let threads = statement
            .query_map(
                params![canonical_root, limit as i64],
                stored_thread_from_row,
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(threads)
    }

    pub(crate) fn recent_threads(&self, limit: usize) -> anyhow::Result<Vec<StoredThread>> {
        let connection = self.connect()?;
        let mut statement = connection.prepare(
            "SELECT th.id, th.canonical_root, th.model_id, th.model_sha256,
                    th.compacted_through_turn,
                    (SELECT COUNT(*) FROM workspace_compactions c WHERE c.thread_id = th.id),
                    th.updated_at, COUNT(t.id), th.title
             FROM workspace_threads AS th
             LEFT JOIN workspace_turns AS t ON t.thread_id = th.id
             GROUP BY th.id ORDER BY th.updated_at DESC LIMIT ?1",
        )?;
        let threads = statement
            .query_map([limit as i64], stored_thread_from_row)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(threads)
    }

    pub(crate) fn delete_thread(&self, thread_id: &str) -> anyhow::Result<bool> {
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "DELETE FROM workspace_turns_fts WHERE rowid IN
             (SELECT id FROM workspace_turns WHERE thread_id = ?1)",
            [thread_id],
        )?;
        let deleted =
            transaction.execute("DELETE FROM workspace_threads WHERE id = ?1", [thread_id])?;
        transaction.commit()?;
        Ok(deleted == 1)
    }

    pub(crate) fn compact_thread(&self, thread_id: &str) -> anyhow::Result<CompactionResult> {
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let previous: Option<u32> = transaction
            .query_row(
                "SELECT compacted_through_turn FROM workspace_threads WHERE id = ?1",
                [thread_id],
                |row| row.get(0),
            )
            .optional()?
            .flatten();
        let latest: Option<u32> = transaction.query_row(
            "SELECT MAX(turn_index) FROM workspace_turns WHERE thread_id = ?1",
            [thread_id],
            |row| row.get(0),
        )?;
        let Some(latest) = latest else {
            transaction.commit()?;
            return Ok(CompactionResult {
                compacted_through_turn: previous,
                archived_turns: 0,
                compaction_count: compaction_count(&self.connect()?, thread_id)?,
            });
        };
        let archived_turns: u32 = transaction.query_row(
            "SELECT COUNT(*) FROM workspace_turns
             WHERE thread_id = ?1 AND turn_index > COALESCE(?2, -1) AND turn_index <= ?3",
            params![thread_id, previous, latest],
            |row| row.get(0),
        )?;
        if archived_turns > 0 {
            transaction.execute(
                "INSERT INTO workspace_compactions
                 (thread_id, previous_boundary, new_boundary, created_at)
                 VALUES (?1, ?2, ?3, ?4)",
                params![thread_id, previous, latest, now_epoch_seconds()],
            )?;
            transaction.execute(
                "UPDATE workspace_threads
                 SET compacted_through_turn = ?2, updated_at = ?3 WHERE id = ?1",
                params![thread_id, latest, now_epoch_seconds()],
            )?;
        }
        transaction.commit()?;
        Ok(CompactionResult {
            compacted_through_turn: Some(latest),
            archived_turns,
            compaction_count: compaction_count(&self.connect()?, thread_id)?,
        })
    }

    pub(crate) fn undo_compaction(&self, thread_id: &str) -> anyhow::Result<CompactionResult> {
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let last = transaction
            .query_row(
                "SELECT id, previous_boundary, new_boundary FROM workspace_compactions
                 WHERE thread_id = ?1 ORDER BY id DESC LIMIT 1",
                [thread_id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Option<u32>>(1)?,
                        row.get::<_, u32>(2)?,
                    ))
                },
            )
            .optional()?;
        let Some((compaction_id, previous, boundary)) = last else {
            transaction.commit()?;
            return Ok(CompactionResult {
                compacted_through_turn: None,
                archived_turns: 0,
                compaction_count: 0,
            });
        };
        let restored: u32 = transaction.query_row(
            "SELECT COUNT(*) FROM workspace_turns
             WHERE thread_id = ?1 AND turn_index > COALESCE(?2, -1) AND turn_index <= ?3",
            params![thread_id, previous, boundary],
            |row| row.get(0),
        )?;
        transaction.execute(
            "UPDATE workspace_threads
             SET compacted_through_turn = ?2, updated_at = ?3 WHERE id = ?1",
            params![thread_id, previous, now_epoch_seconds()],
        )?;
        transaction.execute(
            "DELETE FROM workspace_compactions WHERE id = ?1",
            [compaction_id],
        )?;
        transaction.commit()?;
        Ok(CompactionResult {
            compacted_through_turn: previous,
            archived_turns: restored,
            compaction_count: compaction_count(&self.connect()?, thread_id)?,
        })
    }

    #[cfg(test)]
    pub(crate) fn append_turn(
        &self,
        thread_id: &str,
        client_message_id: &str,
        user_text: &str,
        assistant_text: &str,
    ) -> anyhow::Result<AppendTurn> {
        self.append_turn_with_evidence(thread_id, client_message_id, user_text, assistant_text, &[])
    }

    #[cfg(test)]
    pub(crate) fn append_turn_with_evidence(
        &self,
        thread_id: &str,
        client_message_id: &str,
        user_text: &str,
        assistant_text: &str,
        evidence: &[EvidenceInput],
    ) -> anyhow::Result<AppendTurn> {
        self.append_turn_record(
            thread_id,
            client_message_id,
            user_text,
            assistant_text,
            "answered",
            evidence,
        )
    }

    pub(crate) fn append_terminal_turn(
        &self,
        thread_id: &str,
        client_message_id: &str,
        user_text: &str,
        assistant_text: &str,
        terminal_outcome: &str,
        evidence: &[EvidenceInput],
    ) -> anyhow::Result<AppendTurn> {
        anyhow::ensure!(
            matches!(
                terminal_outcome,
                "answered" | "aborted" | "step_capped" | "repeated" | "driver_error"
            ),
            "unsupported Workspace terminal outcome {terminal_outcome}"
        );
        self.append_turn_record(
            thread_id,
            client_message_id,
            user_text,
            assistant_text,
            terminal_outcome,
            evidence,
        )
    }

    fn append_turn_record(
        &self,
        thread_id: &str,
        client_message_id: &str,
        user_text: &str,
        assistant_text: &str,
        terminal_outcome: &str,
        evidence: &[EvidenceInput],
    ) -> anyhow::Result<AppendTurn> {
        let mut connection = self.connect()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if let Some(existing) = transaction
            .query_row(
                "SELECT id FROM workspace_turns
                 WHERE thread_id = ?1 AND client_message_id = ?2",
                params![thread_id, client_message_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
        {
            transaction.commit()?;
            return Ok(AppendTurn::Duplicate(existing));
        }
        let next_index: u32 = transaction.query_row(
            "SELECT COALESCE(MAX(turn_index), -1) + 1
             FROM workspace_turns WHERE thread_id = ?1",
            [thread_id],
            |row| row.get(0),
        )?;
        let now = now_epoch_seconds();
        transaction.execute(
            "INSERT INTO workspace_turns
             (thread_id, turn_index, client_message_id, user_text, assistant_text,
              terminal_outcome, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                thread_id,
                next_index,
                client_message_id,
                user_text,
                assistant_text,
                terminal_outcome,
                now
            ],
        )?;
        let turn_id = transaction.last_insert_rowid();
        transaction.execute(
            "INSERT INTO workspace_turns_fts
             (rowid, thread_id, user_text, assistant_text)
             VALUES (?1, ?2, ?3, ?4)",
            params![turn_id, thread_id, user_text, assistant_text],
        )?;
        let mut evidence_keys = HashSet::new();
        for entry in evidence.iter().take(MAX_EVIDENCE_PER_TURN) {
            let detail = bounded_text(&entry.detail, 4 * 1024);
            let observation = bounded_text(&entry.observation, 4 * 1024);
            let digest = format!("{:x}", Sha256::digest(observation.as_bytes()));
            let key = format!("{}\0{}\0{}", entry.tool, detail, digest);
            if !evidence_keys.insert(key) {
                continue;
            }
            transaction.execute(
                "INSERT INTO workspace_evidence
                 (turn_id, tool, detail, observation, observation_sha256)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![turn_id, entry.tool, detail, observation, digest],
            )?;
        }
        transaction.execute(
            "UPDATE workspace_threads SET updated_at = ?2 WHERE id = ?1",
            params![thread_id, now],
        )?;
        transaction.commit()?;
        Ok(AppendTurn::Inserted(turn_id))
    }

    pub(crate) fn recent_turns(
        &self,
        thread_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<StoredTurn>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let connection = self.connect()?;
        let mut statement = connection.prepare(
            "SELECT id, turn_index, user_text, assistant_text, terminal_outcome
             FROM workspace_turns WHERE thread_id = ?1
             ORDER BY turn_index DESC LIMIT ?2",
        )?;
        let mut turns = statement
            .query_map(params![thread_id, limit as i64], stored_turn_from_row)?
            .collect::<Result<Vec<_>, _>>()?;
        turns.reverse();
        Ok(turns)
    }

    pub(crate) fn turn_by_client_message(
        &self,
        thread_id: &str,
        client_message_id: &str,
    ) -> anyhow::Result<Option<StoredTurn>> {
        let connection = self.connect()?;
        Ok(connection
            .query_row(
                "SELECT id, turn_index, user_text, assistant_text, terminal_outcome
                 FROM workspace_turns
                 WHERE thread_id = ?1 AND client_message_id = ?2",
                params![thread_id, client_message_id],
                stored_turn_from_row,
            )
            .optional()?)
    }

    pub(crate) fn search_turns(
        &self,
        thread_id: &str,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<StoredTurn>> {
        let Some(fts_query) = fts_query(query) else {
            return Ok(Vec::new());
        };
        if limit == 0 {
            return Ok(Vec::new());
        }
        let connection = self.connect()?;
        let mut statement = connection.prepare(
            "SELECT t.id, t.turn_index, t.user_text, t.assistant_text, t.terminal_outcome
             FROM workspace_turns_fts AS f
             JOIN workspace_turns AS t ON t.id = f.rowid
                         WHERE workspace_turns_fts MATCH ?1 AND f.thread_id = ?2
                             AND t.terminal_outcome = 'answered'
             ORDER BY bm25(workspace_turns_fts), t.turn_index DESC
             LIMIT ?3",
        )?;
        let turns = statement
            .query_map(
                params![fts_query, thread_id, limit as i64],
                stored_turn_from_row,
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(turns)
    }

    pub(crate) fn context_for(
        &self,
        thread_id: &str,
        query: &str,
        max_chars: usize,
    ) -> anyhow::Result<MemoryContext> {
        let recent = self.recent_context_turns(thread_id, 3)?;
        let recent_ids = recent.iter().map(|turn| turn.id).collect::<HashSet<_>>();
        let relevant = self
            .search_turns(thread_id, query, 6)?
            .into_iter()
            .filter(|turn| !recent_ids.contains(&turn.id))
            .collect::<Vec<_>>();
        let mut context = bound_context(relevant, recent, max_chars);
        let used = context
            .relevant
            .iter()
            .chain(&context.recent)
            .map(|turn| {
                turn.user_text
                    .len()
                    .saturating_add(turn.assistant_text.len())
            })
            .sum::<usize>();
        let turn_ids = context
            .relevant
            .iter()
            .chain(&context.recent)
            .map(|turn| turn.id)
            .collect::<Vec<_>>();
        context.evidence = self.evidence_for_turns(&turn_ids, max_chars.saturating_sub(used))?;
        Ok(context)
    }

    fn recent_context_turns(
        &self,
        thread_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<StoredTurn>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let connection = self.connect()?;
        let mut statement = connection.prepare(
            "SELECT t.id, t.turn_index, t.user_text, t.assistant_text, t.terminal_outcome
             FROM workspace_turns t
             JOIN workspace_threads th ON th.id = t.thread_id
             WHERE t.thread_id = ?1
               AND t.turn_index > COALESCE(th.compacted_through_turn, -1)
                    AND t.terminal_outcome = 'answered'
             ORDER BY t.turn_index DESC LIMIT ?2",
        )?;
        let mut turns = statement
            .query_map(params![thread_id, limit as i64], stored_turn_from_row)?
            .collect::<Result<Vec<_>, _>>()?;
        turns.reverse();
        Ok(turns)
    }

    fn evidence_for_turns(
        &self,
        turn_ids: &[i64],
        max_chars: usize,
    ) -> anyhow::Result<Vec<StoredEvidence>> {
        let connection = self.connect()?;
        let mut statement = connection.prepare(
            "SELECT tool, detail, observation, observation_sha256
             FROM workspace_evidence WHERE turn_id = ?1 ORDER BY id",
        )?;
        let mut evidence = Vec::new();
        let mut used = 0usize;
        for turn_id in turn_ids {
            let entries = statement
                .query_map([turn_id], stored_evidence_from_row)?
                .collect::<Result<Vec<_>, _>>()?;
            for entry in entries {
                let actual = format!("{:x}", Sha256::digest(entry.observation.as_bytes()));
                anyhow::ensure!(
                    actual == entry.observation_sha256,
                    "Workspace evidence integrity check failed for turn {turn_id}"
                );
                let size = entry.detail.len().saturating_add(entry.observation.len());
                if used.saturating_add(size) > max_chars {
                    return Ok(evidence);
                }
                used += size;
                evidence.push(entry);
            }
        }
        Ok(evidence)
    }

    fn connect(&self) -> anyhow::Result<Connection> {
        let connection = Connection::open(&self.path)?;
        connection.busy_timeout(std::time::Duration::from_secs(5))?;
        connection.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;")?;
        Ok(connection)
    }
}

fn stored_turn_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredTurn> {
    Ok(StoredTurn {
        id: row.get(0)?,
        turn_index: row.get(1)?,
        user_text: row.get(2)?,
        assistant_text: row.get(3)?,
        terminal_outcome: row.get(4)?,
    })
}

fn stored_thread_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredThread> {
    Ok(StoredThread {
        id: row.get(0)?,
        title: row.get(8)?,
        canonical_root: row.get(1)?,
        model_id: row.get(2)?,
        model_sha256: row.get(3)?,
        compacted_through_turn: row.get(4)?,
        compaction_count: row.get(5)?,
        updated_at: row.get(6)?,
        turn_count: row.get(7)?,
    })
}

fn workspace_thread_title(initial_goal: &str) -> String {
    let line = initial_goal
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("Workspace conversation");
    let normalized = line.split_whitespace().collect::<Vec<_>>().join(" ");
    let count = normalized.chars().count();
    if count <= MAX_THREAD_TITLE_CHARS {
        return normalized;
    }
    let kept = MAX_THREAD_TITLE_CHARS.saturating_sub(3);
    format!("{}...", normalized.chars().take(kept).collect::<String>())
}

fn compaction_count(connection: &Connection, thread_id: &str) -> anyhow::Result<u32> {
    Ok(connection.query_row(
        "SELECT COUNT(*) FROM workspace_compactions WHERE thread_id = ?1",
        [thread_id],
        |row| row.get(0),
    )?)
}

fn stored_evidence_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredEvidence> {
    Ok(StoredEvidence {
        tool: row.get(0)?,
        detail: row.get(1)?,
        observation: row.get(2)?,
        observation_sha256: row.get(3)?,
    })
}

fn migrate(connection: &mut Connection) -> anyhow::Result<bool> {
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let version: i64 = transaction.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    anyhow::ensure!(
        version <= SCHEMA_VERSION,
        "Workspace memory schema {version} is newer than supported schema {SCHEMA_VERSION}"
    );
    if version == 0 {
        transaction.execute_batch(
            "CREATE TABLE workspace_threads (
               id TEXT PRIMARY KEY,
                             title TEXT NOT NULL,
               canonical_root TEXT NOT NULL,
               model_id TEXT NOT NULL,
                             model_sha256 TEXT NOT NULL,
                             compacted_through_turn INTEGER,
               created_at INTEGER NOT NULL,
               updated_at INTEGER NOT NULL
             );
             CREATE TABLE workspace_turns (
               id INTEGER PRIMARY KEY,
               thread_id TEXT NOT NULL REFERENCES workspace_threads(id) ON DELETE CASCADE,
               turn_index INTEGER NOT NULL,
               client_message_id TEXT NOT NULL,
               user_text TEXT NOT NULL,
               assistant_text TEXT NOT NULL,
                             terminal_outcome TEXT NOT NULL DEFAULT 'answered'
                                 CHECK(terminal_outcome IN ('answered', 'aborted', 'step_capped', 'repeated', 'driver_error')),
               created_at INTEGER NOT NULL,
               UNIQUE(thread_id, turn_index),
               UNIQUE(thread_id, client_message_id)
             );
             CREATE VIRTUAL TABLE workspace_turns_fts USING fts5(
               thread_id UNINDEXED,
               user_text,
               assistant_text
             );
                         CREATE TABLE workspace_evidence (
                             id INTEGER PRIMARY KEY,
                             turn_id INTEGER NOT NULL REFERENCES workspace_turns(id) ON DELETE CASCADE,
                             tool TEXT NOT NULL,
                             detail TEXT NOT NULL,
                             observation TEXT NOT NULL,
                             observation_sha256 TEXT NOT NULL
                         );
                         CREATE INDEX workspace_evidence_turn ON workspace_evidence(turn_id);
                         CREATE TABLE workspace_compactions (
                             id INTEGER PRIMARY KEY,
                             thread_id TEXT NOT NULL REFERENCES workspace_threads(id) ON DELETE CASCADE,
                             previous_boundary INTEGER,
                             new_boundary INTEGER NOT NULL,
                             created_at INTEGER NOT NULL
                         );
                         CREATE INDEX workspace_compactions_thread
                             ON workspace_compactions(thread_id, id);
                         PRAGMA user_version = 6;",
        )?;
    } else {
        if version < 2 {
            transaction.execute_batch(
                            "CREATE TABLE workspace_evidence (
                             id INTEGER PRIMARY KEY,
                             turn_id INTEGER NOT NULL REFERENCES workspace_turns(id) ON DELETE CASCADE,
                             tool TEXT NOT NULL,
                             detail TEXT NOT NULL,
                             observation TEXT NOT NULL,
                             observation_sha256 TEXT NOT NULL
                         );
                         CREATE INDEX workspace_evidence_turn ON workspace_evidence(turn_id);",
                        )?;
        }
        if version < 3 {
            transaction.execute_batch(
                "ALTER TABLE workspace_threads
                             ADD COLUMN model_sha256 TEXT NOT NULL DEFAULT '';",
            )?;
        }
        if version < 4 {
            transaction.execute_batch(
                                "ALTER TABLE workspace_threads
                                     ADD COLUMN compacted_through_turn INTEGER;
                                 CREATE TABLE workspace_compactions (
                                     id INTEGER PRIMARY KEY,
                                     thread_id TEXT NOT NULL REFERENCES workspace_threads(id) ON DELETE CASCADE,
                                     previous_boundary INTEGER,
                                     new_boundary INTEGER NOT NULL,
                                     created_at INTEGER NOT NULL
                                 );
                                 CREATE INDEX workspace_compactions_thread
                                     ON workspace_compactions(thread_id, id);",
                        )?;
        }
        if version < 5 {
            transaction.execute_batch(
                "ALTER TABLE workspace_turns
                 ADD COLUMN terminal_outcome TEXT NOT NULL DEFAULT 'answered'
                   CHECK(terminal_outcome IN ('answered', 'aborted', 'step_capped', 'repeated', 'driver_error'));",
            )?;
        }
        if version < 6 {
            transaction.execute_batch(
                "ALTER TABLE workspace_threads
                 ADD COLUMN title TEXT NOT NULL DEFAULT '';",
            )?;
            let existing = {
                let mut statement = transaction.prepare(
                    "SELECT th.id,
                            COALESCE((SELECT t.user_text FROM workspace_turns t
                                      WHERE t.thread_id = th.id
                                      ORDER BY t.turn_index ASC LIMIT 1), '')
                     FROM workspace_threads th WHERE th.title = ''",
                )?;
                let rows = statement
                    .query_map([], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                rows
            };
            for (thread_id, first_prompt) in existing {
                transaction.execute(
                    "UPDATE workspace_threads SET title = ?2 WHERE id = ?1",
                    params![thread_id, workspace_thread_title(&first_prompt)],
                )?;
            }
        }
        transaction.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    }
    transaction.commit()?;
    Ok(version < SCHEMA_VERSION)
}

fn verify_schema(connection: &Connection) -> anyhow::Result<()> {
    for name in [
        "workspace_threads",
        "workspace_turns",
        "workspace_turns_fts",
        "workspace_evidence",
        "workspace_evidence_turn",
        "workspace_compactions",
        "workspace_compactions_thread",
    ] {
        let exists: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name = ?1)",
            [name],
            |row| row.get(0),
        )?;
        anyhow::ensure!(exists, "Workspace memory schema is missing {name}");
    }
    for (table, required) in [
        (
            "workspace_threads",
            &[
                "id",
                "title",
                "canonical_root",
                "model_id",
                "model_sha256",
                "compacted_through_turn",
                "updated_at",
            ][..],
        ),
        (
            "workspace_turns",
            &[
                "id",
                "thread_id",
                "turn_index",
                "client_message_id",
                "user_text",
                "assistant_text",
                "terminal_outcome",
            ][..],
        ),
        (
            "workspace_evidence",
            &[
                "id",
                "turn_id",
                "tool",
                "detail",
                "observation",
                "observation_sha256",
            ][..],
        ),
        (
            "workspace_compactions",
            &[
                "id",
                "thread_id",
                "previous_boundary",
                "new_boundary",
                "created_at",
            ][..],
        ),
    ] {
        let columns = table_columns(connection, table)?;
        for column in required {
            anyhow::ensure!(
                columns.contains(*column),
                "Workspace memory schema is missing {table}.{column}"
            );
        }
    }
    let evidence_index_sql: String = connection.query_row(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='workspace_evidence_turn'",
        [],
        |row| row.get(0),
    )?;
    anyhow::ensure!(
        evidence_index_sql
            .replace(' ', "")
            .contains("workspace_evidence(turn_id)"),
        "Workspace evidence index does not target turn_id"
    );
    let compaction_index_sql: String = connection.query_row(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='workspace_compactions_thread'",
        [],
        |row| row.get(0),
    )?;
    anyhow::ensure!(
        compaction_index_sql
            .replace(' ', "")
            .contains("workspace_compactions(thread_id,id)"),
        "Workspace compaction index does not target thread_id,id"
    );
    Ok(())
}

fn verify_foreign_keys(connection: &Connection) -> anyhow::Result<()> {
    let foreign_key_error: Option<String> = connection
        .query_row("PRAGMA foreign_key_check", [], |row| row.get(0))
        .optional()?;
    anyhow::ensure!(
        foreign_key_error.is_none(),
        "Workspace memory failed foreign-key validation"
    );
    Ok(())
}

fn table_columns(connection: &Connection, table: &str) -> anyhow::Result<HashSet<String>> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<HashSet<_>, _>>()?;
    Ok(columns)
}

fn fts_query(text: &str) -> Option<String> {
    let terms = text
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .filter(|term| term.len() >= 2)
        .take(16)
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect::<Vec<_>>();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}

fn bound_context(
    relevant: Vec<StoredTurn>,
    recent: Vec<StoredTurn>,
    max_chars: usize,
) -> MemoryContext {
    let had_recent = !recent.is_empty();
    let mut used = 0usize;
    let mut kept_recent = Vec::new();
    for turn in recent.into_iter().rev() {
        let size = turn
            .user_text
            .len()
            .saturating_add(turn.assistant_text.len());
        if used.saturating_add(size) <= max_chars {
            used += size;
            kept_recent.push(turn);
        } else {
            break;
        }
    }
    kept_recent.reverse();
    let mut kept_relevant = Vec::new();
    if !had_recent || !kept_recent.is_empty() {
        for turn in relevant {
            let size = turn
                .user_text
                .len()
                .saturating_add(turn.assistant_text.len());
            if used.saturating_add(size) <= max_chars {
                used += size;
                kept_relevant.push(turn);
            }
        }
    }
    kept_relevant.sort_by_key(|turn| turn.turn_index);
    MemoryContext {
        relevant: kept_relevant,
        recent: kept_recent,
        evidence: Vec::new(),
    }
}

fn bounded_text(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

fn now_epoch_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .min(i64::MAX as u64) as i64
}

pub(crate) fn default_store_path() -> PathBuf {
    if let Some(path) = std::env::var_os("CAMELID_WORKSPACE_MEMORY_DB") {
        return PathBuf::from(path);
    }
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
            })
    };
    base.unwrap_or_else(std::env::temp_dir)
        .join("camelid")
        .join("workspace-memory.sqlite3")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn store(dir: &Path) -> WorkspaceMemoryStore {
        WorkspaceMemoryStore::open(dir.join("memory.sqlite3")).unwrap()
    }

    #[test]
    fn thread_title_uses_first_nonempty_prompt_line_and_is_bounded() {
        assert_eq!(
            workspace_thread_title("\n  Review   authentication flow  \nIgnore this line"),
            "Review authentication flow"
        );
        assert_eq!(workspace_thread_title("\n \t"), "Workspace conversation");
        let long = "é".repeat(MAX_THREAD_TITLE_CHARS + 10);
        let title = workspace_thread_title(&long);
        assert_eq!(title.chars().count(), MAX_THREAD_TITLE_CHARS);
        assert!(title.ends_with("..."));
    }

    #[test]
    fn turns_are_transactional_idempotent_and_thread_scoped() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        store.create_thread("a", "C:/repo", "qwen").unwrap();
        store.create_thread("b", "C:/repo", "qwen").unwrap();
        let inserted = store
            .append_turn("a", "message-1", "Where is login?", "In src/auth.rs")
            .unwrap();
        let duplicate = store
            .append_turn("a", "message-1", "changed", "changed")
            .unwrap();
        assert_eq!(
            duplicate,
            AppendTurn::Duplicate(match inserted {
                AppendTurn::Inserted(id) => id,
                AppendTurn::Duplicate(_) => unreachable!(),
            })
        );
        assert_eq!(store.recent_turns("a", 10).unwrap().len(), 1);
        assert!(store.recent_turns("b", 10).unwrap().is_empty());
    }

    #[test]
    fn terminal_attempts_are_durable_idempotent_and_not_model_context() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        store.create_thread("a", "C:/repo", "qwen").unwrap();
        let inserted = store
            .append_terminal_turn("a", "message-1", "Summarize all files", "", "aborted", &[])
            .unwrap();
        let duplicate = store
            .append_terminal_turn("a", "message-1", "changed", "changed", "driver_error", &[])
            .unwrap();

        assert_eq!(
            duplicate,
            AppendTurn::Duplicate(match inserted {
                AppendTurn::Inserted(id) => id,
                AppendTurn::Duplicate(_) => unreachable!(),
            })
        );
        let turns = store.recent_turns("a", 10).unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].user_text, "Summarize all files");
        assert_eq!(turns[0].assistant_text, "");
        assert_eq!(turns[0].terminal_outcome, "aborted");
        assert_eq!(
            store
                .turn_by_client_message("a", "message-1")
                .unwrap()
                .unwrap()
                .terminal_outcome,
            "aborted"
        );
        assert!(store
            .context_for("a", "Summarize files", 10_000)
            .unwrap()
            .recent
            .is_empty());
        assert!(store
            .search_turns("a", "Summarize files", 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn lexical_search_returns_relevant_older_turns_only_from_the_thread() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        store.create_thread("a", "C:/repo", "qwen").unwrap();
        store.create_thread("b", "C:/repo", "qwen").unwrap();
        store
            .append_turn(
                "a",
                "1",
                "Inspect authentication",
                "Login is in src/auth.rs",
            )
            .unwrap();
        store
            .append_turn("a", "2", "Check CSS spacing", "Spacing is in workspace.css")
            .unwrap();
        store
            .append_turn("b", "3", "Inspect authentication", "Other thread")
            .unwrap();
        let matches = store.search_turns("a", "login authentication", 5).unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].assistant_text, "Login is in src/auth.rs");
    }

    #[test]
    fn unsupported_newer_schema_fails_without_overwriting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.sqlite3");
        let connection = Connection::open(&path).unwrap();
        connection.pragma_update(None, "user_version", 99).unwrap();
        drop(connection);
        let error = WorkspaceMemoryStore::open(&path).err().unwrap().to_string();
        assert!(error.contains("newer than supported"));
        let version: i64 = Connection::open(path)
            .unwrap()
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 99);
    }

    #[test]
    fn context_selection_prefers_complete_recent_turns_and_deduplicates_search() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        store.create_thread("a", "C:/repo", "qwen").unwrap();
        store
            .append_turn("a", "1", "old authentication", "src/auth.rs")
            .unwrap();
        store
            .append_turn("a", "2", "new authentication", "still auth")
            .unwrap();
        let context = store.context_for("a", "authentication", 10_000).unwrap();
        assert_eq!(context.recent.len(), 2);
        assert!(context.relevant.is_empty());

        let bounded = store.context_for("a", "authentication", 20).unwrap();
        assert!(bounded.relevant.is_empty());
        assert!(bounded.recent.len() <= 1);
    }

    #[test]
    fn oversized_newest_turn_does_not_fall_back_to_stale_context() {
        let turn = |id, turn_index, user_text: &str| StoredTurn {
            id,
            turn_index,
            user_text: user_text.to_string(),
            assistant_text: String::new(),
            terminal_outcome: "answered".to_string(),
        };
        let context = bound_context(
            vec![turn(3, 0, "old retrieved")],
            vec![turn(1, 0, "old"), turn(2, 1, "newest is too large")],
            8,
        );

        assert!(context.recent.is_empty());
        assert!(context.relevant.is_empty());
    }

    #[test]
    fn saved_thread_is_listed_and_retrievable_after_store_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.sqlite3");
        {
            let store = WorkspaceMemoryStore::open(&path).unwrap();
            store
                .create_thread_for_model(
                    "a",
                    "C:/repo",
                    "qwen",
                    "test-model-sha256",
                    "\n  Review   authentication flow  \nIgnore this line",
                )
                .unwrap();
            store
                .append_turn("a", "1", "first question", "first answer")
                .unwrap();
        }
        let reopened = WorkspaceMemoryStore::open(&path).unwrap();
        let threads = reopened.threads_for_root("C:/repo", 10).unwrap();
        assert_eq!(threads.len(), 1);
        assert_eq!(threads[0].id, "a");
        assert_eq!(threads[0].title, "Review authentication flow");
        assert_eq!(threads[0].model_id, "qwen");
        assert_eq!(threads[0].model_sha256, "test-model-sha256");
        assert_eq!(threads[0].turn_count, 1);
        assert_eq!(
            reopened.recent_turns("a", 10).unwrap()[0].assistant_text,
            "first answer"
        );
    }

    #[test]
    fn deleting_a_thread_removes_transcript_and_fts_rows_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        store.create_thread("a", "C:/repo", "qwen").unwrap();
        store
            .append_turn_with_evidence(
                "a",
                "1",
                "secret question",
                "secret answer",
                &[
                    EvidenceInput {
                        tool: "read_file".into(),
                        detail: "read_file(src/auth.rs, start_line=10, max_lines=20)".into(),
                        observation: "10: fn login()".into(),
                    },
                    EvidenceInput {
                        tool: "read_file".into(),
                        detail: "read_file(src/auth.rs, start_line=10, max_lines=20)".into(),
                        observation: "10: fn login()".into(),
                    },
                ],
            )
            .unwrap();
        let context = store.context_for("a", "secret", 10_000).unwrap();
        assert_eq!(context.evidence.len(), 1);
        assert_eq!(context.evidence[0].observation_sha256.len(), 64);
        assert!(store.delete_thread("a").unwrap());
        assert!(store.thread("a").unwrap().is_none());
        assert!(store.recent_turns("a", 10).unwrap().is_empty());
        assert!(store.search_turns("a", "secret", 10).unwrap().is_empty());
        assert!(!store.delete_thread("a").unwrap());
    }

    #[test]
    fn evidence_tampering_fails_closed_on_retrieval() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        store.create_thread("a", "C:/repo", "qwen").unwrap();
        store
            .append_turn_with_evidence(
                "a",
                "1",
                "question",
                "answer",
                &[EvidenceInput {
                    tool: "read_file".into(),
                    detail: "read_file(src/auth.rs)".into(),
                    observation: "original".into(),
                }],
            )
            .unwrap();
        Connection::open(&store.path)
            .unwrap()
            .execute("UPDATE workspace_evidence SET observation = 'tampered'", [])
            .unwrap();
        let error = store.context_for("a", "question", 10_000).unwrap_err();
        assert!(error.to_string().contains("integrity check failed"));
    }

    #[test]
    fn schema_v1_migrates_to_current_evidence_schema() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.sqlite3");
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE workspace_threads (
                   id TEXT PRIMARY KEY, canonical_root TEXT NOT NULL, model_id TEXT NOT NULL,
                   created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
                 );
                 CREATE TABLE workspace_turns (
                   id INTEGER PRIMARY KEY,
                   thread_id TEXT NOT NULL REFERENCES workspace_threads(id) ON DELETE CASCADE,
                   turn_index INTEGER NOT NULL, client_message_id TEXT NOT NULL,
                   user_text TEXT NOT NULL, assistant_text TEXT NOT NULL, created_at INTEGER NOT NULL,
                   UNIQUE(thread_id, turn_index), UNIQUE(thread_id, client_message_id)
                 );
                 CREATE VIRTUAL TABLE workspace_turns_fts USING fts5(
                   thread_id UNINDEXED, user_text, assistant_text
                 );
                                 INSERT INTO workspace_threads
                                     (id, canonical_root, model_id, created_at, updated_at)
                                     VALUES ('legacy', 'C:/repo', 'qwen', 1, 1);
                                 INSERT INTO workspace_turns
                                     (thread_id, turn_index, client_message_id, user_text, assistant_text, created_at)
                                     VALUES ('legacy', 0, 'message-1', 'Migrated title', 'answer', 1);
                 PRAGMA user_version = 1;",
            )
            .unwrap();
        drop(connection);
        WorkspaceMemoryStore::open(&path).unwrap();
        let migrated = Connection::open(path).unwrap();
        let version: i64 = migrated
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 6);
        let title: String = migrated
            .query_row(
                "SELECT title FROM workspace_threads WHERE id = 'legacy'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(title, "Migrated title");
        let evidence_exists: bool = migrated
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='workspace_evidence')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(evidence_exists);
    }

    #[test]
    fn concurrent_first_open_serializes_schema_creation() {
        let dir = tempfile::tempdir().unwrap();
        let path = std::sync::Arc::new(dir.path().join("memory.sqlite3"));
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let joins = (0..2)
            .map(|_| {
                let path = std::sync::Arc::clone(&path);
                let barrier = std::sync::Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    WorkspaceMemoryStore::open(path.as_ref()).map(|_| ())
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        for join in joins {
            join.join().unwrap().unwrap();
        }
    }

    #[test]
    fn claimed_current_schema_with_missing_objects_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("memory.sqlite3");
        let connection = Connection::open(&path).unwrap();
        connection.pragma_update(None, "user_version", 6).unwrap();
        drop(connection);
        let error = WorkspaceMemoryStore::open(path).err().unwrap();
        assert!(error.to_string().contains("schema is missing"));
    }

    #[test]
    fn compaction_is_reversible_and_preserves_fts_retrieval() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        store.create_thread("a", "C:/repo", "qwen").unwrap();
        store
            .append_turn("a", "1", "authentication location", "src/auth.rs")
            .unwrap();
        store
            .append_turn("a", "2", "spacing question", "workspace.css")
            .unwrap();
        let before = store.context_for("a", "unrelated", 10_000).unwrap();
        assert_eq!(before.recent.len(), 2);

        let compacted = store.compact_thread("a").unwrap();
        assert_eq!(compacted.compacted_through_turn, Some(1));
        assert_eq!(compacted.archived_turns, 2);
        assert_eq!(compacted.compaction_count, 1);
        let unrelated = store.context_for("a", "unrelated", 10_000).unwrap();
        assert!(unrelated.recent.is_empty());
        assert!(unrelated.relevant.is_empty());
        let relevant = store.context_for("a", "authentication", 10_000).unwrap();
        assert!(relevant.recent.is_empty());
        assert_eq!(relevant.relevant.len(), 1);
        assert_eq!(relevant.relevant[0].assistant_text, "src/auth.rs");
        assert_eq!(store.recent_turns("a", 10).unwrap().len(), 2);

        let undone = store.undo_compaction("a").unwrap();
        assert_eq!(undone.compacted_through_turn, None);
        assert_eq!(undone.archived_turns, 2);
        assert_eq!(undone.compaction_count, 0);
        assert_eq!(
            store
                .context_for("a", "unrelated", 10_000)
                .unwrap()
                .recent
                .len(),
            2
        );
    }
}
