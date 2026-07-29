//! Read-only semantic retrieval for the Workspace agent.
//!
//! The index is built lazily per Workspace session, remains in memory for later
//! turns, and never writes into the selected project. File contents are still
//! framed as untrusted memory before they reach the model.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::embedding::{cosine_similarity, NomicBertRuntime};

const MAX_INDEX_FILES: usize = 192;
const MAX_DISCOVERED_FILES: usize = 4_096;
const MAX_INDEX_CHUNKS: usize = 128;
const MAX_CHUNKS_PER_FILE: usize = 4;
const MAX_FILE_BYTES: u64 = 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 8 * 1024 * 1024;
const TARGET_CHUNK_BYTES: usize = 1_600;
const MAX_RENDERED_SNIPPET_CHARS: usize = 420;
const SKIP_DIRECTORIES: &[&str] = &[
    ".git",
    ".camelid",
    "target",
    "node_modules",
    "vendor",
    ".venv",
    "venv",
    "dist",
    "build",
];

#[derive(Debug, Clone)]
struct SourceChunk {
    path: String,
    start_line: usize,
    text: String,
}

#[derive(Debug)]
struct SemanticIndex {
    chunks: Vec<SourceChunk>,
    vectors: Vec<Vec<f32>>,
}

pub(crate) struct WorkspaceSemanticRetriever {
    root: PathBuf,
    model_id: String,
    runtime: Arc<NomicBertRuntime>,
    index: Mutex<Option<Arc<SemanticIndex>>>,
}

impl std::fmt::Debug for WorkspaceSemanticRetriever {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WorkspaceSemanticRetriever")
            .field("root", &self.root)
            .field("model_id", &self.model_id)
            .field(
                "indexed",
                &self
                    .index
                    .lock()
                    .map(|index| index.is_some())
                    .unwrap_or(false),
            )
            .finish_non_exhaustive()
    }
}

impl WorkspaceSemanticRetriever {
    pub(crate) fn new(root: PathBuf, model_id: String, runtime: Arc<NomicBertRuntime>) -> Self {
        Self {
            root,
            model_id,
            runtime,
            index: Mutex::new(None),
        }
    }

    pub(crate) fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Return a bounded, already-framed memory block for one Workspace goal.
    pub(crate) fn retrieve_context(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Option<String>, String> {
        if query.trim().is_empty() || limit == 0 {
            return Ok(None);
        }
        let index = self.index()?;
        if index.chunks.is_empty() {
            return Ok(None);
        }
        let query = with_prefix(query, "search_query: ");
        let query_vector = self
            .runtime
            .embed(&query, None)
            .map_err(|error| format!("semantic query embedding failed: {error}"))?;
        let mut ranked = index
            .vectors
            .iter()
            .enumerate()
            .map(|(index, vector)| {
                cosine_similarity(&query_vector, vector)
                    .map(|score| (index, score))
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(limit.min(index.chunks.len()));

        let mut rendered = format!(
            "Semantically relevant workspace excerpts (embedding model {}; untrusted file data):\n",
            self.model_id
        );
        for (chunk_index, score) in ranked {
            let chunk = &index.chunks[chunk_index];
            let snippet = bounded_snippet(&chunk.text);
            rendered.push_str(&format!(
                "- {}:{} (cosine {:.4})\n  {}\n",
                chunk.path,
                chunk.start_line,
                score,
                snippet.replace('\n', "\n  ")
            ));
        }
        Ok(Some(rendered))
    }

    fn index(&self) -> Result<Arc<SemanticIndex>, String> {
        let mut cached = self
            .index
            .lock()
            .map_err(|_| "semantic index lock is unavailable".to_string())?;
        if let Some(index) = cached.as_ref() {
            return Ok(Arc::clone(index));
        }
        let chunks = collect_chunks(&self.root)?;
        let documents = chunks
            .iter()
            .map(|chunk| {
                with_prefix(
                    &format!("{}\n{}", chunk.path, chunk.text),
                    "search_document: ",
                )
            })
            .collect::<Vec<_>>();
        let vectors = if documents.is_empty() {
            Vec::new()
        } else {
            self.runtime
                .embed_batch(&documents, None)
                .map_err(|error| format!("semantic workspace indexing failed: {error}"))?
        };
        let index = Arc::new(SemanticIndex { chunks, vectors });
        *cached = Some(Arc::clone(&index));
        Ok(index)
    }
}

fn with_prefix(text: &str, prefix: &str) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with("search_query:") || trimmed.starts_with("search_document:") {
        trimmed.to_string()
    } else {
        format!("{prefix}{trimmed}")
    }
}

fn collect_chunks(root: &Path) -> Result<Vec<SourceChunk>, String> {
    let canonical_root = std::fs::canonicalize(root)
        .map_err(|error| format!("semantic workspace root is unavailable: {error}"))?;
    let mut pending = vec![canonical_root.clone()];
    let mut visited_directories = HashSet::new();
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        // Canonical containment also catches Windows junctions/reparse points,
        // which are not guaranteed to present as ordinary symlinks. The visited
        // set makes directory-link cycles harmless.
        let directory = match std::fs::canonicalize(&directory) {
            Ok(directory)
                if directory.starts_with(&canonical_root)
                    && visited_directories.insert(directory.clone()) =>
            {
                directory
            }
            _ => continue,
        };
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        let mut entries = entries
            .filter_map(Result::ok)
            .collect::<Vec<std::fs::DirEntry>>();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries.into_iter().rev() {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                let name = entry.file_name();
                if !SKIP_DIRECTORIES
                    .iter()
                    .any(|skip| name.to_string_lossy().eq_ignore_ascii_case(skip))
                {
                    let canonical = match std::fs::canonicalize(path) {
                        Ok(canonical) if canonical.starts_with(&canonical_root) => canonical,
                        _ => continue,
                    };
                    pending.push(canonical);
                }
            } else if file_type.is_file() && is_indexable_file(&path) {
                files.push(path);
                if files.len() >= MAX_DISCOVERED_FILES {
                    pending.clear();
                    break;
                }
            }
        }
    }
    files.sort_by_key(|path| index_priority(&canonical_root, path));
    files.truncate(MAX_INDEX_FILES);

    let mut chunks_by_file = Vec::new();
    let mut total_bytes = 0_u64;
    for path in files {
        let length = match std::fs::metadata(&path) {
            Ok(metadata) => metadata.len(),
            Err(_) => continue,
        };
        if length == 0
            || length > MAX_FILE_BYTES
            || total_bytes.saturating_add(length) > MAX_TOTAL_BYTES
        {
            continue;
        }
        let raw = match std::fs::read(&path) {
            Ok(raw) => raw,
            Err(_) => continue,
        };
        let text = match std::str::from_utf8(&raw) {
            Ok(text) => text,
            Err(_) => continue,
        };
        total_bytes += length;
        let relative = path
            .strip_prefix(&canonical_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let file_chunks = chunk_file(&relative, text);
        if !file_chunks.is_empty() {
            chunks_by_file.push(file_chunks);
        }
    }
    Ok(select_chunks_breadth_first(&chunks_by_file))
}

/// Prefer the project's primary source tree, followed by nested application
/// source trees, tests/examples, and finally prose/configuration. Candidate
/// discovery remains bounded, but a large `docs/` or `qa/` tree can no longer
/// crowd the implementation itself out of the semantic index.
fn index_priority(root: &Path, path: &Path) -> (u8, String) {
    let relative = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase();
    let tier = if relative.starts_with("src/") {
        0
    } else if relative.contains("/src/")
        || relative.starts_with("app/")
        || relative.starts_with("lib/")
        || relative.starts_with("packages/")
    {
        1
    } else if relative.starts_with("tests/")
        || relative.starts_with("test/")
        || relative.starts_with("examples/")
    {
        2
    } else if !relative.contains('/') {
        3
    } else {
        4
    };
    (tier, relative)
}

/// Take the first chunk from as many files as possible before taking a second
/// chunk from any file. This provides useful repository breadth under the hard
/// chunk cap and avoids four early files consuming the whole index.
fn select_chunks_breadth_first(chunks_by_file: &[Vec<SourceChunk>]) -> Vec<SourceChunk> {
    let mut selected = Vec::with_capacity(MAX_INDEX_CHUNKS);
    for chunk_index in 0..MAX_CHUNKS_PER_FILE {
        for file_chunks in chunks_by_file {
            if let Some(chunk) = file_chunks.get(chunk_index) {
                selected.push(chunk.clone());
                if selected.len() == MAX_INDEX_CHUNKS {
                    return selected;
                }
            }
        }
    }
    selected
}

fn is_indexable_file(path: &Path) -> bool {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if matches!(
        file_name.to_ascii_lowercase().as_str(),
        "dockerfile" | "makefile" | "cmakelists.txt" | "cargo.toml"
    ) {
        return true;
    }
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "rs" | "toml"
            | "md"
            | "txt"
            | "json"
            | "yaml"
            | "yml"
            | "js"
            | "mjs"
            | "cjs"
            | "ts"
            | "tsx"
            | "jsx"
            | "py"
            | "go"
            | "java"
            | "kt"
            | "c"
            | "cc"
            | "cpp"
            | "h"
            | "hpp"
            | "cs"
            | "swift"
            | "sh"
            | "ps1"
            | "sql"
            | "html"
            | "css"
            | "scss"
            | "xml"
    )
}

fn chunk_file(path: &str, text: &str) -> Vec<SourceChunk> {
    let mut chunks = Vec::new();
    let mut cursor = 0;
    let mut start_line = 1;
    while cursor < text.len() && chunks.len() < MAX_CHUNKS_PER_FILE {
        let remaining = &text[cursor..];
        let mut end = remaining.len().min(TARGET_CHUNK_BYTES);
        while end > 0 && !remaining.is_char_boundary(end) {
            end -= 1;
        }
        if end == 0 {
            end = remaining
                .char_indices()
                .nth(1)
                .map(|(index, _)| index)
                .unwrap_or(remaining.len());
        }
        if end < remaining.len() {
            if let Some(newline) = remaining[..end].rfind('\n') {
                if newline > 0 {
                    end = newline + 1;
                }
            }
        }
        let body = &remaining[..end];
        let next_line = start_line + body.bytes().filter(|byte| *byte == b'\n').count();
        cursor += end;
        if body.trim().is_empty() {
            start_line = next_line;
            continue;
        }
        chunks.push(SourceChunk {
            path: path.to_string(),
            start_line,
            text: body.trim().to_string(),
        });
        start_line = next_line;
    }
    chunks
}

fn bounded_snippet(text: &str) -> String {
    let mut snippet = text
        .chars()
        .filter(|character| *character != '\0')
        .take(MAX_RENDERED_SNIPPET_CHARS)
        .collect::<String>();
    if text.chars().count() > MAX_RENDERED_SNIPPET_CHARS {
        snippet.push_str(" …");
    }
    snippet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_are_bounded_and_keep_source_lines() {
        let text = (1..=200)
            .map(|line| format!("line {line}: camelid semantic content"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_file("src/lib.rs", &text);
        assert!(!chunks.is_empty());
        assert!(chunks.len() <= MAX_CHUNKS_PER_FILE);
        assert_eq!(chunks[0].path, "src/lib.rs");
        assert_eq!(chunks[0].start_line, 1);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.text.len() <= TARGET_CHUNK_BYTES));
        assert!(chunks
            .windows(2)
            .all(|pair| pair[0].start_line < pair[1].start_line));
    }

    #[test]
    fn minified_and_multibyte_lines_are_split_on_utf8_boundaries() {
        let text = "🦙".repeat(TARGET_CHUNK_BYTES);
        let chunks = chunk_file("frontend/bundle.js", &text);
        assert_eq!(chunks.len(), MAX_CHUNKS_PER_FILE);
        assert!(chunks
            .iter()
            .all(|chunk| chunk.text.len() <= TARGET_CHUNK_BYTES));
        assert!(chunks
            .iter()
            .all(|chunk| chunk.text.is_char_boundary(chunk.text.len())));
    }

    #[test]
    fn indexable_file_policy_skips_build_outputs_and_binary_extensions() {
        assert!(is_indexable_file(Path::new("src/main.rs")));
        assert!(is_indexable_file(Path::new("README.md")));
        assert!(is_indexable_file(Path::new("Dockerfile")));
        assert!(!is_indexable_file(Path::new("model.gguf")));
        assert!(!is_indexable_file(Path::new("image.png")));
    }

    #[test]
    fn source_priority_and_breadth_first_selection_cover_many_files() {
        let root = Path::new("workspace");
        assert!(
            index_priority(root, Path::new("workspace/src/lib.rs"))
                < index_priority(root, Path::new("workspace/docs/design.md"))
        );
        let groups = (0..40)
            .map(|file| {
                (0..MAX_CHUNKS_PER_FILE)
                    .map(|chunk| SourceChunk {
                        path: format!("src/file-{file}.rs"),
                        start_line: chunk + 1,
                        text: format!("file {file}, chunk {chunk}"),
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let selected = select_chunks_breadth_first(&groups);
        assert_eq!(selected.len(), MAX_INDEX_CHUNKS);
        assert_eq!(
            selected[..40]
                .iter()
                .map(|chunk| chunk.path.as_str())
                .collect::<HashSet<_>>()
                .len(),
            40,
            "the first selection round must cover every candidate file"
        );
        assert_eq!(selected[40].start_line, 2);
    }
}
