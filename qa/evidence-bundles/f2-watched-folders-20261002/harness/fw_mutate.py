#!/usr/bin/env python3
"""Apply one named mutation to src/api/document_folders.rs (run from the worktree root).

usage: fw_mutate.py <name>
"""
import sys

PATH = "src/api/document_folders.rs"
MUTATIONS = {
    # Folder routes answer any caller.
    "open_routes": ("serve_addr.ip().is_loopback() && super::workspace::local_management_request_allowed(headers)", "{ let _ = (serve_addr, headers); true }"),
    # Links are followed: the kind is read through the link.
    "follow_links": ("let Ok(kind) = entry.file_type() else {", "let Ok(kind) = std::fs::metadata(entry.path()).map(|m| m.file_type()) else {"),
    # Hidden entries are read.
    "read_hidden": ("if name.starts_with('.') {", "if name.starts_with(\"\\u{0}\") {"),
    # Documents under a folder that could not be read are removed as if gone.
    "drop_unreadable": ("if kept(&rel_path) {", "if kept(&rel_path) && false {"),
    # Unchanged bytes are re-indexed whenever the file's time moves.
    "rehash_ignored": ("return Read::Same(sha256);", "let _ = Read::Same(sha256.clone());"),
    # Overlapping folders are accepted.
    "allow_overlap": ("canonical.starts_with(other) || other.starts_with(canonical)", "{ let _ = other; false }"),
}

name = sys.argv[1]
old, new = MUTATIONS[name]
src = open(PATH, encoding="utf-8").read()
assert src.count(old) == 1, f"{name}: anchor found {src.count(old)} times"
open(PATH, "w", encoding="utf-8").write(src.replace(old, new))
print(f"applied {name}")
