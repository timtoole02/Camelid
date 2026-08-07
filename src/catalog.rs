//! `camelid pull` — fetch a supported model so the user never has to hunt for a
//! compatible GGUF by hand.
//!
//! Camelid only serves specific Q8_0 (and one Q4_0 QAT) rows, and most GGUFs on
//! the web are other quantizations that fail closed. This command downloads one
//! of the known-good rows from the same curated catalog the web UI uses, into
//! `./models`, and prints the exact `camelid serve` command to run next.

use std::path::{Path, PathBuf};

use crate::api::{curated_catalog, CatalogItem};

/// Entry point for the `Pull` subcommand. With no `query`, prints the catalog;
/// otherwise resolves `query` to exactly one row and downloads it into
/// `models_dir`.
pub fn run_pull(query: Option<&str>, models_dir: &Path) -> anyhow::Result<()> {
    run_pull_opts(query, models_dir, false)
}

/// [`run_pull`] with CLI-only options: `dry_run` resolves and prints what would
/// be downloaded without moving bytes.
pub fn run_pull_opts(query: Option<&str>, models_dir: &Path, dry_run: bool) -> anyhow::Result<()> {
    // An `org/repo[:quant]` spec pulls straight from Hugging Face (experimental
    // lane, unverified); curated catalog ids never contain a '/'.
    if let Some(query) = query {
        if crate::hf_pull::is_hf_spec(query) {
            return crate::hf_pull::run_hf_pull(query, models_dir, dry_run).map(|_| ());
        }
    }

    let entries = curated_catalog();

    let Some(query) = query else {
        print_catalog(&entries);
        eprintln!("\nDownload one with:  camelid pull <id>   (e.g. camelid pull llama32_3b)");
        return Ok(());
    };

    let item = resolve(&entries, query)?;
    if dry_run {
        eprintln!("Would download:");
        eprintln!("  model: {} ({})", item.name, item.quant);
        eprintln!("  repo:  {}", item.repo_id);
        eprintln!(
            "  file:  {} ({:.1} GB)",
            item.filename,
            item.size_bytes as f64 / 1e9
        );
        eprintln!("  dest:  {}", models_dir.join(item.filename).display());
        return Ok(());
    }
    let dest = download(&item, models_dir)?;

    eprintln!("\n✓ {} is ready at {}", item.name, dest.display());
    eprintln!(
        "\nStart chatting:\n  camelid serve --model {}",
        dest.display()
    );
    Ok(())
}

/// Match `query` against the catalog by id or name, ignoring case and the
/// `-`/`_`/`.`/space separators people mix up. Errors helpfully on no match or
/// an ambiguous match rather than guessing.
fn resolve(entries: &[CatalogItem], query: &str) -> anyhow::Result<CatalogItem> {
    let needle = normalize(query);

    let matches: Vec<&CatalogItem> = entries
        .iter()
        .filter(|item| {
            normalize(item.catalog_id).contains(&needle) || normalize(item.name).contains(&needle)
        })
        .collect();

    // A fully-typed id wins outright. Substring matching alone makes any id that
    // is a prefix of a longer one unresolvable — the user who spelled the whole
    // thing out has already been as specific as the CLI allows, and answering
    // "be more specific" to an exact id would be a dead end.
    if let Some(exact) = matches
        .iter()
        .find(|item| normalize(item.catalog_id) == needle || normalize(item.name) == needle)
    {
        return Ok((*exact).clone());
    }

    match matches.as_slice() {
        [] => {
            print_catalog(entries);
            anyhow::bail!(
                "no supported model matches \"{query}\" — pick an id from the list above"
            );
        }
        [only] => Ok((*only).clone()),
        many => {
            let ids: Vec<&str> = many.iter().map(|item| item.catalog_id).collect();
            anyhow::bail!(
                "\"{query}\" matches several models ({}); be more specific",
                ids.join(", ")
            );
        }
    }
}

/// Ask the Hugging Face Hub for the current byte size of `item`'s GGUF.
///
/// The catalog ships a `size_bytes` constant, but uploaders occasionally
/// re-publish a row (a re-quant, a metadata fix) and the byte count shifts.
/// Gating "is this download complete?" on a baked-in constant then breaks: a
/// fully-downloaded file stops matching, so `pull` tries to resume a file that
/// is already whole. Querying the Hub's file tree keeps the check honest.
///
/// Returns `None` when offline or the response can't be parsed; callers then
/// fall back to the catalog constant.
fn remote_size(item: &CatalogItem) -> Option<u64> {
    let url = format!(
        "https://huggingface.co/api/models/{}/tree/main?recursive=1",
        item.repo_id
    );
    let output = std::process::Command::new("curl")
        .args(["-fsSL", &url])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let tree: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    for entry in tree.as_array()? {
        if entry.get("path").and_then(|p| p.as_str()) == Some(item.filename) {
            // LFS/xet-backed files report the real content size under `lfs.size`;
            // the top-level `size` for those is just the pointer's byte count.
            return entry
                .get("lfs")
                .and_then(|lfs| lfs.get("size"))
                .and_then(|s| s.as_u64())
                .or_else(|| entry.get("size").and_then(|s| s.as_u64()));
        }
    }
    None
}

/// Download `item` into `models_dir` via `curl` (resumable, streaming progress
/// to the terminal). Skips a complete copy, resumes a partial one, and re-fetches
/// a stale/oversized one — judged against the Hub's *current* file size, not a
/// baked-in constant, so a re-published row can't trick `pull` into resuming a
/// file that is already whole.
fn download(item: &CatalogItem, models_dir: &Path) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(models_dir)?;
    let dest = models_dir.join(item.filename);

    // Authoritative size from the Hub; fall back to the catalog constant offline.
    let expected = remote_size(item);
    let target = expected.unwrap_or(item.size_bytes);

    if let Ok(meta) = std::fs::metadata(&dest) {
        let have = meta.len();
        if have == target {
            eprintln!(
                "{} already downloaded at {} ({:.1} GB, size-verified)",
                item.name,
                dest.display(),
                target as f64 / 1e9
            );
            return Ok(dest);
        }
        if expected.is_some() && have > target {
            // Larger than the Hub's current file: a stale or corrupt copy. A
            // byte-range resume can't repair that, so start clean.
            eprintln!(
                "Local {} is {have} bytes but the Hub file is {target} — re-downloading fresh",
                item.filename
            );
            let _ = std::fs::remove_file(&dest);
        }
        // Otherwise the file is shorter than expected: a partial download that
        // `curl -C -` resumes below.
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        item.repo_id, item.filename
    );
    eprintln!(
        "Downloading {} ({:.1} GB) from {}",
        item.name,
        target as f64 / 1e9,
        item.repo_id
    );

    // -L follow redirects, -C - resume a partial file, --fail surface HTTP
    // errors as a non-zero exit. curl writes its own progress bar to stderr.
    let status = std::process::Command::new("curl")
        .args(["-L", "-C", "-", "--fail", "-o"])
        .arg(&dest)
        .arg(&url)
        .status()
        .map_err(|err| anyhow::anyhow!("could not run curl (is it installed?): {err}"))?;

    let have_after = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);

    if !status.success() {
        // curl returns non-zero (HTTP 416) when asked to resume a file that is
        // already complete. If every byte is here, that's success, not failure;
        // otherwise the download genuinely broke.
        if have_after == 0 || have_after != target {
            anyhow::bail!("download failed (curl exited with {status}); re-run to resume");
        }
    }

    // Final integrity gate: never hand back a short or oversized file when the
    // Hub told us the size to expect.
    if expected.is_some() && have_after != target {
        anyhow::bail!(
            "download incomplete: {} is {have_after} bytes, expected {target} — re-run to resume",
            item.filename
        );
    }

    Ok(dest)
}

fn print_catalog(entries: &[CatalogItem]) {
    eprintln!("Supported models (download into ./models):\n");
    // Annotate each row with a capacity verdict for THIS host (fit axis only — not
    // a support claim). Probed once, reused across rows.
    let hw = crate::capability::HardwareProfile::cached();
    for line in catalog_table_lines(entries, hw) {
        eprintln!("{line}");
    }
}

/// Render the catalog as aligned table lines (header first).
///
/// Every column is sized to the wider of its header and its widest cell, so no
/// value can overflow a hardcoded width and shove a row's later columns out of
/// line -- the original code hardcoded a 28-wide id column, which the 29-char
/// `mistral_7b_instruct_v0_3_q8_0` id overflowed, and a SIZE header a column
/// narrower than its `X.X GB` cells. Kept separate from `print_catalog` so the
/// alignment can be unit-tested.
///
/// Widths are byte lengths, so alignment assumes ASCII cells. That holds here:
/// only curated `CatalogItem`s (all `&'static str`, ASCII) reach this function;
/// arbitrary/UTF-8 Hugging Face rows are the separate owned view type and are
/// never printed here. A future caller passing non-ASCII names would need a
/// display-width measure instead of `len()`.
fn catalog_table_lines(
    entries: &[CatalogItem],
    hw: &crate::capability::HardwareProfile,
) -> Vec<String> {
    const ID_H: &str = "ID";
    const QUANT_H: &str = "QUANT";
    const SIZE_H: &str = "SIZE";
    const FIT_H: &str = "FIT (this host)";
    const NAME_H: &str = "NAME";

    // Materialize each row's rendered cells up front so the column widths can be
    // measured from the real strings (not guessed).
    struct Cells {
        id: String,
        quant: String,
        size: String,
        fit: String,
        name: String,
    }
    let rows: Vec<Cells> = entries
        .iter()
        .map(|item| {
            let verdict = crate::fit::assess(hw, &crate::fit::advisory_footprint(item.size_bytes));
            Cells {
                id: item.catalog_id.to_string(),
                quant: item.quant.to_string(),
                size: format!("{:.1} GB", item.size_bytes as f64 / 1e9),
                fit: verdict.cli_label().to_string(),
                name: item.name.to_string(),
            }
        })
        .collect();

    let width = |header: &str, cell: fn(&Cells) -> &str| {
        rows.iter()
            .map(|r| cell(r).len())
            .max()
            .unwrap_or(0)
            .max(header.len())
    };
    let id_w = width(ID_H, |r| r.id.as_str());
    let quant_w = width(QUANT_H, |r| r.quant.as_str());
    let size_w = width(SIZE_H, |r| r.size.as_str());
    let fit_w = width(FIT_H, |r| r.fit.as_str());

    // SIZE is right-aligned (numbers), the rest left-aligned. Two spaces set the
    // FIT column off from SIZE; single spaces separate the others.
    let row_line = |id: &str, quant: &str, size: &str, fit: &str, name: &str| {
        format!("  {id:<id_w$} {quant:<quant_w$} {size:>size_w$}  {fit:<fit_w$} {name}")
    };

    let mut lines = Vec::with_capacity(rows.len() + 1);
    lines.push(row_line(ID_H, QUANT_H, SIZE_H, FIT_H, NAME_H));
    for r in &rows {
        lines.push(row_line(&r.id, &r.quant, &r.size, &r.fit, &r.name));
    }
    lines
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | '.' | ' '))
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_by_id_fragment_ignoring_separators() {
        let entries = curated_catalog();
        let item = resolve(&entries, "3b-instruct-q8").unwrap();
        assert_eq!(item.catalog_id, "llama32_3b_instruct_q8_0");
    }

    #[test]
    fn a_family_fragment_spanning_several_quants_refuses_to_guess() {
        // BEHAVIOR CHANGE: `llama32_3b` was the README quickstart id and resolved
        // to the Q8_0 row while that file was the only Llama 3.2 3B in the
        // catalog. The certified Q4_K_M and Q5_K_M rows are now pullable too, so
        // the fragment names three different multi-GB downloads and no longer
        // resolves. Keeping the historical winner would be a guess dressed as
        // resolution; the docs moved to quant-qualified ids instead
        // (`3b_instruct_q8`). The error must list every candidate so the next
        // command is obvious.
        let entries = curated_catalog();
        for ambiguous in ["llama32-3b", "3b-instruct"] {
            let err = resolve(&entries, ambiguous).expect_err("family fragment is ambiguous");
            let message = err.to_string();
            for expected in [
                "llama32_3b_instruct_q8_0",
                "llama_3_2_3b_instruct_q4_k_m",
                "llama_3_2_3b_instruct_q5_k_m",
            ] {
                assert!(
                    message.contains(expected),
                    "{expected} missing from {ambiguous:?} error: {message}"
                );
            }
        }
    }

    #[test]
    fn the_documented_quant_qualified_ids_all_resolve() {
        // Every pull id printed in the README table must resolve to exactly the
        // row it names -- that table is the only place most users look.
        let entries = curated_catalog();
        for (id, expected) in [
            ("1b_instruct_q8", "llama32_1b_instruct_q8_0"),
            ("iq4_xs", "llama3_2_1b_instruct_iq4_xs"),
            ("3b_instruct_q8", "llama32_3b_instruct_q8_0"),
            ("3b_instruct_q4", "llama_3_2_3b_instruct_q4_k_m"),
            ("3b_instruct_q5", "llama_3_2_3b_instruct_q5_k_m"),
            ("ornith", "Ornith 1.0 9B"),
        ] {
            let item =
                resolve(&entries, id).unwrap_or_else(|err| panic!("id {id} must resolve: {err}"));
            assert_eq!(item.catalog_id, expected, "id {id}");
        }
    }

    #[test]
    fn a_fully_typed_id_beats_a_longer_id_that_contains_it() {
        // Exactness is the tie-break, not row order: the whole id is the most
        // specific thing a user can type, so it must never lose to a longer id
        // that merely contains it as a substring.
        let entries = vec![
            synthetic("qwen3_4b_q4_k_m", "Q4_K_M", 1, "Qwen3 4B Q4_K_M"),
            synthetic(
                "qwen3_4b_q4_k_m_imatrix",
                "Q4_K_M",
                2,
                "Qwen3 4B Q4_K_M imatrix",
            ),
        ];
        let item = resolve(&entries, "qwen3_4b_q4_k_m").expect("exact id resolves");
        assert_eq!(item.catalog_id, "qwen3_4b_q4_k_m");
    }

    #[test]
    fn resolves_by_name_fragment() {
        let entries = curated_catalog();
        let item = resolve(&entries, "tinyllama").unwrap();
        assert!(item.catalog_id.contains("tinyllama"));
    }

    #[test]
    fn unknown_query_is_an_error() {
        let entries = curated_catalog();
        assert!(resolve(&entries, "gpt-9-turbo").is_err());
    }

    /// Build a throwaway catalog row for alignment tests. Only the fields the
    /// table renders (`catalog_id`, `quant`, `size_bytes`, `name`) matter; the
    /// rest are filler.
    fn synthetic(
        catalog_id: &'static str,
        quant: &'static str,
        size_bytes: u64,
        name: &'static str,
    ) -> CatalogItem {
        CatalogItem {
            catalog_id,
            name,
            repo_id: "org/repo",
            filename: "model.gguf",
            size_bytes,
            downloads: 0,
            likes: 0,
            quant,
            architecture: "llama",
            license: "apache-2.0",
            task_tags: &["general"],
        }
    }

    /// Assert the header and every data row share the same column geometry.
    ///
    /// Column offsets are read from the header, then checked against each row:
    /// a left-aligned column's content must begin exactly under its header
    /// label (with a separating space just before it), and SIZE (right-aligned)
    /// must end exactly two spaces before the FIT column. If any earlier cell
    /// overflowed its width, these offsets shift and the assertions fire. All
    /// rendered content is ASCII, so byte indices equal display columns.
    fn assert_aligned(lines: &[String]) {
        assert!(!lines.is_empty(), "expected at least a header line");
        let header = &lines[0];
        let quant_col = header.find("QUANT").expect("header names QUANT");
        let size_end = header.find("  FIT (this host)").expect("header names FIT");
        let fit_col = header.find("FIT (this host)").expect("header names FIT");
        let name_col = header.rfind("NAME").expect("header names NAME");

        for line in &lines[1..] {
            let b = line.as_bytes();
            assert!(
                b.len() > name_col,
                "row is shorter than the header columns: {line:?}"
            );
            // QUANT (left-aligned) begins under its header label.
            assert_eq!(
                b[quant_col - 1],
                b' ',
                "no separator before QUANT: {line:?}"
            );
            assert_ne!(b[quant_col], b' ', "QUANT overflowed by the id: {line:?}");
            // SIZE (right-aligned) ends exactly two spaces before FIT.
            assert_eq!(b[size_end], b' ', "SIZE/FIT gap wrong: {line:?}");
            assert_eq!(b[size_end + 1], b' ', "SIZE/FIT gap wrong: {line:?}");
            assert_ne!(
                b[size_end - 1],
                b' ',
                "SIZE right edge misaligned: {line:?}"
            );
            // FIT (left-aligned) begins under its header label.
            assert_eq!(b[fit_col - 1], b' ', "no separator before FIT: {line:?}");
            assert_ne!(b[fit_col], b' ', "FIT column misaligned: {line:?}");
            // NAME (left-aligned) begins under its header label.
            assert_eq!(b[name_col - 1], b' ', "no separator before NAME: {line:?}");
            assert_ne!(b[name_col], b' ', "NAME column misaligned: {line:?}");
        }
    }

    #[test]
    fn catalog_table_columns_stay_aligned() {
        // Regression: `mistral_7b_instruct_v0_3_q8_0` (29 chars) used to overflow
        // a hardcoded 28-wide id column and push that row's later columns one
        // space right of the header, and the SIZE header was a column narrower
        // than the rendered `X.X GB` cells.
        let entries = curated_catalog();
        let hw = crate::capability::HardwareProfile::cached();
        let lines = catalog_table_lines(&entries, hw);
        assert_eq!(lines.len(), entries.len() + 1, "header + one line per row");
        assert!(
            entries
                .iter()
                .any(|e| e.catalog_id == "mistral_7b_instruct_v0_3_q8_0"),
            "the longest id must still be in the catalog for this regression to bite"
        );
        assert_aligned(&lines);
    }

    #[test]
    fn long_id_and_quant_do_not_overflow() {
        let hw = crate::capability::HardwareProfile::cached();
        let entries = vec![
            synthetic("short_q8_0", "Q8_0", 1_300_000_000, "Short Model"),
            synthetic(
                "an_extremely_long_catalog_id_that_would_overflow_q8_0",
                "Q4_K_M",
                7_700_000_000,
                "Long Id Model",
            ),
            synthetic(
                "mid_q4_0",
                "A_VERY_LONG_QUANT_LABEL",
                3_400_000_000,
                "Mid Model",
            ),
        ];
        let lines = catalog_table_lines(&entries, hw);
        assert_eq!(lines.len(), entries.len() + 1);
        assert_aligned(&lines);
    }

    #[test]
    fn oversized_size_value_keeps_columns_aligned() {
        let hw = crate::capability::HardwareProfile::cached();
        let entries = vec![
            synthetic("small_q8_0", "Q8_0", 900_000_000, "Small"), // "0.9 GB"
            synthetic("huge_q8_0", "Q8_0", 123_456_789_000, "Huge"), // "123.5 GB"
        ];
        let lines = catalog_table_lines(&entries, hw);
        assert_aligned(&lines);
        assert!(
            lines.iter().any(|l| l.contains("123.5 GB")),
            "the wide size cell must render and widen its column"
        );
    }

    /// Proof the alignment assertions are non-tautological: the *original*
    /// hardcoded-width rendering must be REJECTED by `assert_aligned`. These are
    /// the exact format strings this PR removed: a 28-wide id column (which the
    /// 29-char `mistral_7b_instruct_v0_3_q8_0` overflows) and an 8-wide SIZE
    /// header against 9-char `{:>6.1} GB` cells.
    #[test]
    fn assert_aligned_rejects_the_original_buggy_layout() {
        let entries = curated_catalog();
        let hw = crate::capability::HardwareProfile::cached();
        let mut lines = vec![format!(
            "  {:<28} {:<8} {:>8}  {:<15} NAME",
            "ID", "QUANT", "SIZE", "FIT (this host)"
        )];
        for item in &entries {
            let verdict = crate::fit::assess(hw, &crate::fit::advisory_footprint(item.size_bytes));
            lines.push(format!(
                "  {:<28} {:<8} {:>6.1} GB  {:<15} {}",
                item.catalog_id,
                item.quant,
                item.size_bytes as f64 / 1e9,
                verdict.cli_label(),
                item.name,
            ));
        }
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(|| assert_aligned(&lines));
        std::panic::set_hook(prev);
        assert!(
            caught.is_err(),
            "assert_aligned should reject the original misaligned layout"
        );
    }

    #[test]
    fn empty_catalog_renders_header_only() {
        let hw = crate::capability::HardwareProfile::cached();
        let lines = catalog_table_lines(&[], hw);
        assert_eq!(lines.len(), 1, "no rows means header only, no panic");
        for label in ["ID", "QUANT", "SIZE", "FIT (this host)", "NAME"] {
            assert!(lines[0].contains(label), "header missing {label}");
        }
    }

    #[test]
    fn qwen3_4b_q4_k_m_is_pullable_by_its_ledger_id() {
        // Regression for upstream #469: the supported `qwen3_4b_q4_k_m` exact row
        // appeared in the ledger, the TUI picker, and the Web UI browse but was
        // missing from the pull catalog, so `camelid pull qwen3_4b_q4_k_m` failed
        // with "no supported model matches". Its catalog id must equal the ledger
        // row id (the picker joins on it) and point at the official Qwen upload.
        let entries = curated_catalog();
        let item = resolve(&entries, "qwen3_4b_q4_k_m").expect("row must resolve");
        assert_eq!(item.catalog_id, "qwen3_4b_q4_k_m");
        assert_eq!(item.repo_id, "Qwen/Qwen3-4B-GGUF");
        assert_eq!(item.filename, "Qwen3-4B-Q4_K_M.gguf");
        assert_eq!(item.quant, "Q4_K_M");
    }

    #[test]
    fn curated_dry_run_resolves_offline_and_writes_nothing() {
        // --dry-run must stop after resolution: no models dir, no bytes, no
        // network (the remote size check lives inside download()).
        let dir = std::env::temp_dir().join(format!(
            "camelid-catalog-dry-run-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        run_pull_opts(Some("qwen3_4b_q4_k_m"), &dir, true).expect("dry-run resolves");
        assert!(!dir.exists(), "dry-run must not create the models dir");
    }
}
