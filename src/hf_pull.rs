//! `camelid pull org/repo[:quant]` — fetch an arbitrary Hugging Face GGUF from
//! the terminal, the CLI twin of the Models page's "Experimental (Hugging Face)"
//! browse lane.
//!
//! Policy: a download path is not a support claim. Everything fetched here is
//! experimental-lane — unverified, no parity claim — and whether the file loads
//! is decided at load time by the inspect-first typed-blocker flow (fail-closed),
//! exactly as for a file downloaded from the web UI. This module therefore
//! discovers, selects, and downloads; it never asserts runnability.
//!
//! Selection is deliberately fail-closed too: a repo with several GGUFs requires
//! an explicit `:quant` tag rather than guessing on the user's behalf, and a tag
//! only matches on token boundaries so `:F16` cannot silently pick a `BF16`
//! file. Vision projector (`mmproj`) companions and multi-part shards are never
//! selectable — the engine loads single-file models from `models/*.gguf`.
//!
//! Network and disk mechanics follow the audited paths: the tree listing reuses
//! `hf_browse` (LFS-aware sizes, curl subprocess, offline degrades to a typed
//! error) and the download mirrors the web installer's semantics — bare-filename
//! destinations gated by `valid_local_model_filename`, a `.part` file promoted
//! by rename only after the size gate passes, resume via `curl -C -`, and the
//! download ceiling (`CAMELID_MAX_DOWNLOAD_BYTES`; `serve --hf` honors the
//! `--max-download-bytes` flag) enforced both before and during the transfer.
//! Downloads are anonymous: gated/private repos are not supported.

use std::path::{Path, PathBuf};

use crate::hf_browse::HfGgufFile;

/// A parsed `org/repo[:quant]` spec. `quant` is stored uppercased; matching is
/// case-insensitive throughout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfModelSpec {
    pub repo_id: String,
    pub quant: Option<String>,
}

/// `pull` treats any query containing `/` as a Hugging Face spec: curated
/// catalog ids never contain one.
pub fn is_hf_spec(query: &str) -> bool {
    query.contains('/')
}

impl HfModelSpec {
    /// Parse `org/repo[:quant]`, tolerating pasted `hf.co/…` and
    /// `huggingface.co/…` URL prefixes and a trailing slash.
    pub fn parse(raw: &str) -> anyhow::Result<Self> {
        let mut rest = raw.trim();
        for prefix in [
            "https://huggingface.co/",
            "http://huggingface.co/",
            "https://hf.co/",
            "http://hf.co/",
            "huggingface.co/",
            "hf.co/",
        ] {
            // `get` (not a direct slice): a multi-byte character straddling the
            // prefix byte length must fall through to the parse error below, not
            // panic on a non-char-boundary index.
            if let Some(head) = rest.get(..prefix.len()) {
                if head.eq_ignore_ascii_case(prefix) {
                    rest = &rest[prefix.len()..];
                    break;
                }
            }
        }

        let (repo_part, quant) = match rest.rsplit_once(':') {
            Some((repo, tag)) => (repo, Some(tag)),
            None => (rest, None),
        };
        let repo_id = repo_part.trim_end_matches('/');

        let parts: Vec<&str> = repo_id.split('/').collect();
        let repo_ok = parts.len() == 2
            && parts.iter().all(|part| !part.is_empty())
            && crate::fit_dims::is_safe_hf_component(repo_id);
        if !repo_ok {
            anyhow::bail!(
                "\"{raw}\" is not a Hugging Face model spec — expected org/repo or org/repo:QUANT \
                 (e.g. prism-ml/Ternary-Bonsai-27B-gguf:Q2_0)"
            );
        }

        let quant = match quant {
            None => None,
            Some(tag) => {
                let tag = tag.trim();
                let tag_ok = !tag.is_empty()
                    && tag
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
                if !tag_ok {
                    anyhow::bail!(
                        "\"{raw}\" has an invalid quant tag — write org/repo:QUANT \
                         (e.g. :Q8_0) or drop the colon"
                    );
                }
                Some(tag.to_uppercase())
            }
        };

        Ok(Self {
            repo_id: repo_id.to_string(),
            quant,
        })
    }
}

/// Resolve and download a spec into `models_dir`, printing the follow-up
/// `camelid serve` command. `dry_run` stops after resolution and prints what
/// would be downloaded. Returns the destination path.
pub fn run_hf_pull(raw_spec: &str, models_dir: &Path, dry_run: bool) -> anyhow::Result<PathBuf> {
    pull_spec(raw_spec, models_dir, dry_run, true, None)
}

/// `serve --hf` / `chat --hf` entry point: download-if-missing, no
/// "serve it" hint (the caller is already starting the engine).
/// `max_download_bytes` overrides the `CAMELID_MAX_DOWNLOAD_BYTES` env default
/// — `serve` passes its `--max-download-bytes` value so the flag governs this
/// lane too; callers without such a flag (chat) pass `None`.
pub fn ensure_hf_model(
    raw_spec: &str,
    models_dir: &Path,
    max_download_bytes: Option<u64>,
) -> anyhow::Result<PathBuf> {
    pull_spec(raw_spec, models_dir, false, false, max_download_bytes)
}

fn pull_spec(
    raw_spec: &str,
    models_dir: &Path,
    dry_run: bool,
    print_next_step: bool,
    ceiling_override: Option<u64>,
) -> anyhow::Result<PathBuf> {
    let spec = HfModelSpec::parse(raw_spec)?;

    eprintln!("Listing GGUF files in {} …", spec.repo_id);
    let files = crate::hf_browse::list_gguf_files_blocking(&spec.repo_id).map_err(|err| {
        anyhow::anyhow!(
            "could not list {}: {err} — check the repo id; gated or private repos are not \
             supported (camelid downloads anonymously)",
            spec.repo_id
        )
    })?;
    if files.is_empty() {
        anyhow::bail!(
            "{} has no top-level .gguf files — camelid loads single-file GGUFs from the top \
             level of a repo",
            spec.repo_id
        );
    }

    let chosen = select_file(&spec, &files)?;

    // Disclosure wording is the web UI's, verbatim: this lane never claims more
    // than "the bytes arrived".
    eprintln!("\nExperimental — unverified, no parity claim.");
    eprintln!(
        "A download path is not a support claim; whether this file loads is decided at load \
         time, fail-closed."
    );

    let dest = models_dir.join(&chosen.filename);
    if dry_run {
        eprintln!("\nWould download:");
        eprintln!("  repo:  {}", chosen.repo_id);
        eprintln!(
            "  file:  {} ({})",
            chosen.filename,
            human_gb(chosen.size_bytes)
        );
        eprintln!("  url:   {}", resolve_url(&chosen));
        eprintln!("  dest:  {}", dest.display());
        return Ok(dest);
    }

    let ceiling = ceiling_override.unwrap_or_else(max_download_bytes);
    let dest = download(&chosen, models_dir, ceiling)?;

    eprintln!("\n✓ {} downloaded to {}", chosen.filename, dest.display());
    if files.iter().any(|file| is_mmproj(&file.filename)) {
        eprintln!(
            "note: {} also ships vision projector (mmproj) files; `camelid pull` fetches the \
             model file only.",
            spec.repo_id
        );
    }
    if print_next_step {
        // No runnability claim: loading is decided fail-closed by the engine.
        eprintln!(
            "\nServe it (loads only if the file's architecture is implemented; fails closed \
             otherwise):\n  camelid serve --model {}",
            dest.display()
        );
    }
    Ok(dest)
}

/// Pick exactly one downloadable file for `spec`, or explain why the CLI will
/// not guess. Companions (mmproj) and multi-part shards are excluded up front.
fn select_file(spec: &HfModelSpec, files: &[HfGgufFile]) -> anyhow::Result<HfGgufFile> {
    let candidates: Vec<&HfGgufFile> = files
        .iter()
        .filter(|file| !is_mmproj(&file.filename) && !is_multipart_shard(&file.filename))
        .collect();

    if candidates.is_empty() {
        if files.iter().any(|file| is_multipart_shard(&file.filename)) {
            anyhow::bail!(
                "{} only has multi-part GGUFs, which camelid pull does not support",
                spec.repo_id
            );
        }
        anyhow::bail!(
            "{} has no standalone GGUF model files (only mmproj projector files)",
            spec.repo_id
        );
    }

    let Some(tag) = &spec.quant else {
        if let [only] = candidates.as_slice() {
            return Ok((*only).clone());
        }
        print_file_list(spec, &candidates);
        anyhow::bail!(
            "{} has {} GGUF files — add :<quant> to pick one (e.g. camelid pull {}:{})",
            spec.repo_id,
            candidates.len(),
            spec.repo_id,
            listing_tags(&candidates)[0].0
        );
    };

    // Stage 1: a tag that IS a recognized quant label matches that label exactly
    // (so `:Q4_K` picks Q4_K, never Q4_K_M — labels are boundary-matched with
    // the superstring variants ranked first, so a Q6_K_L file is labeled
    // Q6_K_L, never Q6_K).
    let exact: Vec<&HfGgufFile> = candidates
        .iter()
        .copied()
        .filter(|file| file.quant == *tag)
        .collect();
    if let [only] = exact.as_slice() {
        return Ok((*only).clone());
    }
    // A recognized label that no file carries must NOT fall through to the
    // substring stage: `:Q4_K` in a repo shipping only Q4_K_M would otherwise
    // silently download a different quantization than the user named.
    if exact.is_empty() && crate::hf_browse::known_quant_label(tag) {
        print_file_list(spec, &candidates);
        anyhow::bail!(
            "no file in {} has quant {tag} — pick a tag from the list above",
            spec.repo_id
        );
    }

    // Stage 2: token-boundary substring against the filename, so unrecognized
    // labels (PQ2_0, Q2_g64, …) are still addressable, but `:F16` cannot match
    // a `BF16` file.
    let matched: Vec<&HfGgufFile> = candidates
        .iter()
        .copied()
        .filter(|file| tag_matches(&file.filename, tag))
        .collect();
    match matched.as_slice() {
        [] => {
            print_file_list(spec, &candidates);
            anyhow::bail!(
                "no file in {} matches :{tag} — pick a tag from the list above",
                spec.repo_id
            );
        }
        [only] => Ok((*only).clone()),
        several => {
            print_file_list(spec, several);
            anyhow::bail!(
                ":{tag} matches several files in {} — be more specific",
                spec.repo_id
            );
        }
    }
}

/// True when `tag` (uppercase ASCII) occurs in the filename delimited by
/// non-alphanumerics (or the string ends), compared case-insensitively. Shares
/// the boundary matcher with `guess_quant` so tag selection and labeling can
/// never disagree about what counts as a token.
fn tag_matches(filename: &str, tag: &str) -> bool {
    crate::hf_browse::token_bounded_contains(filename, tag)
}

/// Vision projector companions: never a standalone model.
fn is_mmproj(filename: &str) -> bool {
    filename.to_ascii_lowercase().contains("mmproj")
}

/// `…-00001-of-00003.gguf`-style shard names (any digit widths).
fn is_multipart_shard(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    let Some(stem) = lower.strip_suffix(".gguf") else {
        return false;
    };
    let mut tail = stem.rsplit('-');
    let (Some(total), Some(of), Some(index)) = (tail.next(), tail.next(), tail.next()) else {
        return false;
    };
    of == "of"
        && !total.is_empty()
        && total.bytes().all(|b| b.is_ascii_digit())
        && !index.is_empty()
        && index.bytes().all(|b| b.is_ascii_digit())
}

/// The `:tag` a user could type to select `file` — its recognized quant label
/// when there is one, else the last `-`-separated token of the stem (which the
/// boundary matcher will find case-insensitively), else the whole filename.
fn suggested_tag(file: &HfGgufFile) -> String {
    if !file.quant.is_empty() {
        return file.quant.clone();
    }
    let stem = file_stem(&file.filename);
    stem.rsplit('-')
        .next()
        .filter(|token| !token.is_empty())
        .unwrap_or(stem)
        .to_uppercase()
}

fn file_stem(filename: &str) -> &str {
    filename
        .strip_suffix(".gguf")
        .or_else(|| filename.strip_suffix(".GGUF"))
        .unwrap_or(filename)
}

/// The `(tag, file)` pairs the listing prints. Every printed tag must actually
/// select its file: when two files would get the same suggested tag (two files
/// sharing a quant label), the colliding ones fall back to their full stem,
/// which the boundary matcher resolves uniquely.
fn listing_tags<'a>(files: &[&'a HfGgufFile]) -> Vec<(String, &'a HfGgufFile)> {
    let first_pass: Vec<String> = files.iter().map(|file| suggested_tag(file)).collect();
    files
        .iter()
        .zip(&first_pass)
        .map(|(file, tag)| {
            let collides = first_pass.iter().filter(|other| *other == tag).count() > 1;
            if collides {
                (file_stem(&file.filename).to_uppercase(), *file)
            } else {
                (tag.clone(), *file)
            }
        })
        .collect()
}

fn print_file_list(spec: &HfModelSpec, files: &[&HfGgufFile]) {
    eprintln!("GGUF files in {}:", spec.repo_id);
    for (tag, file) in listing_tags(files) {
        eprintln!(
            "  {:>10}  :{:<10}  {}",
            human_gb(file.size_bytes),
            tag,
            file.filename
        );
    }
}

fn human_gb(bytes: u64) -> String {
    if bytes == 0 {
        return "unknown size".to_string();
    }
    format!("{:.1} GB", bytes as f64 / 1e9)
}

fn resolve_url(file: &HfGgufFile) -> String {
    // repo_id is gated by `is_safe_hf_component` at parse time; the filename is
    // percent-encoded so spaces and reserved characters cannot mangle the URL.
    format!(
        "https://huggingface.co/{}/resolve/main/{}",
        file.repo_id,
        crate::hf_browse::urlencode(&file.filename)
    )
}

/// Download ceiling from the `CAMELID_MAX_DOWNLOAD_BYTES` env var, defaulting
/// to the serve-side installer's 64 GiB constant.
fn max_download_bytes() -> u64 {
    parse_max_download_bytes(std::env::var("CAMELID_MAX_DOWNLOAD_BYTES").ok().as_deref())
}

/// Pure parse half of [`max_download_bytes`]. Unparseable or ZERO values fall
/// back to the default: a 0 ceiling would block every pull. (Deliberate
/// difference from `serve`, which treats an explicit 0 as a startup error via
/// `ServeOptions` validation rather than silently substituting the default.)
fn parse_max_download_bytes(raw: Option<&str>) -> u64 {
    raw.and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(crate::api::DEFAULT_MAX_DOWNLOAD_BYTES)
}

/// What to do about an existing file at the destination path before any bytes
/// move. Pure so the never-overwrite trichotomy is unit-testable.
#[derive(Debug, PartialEq, Eq)]
enum DestState {
    /// No usable file at dest — proceed to download.
    Missing,
    /// A zero-byte leftover — safe to replace (nothing of the user's to lose).
    ReplaceEmpty,
    /// Byte count matches the Hub — keep it, skip the download.
    Complete,
    /// Dest exists but the Hub reports no size — keep the local copy.
    KeepUnknownSize,
    /// Same filename, different bytes (possibly another repo's file) — error,
    /// never overwrite.
    Conflict { have: u64 },
}

fn classify_dest(dest_len: Option<u64>, expected: u64) -> DestState {
    match dest_len {
        None => DestState::Missing,
        Some(0) => DestState::ReplaceEmpty,
        Some(_) if expected == 0 => DestState::KeepUnknownSize,
        Some(have) if have == expected => DestState::Complete,
        Some(have) => DestState::Conflict { have },
    }
}

/// Windows reserved device names (CON, NUL, COM1…) — writing them as filenames
/// misbehaves on Windows, and upstream repo filenames are attacker-adjacent.
/// `valid_local_model_filename` does not cover these (pre-existing gap in the
/// web lane too), so this lane rejects them itself.
fn windows_reserved_stem(filename: &str) -> bool {
    let stem = file_stem(filename);
    let base = stem.split('.').next().unwrap_or(stem);
    let upper = base.to_ascii_uppercase();
    matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (upper.len() == 4
            && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && upper.as_bytes()[3].is_ascii_digit())
}

/// Download `file` into `models_dir` with the web installer's semantics: write
/// to `<filename>.part`, resume with `curl -C -`, verify the byte count against
/// the tree listing, and only then rename into place — a loadable GGUF never
/// exists half-written. An existing complete copy is kept; an existing
/// same-named file with DIFFERENT bytes is an error, never overwritten (the
/// flat models dir means two repos can ship the same filename). A stale `.part`
/// larger than the Hub's current size is deleted up front: resume can never
/// shrink a file, so keeping it would loop "re-run to resume" forever.
fn download(file: &HfGgufFile, models_dir: &Path, max_bytes: u64) -> anyhow::Result<PathBuf> {
    if !crate::model_default::valid_local_model_filename(&file.filename)
        || windows_reserved_stem(&file.filename)
    {
        anyhow::bail!(
            "refusing \"{}\" — upstream filenames must be bare *.gguf names",
            file.filename
        );
    }
    let expected = file.size_bytes; // LFS-aware from the tree listing; 0 = unknown
    if expected > max_bytes {
        anyhow::bail!(
            "{} is {} which exceeds the {} download ceiling — raise it with \
             serve --max-download-bytes or CAMELID_MAX_DOWNLOAD_BYTES",
            file.filename,
            human_gb(expected),
            human_gb(max_bytes)
        );
    }

    std::fs::create_dir_all(models_dir)?;
    let dest = models_dir.join(&file.filename);
    let part = models_dir.join(format!("{}.part", file.filename));

    match classify_dest(std::fs::metadata(&dest).ok().map(|m| m.len()), expected) {
        DestState::Missing => {}
        DestState::ReplaceEmpty => {
            let _ = std::fs::remove_file(&dest);
        }
        DestState::Complete => {
            eprintln!(
                "{} already downloaded at {} ({}, size-verified)",
                file.filename,
                dest.display(),
                human_gb(expected)
            );
            return Ok(dest);
        }
        DestState::KeepUnknownSize => {
            eprintln!(
                "{} already present at {} (upstream size unknown; keeping the local copy)",
                file.filename,
                dest.display()
            );
            return Ok(dest);
        }
        DestState::Conflict { have } => {
            anyhow::bail!(
                "{} already exists at {} with {have} bytes but the Hub reports {expected} — \
                 remove the local file or pull into a different --models-dir",
                file.filename,
                dest.display()
            );
        }
    }

    // A .part larger than the Hub's current size is unrecoverable by resume
    // (upstream re-published smaller, or the partial is corrupt): start clean
    // instead of failing "re-run to resume" forever.
    if expected != 0 {
        if let Ok(meta) = std::fs::metadata(&part) {
            if meta.len() > expected {
                eprintln!(
                    "Discarding stale partial {} ({} bytes; the Hub file is {expected}) — \
                     restarting the download fresh",
                    part.display(),
                    meta.len()
                );
                let _ = std::fs::remove_file(&part);
            }
        }
    }

    eprintln!(
        "Downloading {} ({}) from {}",
        file.filename,
        human_gb(expected),
        file.repo_id
    );

    // Mirrors the web installer's curl line (resume, retries, stall detection,
    // in-flight ceiling) but keeps curl's own progress meter on stderr.
    let status = std::process::Command::new("curl")
        .args([
            "-f",
            "-L",
            "-C",
            "-",
            "--connect-timeout",
            "30",
            "--retry",
            "10",
            "--retry-delay",
            "2",
            "--retry-all-errors",
            "--speed-limit",
            "1024",
            "--speed-time",
            "30",
            "--max-filesize",
            &max_bytes.to_string(),
            "-o",
        ])
        .arg(&part)
        .arg(resolve_url(file))
        .status()
        .map_err(|err| anyhow::anyhow!("could not run curl (is it installed?): {err}"))?;

    let have_after = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

    // An oversized partial can never be repaired by resume; delete it so the
    // next run starts clean instead of failing identically forever.
    let discard_oversized_part = |have: u64| {
        if expected != 0 && have > expected {
            let _ = std::fs::remove_file(&part);
            true
        } else {
            false
        }
    };

    if !status.success() {
        // curl exits non-zero (HTTP 416) when asked to resume an already-complete
        // .part; every byte being present is success, not failure.
        if expected == 0 || have_after != expected {
            if discard_oversized_part(have_after) {
                anyhow::bail!(
                    "download failed (curl exited with {status}); the oversized partial at {} \
                     was discarded — re-run to retry",
                    part.display()
                );
            }
            anyhow::bail!(
                "download failed (curl exited with {status}); re-run to resume {}",
                part.display()
            );
        }
    }
    if expected != 0 && have_after != expected {
        if discard_oversized_part(have_after) {
            anyhow::bail!(
                "download produced {have_after} bytes but the Hub reports {expected}; the stale \
                 partial at {} was discarded — re-run to retry",
                part.display()
            );
        }
        anyhow::bail!(
            "download incomplete: {} is {have_after} bytes, expected {expected} — re-run to \
             resume",
            part.display()
        );
    }
    if have_after == 0 {
        anyhow::bail!(
            "download produced no bytes for {} — re-run to retry",
            file.filename
        );
    }

    std::fs::rename(&part, &dest)?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(name: &str, size: u64) -> HfGgufFile {
        HfGgufFile {
            repo_id: "org/repo".to_string(),
            filename: name.to_string(),
            size_bytes: size,
            downloads: 0,
            likes: 0,
            architecture: String::new(),
            quant: crate::hf_browse::guess_quant(name).unwrap_or_default(),
        }
    }

    /// The real prism-ml/Ternary-Bonsai-27B-gguf layout (2026-08): nonstandard
    /// quant labels, a dspark variant pair, and two mmproj companions.
    fn ternary_bonsai() -> Vec<HfGgufFile> {
        vec![
            file("Ternary-Bonsai-27B-F16.gguf", 53_808_280_640),
            file("Ternary-Bonsai-27B-PQ2_0.gguf", 7_165_121_600),
            file("Ternary-Bonsai-27B-Q2_0.gguf", 7_165_121_600),
            file("Ternary-Bonsai-27B-Q2_g64.gguf", 7_585_330_240),
            file("Ternary-Bonsai-27B-dspark-Q4_1.gguf", 1_946_393_568),
            file("Ternary-Bonsai-27B-dspark-bf16.gguf", 7_291_885_792),
            file("Ternary-Bonsai-27B-mmproj-BF16.gguf", 931_145_760),
            file("Ternary-Bonsai-27B-mmproj-Q8_0.gguf", 629_246_880),
        ]
    }

    fn spec(raw: &str) -> HfModelSpec {
        HfModelSpec::parse(raw).expect("valid spec")
    }

    #[test]
    fn parses_repo_and_quant_specs() {
        assert_eq!(
            spec("prism-ml/Ternary-Bonsai-27B-gguf"),
            HfModelSpec {
                repo_id: "prism-ml/Ternary-Bonsai-27B-gguf".to_string(),
                quant: None,
            }
        );
        assert_eq!(
            spec("org/repo:q4_k_m").quant.as_deref(),
            Some("Q4_K_M"),
            "quant tags are canonicalized to uppercase"
        );
        assert_eq!(spec("hf.co/org/repo").repo_id, "org/repo");
        assert_eq!(spec("https://huggingface.co/org/repo/").repo_id, "org/repo");
        assert_eq!(spec("HF.co/org/repo:Q2_0").quant.as_deref(), Some("Q2_0"));
    }

    #[test]
    fn rejects_malformed_specs() {
        for bad in [
            "org",
            "org/",
            "/repo",
            "org/repo/extra",
            "org/../repo",
            "org/repo:",
            "org/repo: ",
            "org/re po",
            "org/repo:Q4?",
            "org/repo:Q4/K",
        ] {
            assert!(
                HfModelSpec::parse(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn is_hf_spec_only_for_slashed_queries() {
        assert!(is_hf_spec("org/repo"));
        assert!(!is_hf_spec("llama32_3b"));
    }

    #[test]
    fn selects_the_only_file_without_a_tag() {
        let files = vec![file("model-Q8_0.gguf", 100)];
        let chosen = select_file(&spec("org/repo"), &files).expect("single file auto-selects");
        assert_eq!(chosen.filename, "model-Q8_0.gguf");
    }

    #[test]
    fn multiple_files_without_a_tag_is_an_error_not_a_guess() {
        let err = select_file(&spec("org/repo"), &ternary_bonsai()).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("add :<quant>"),
            "error should teach the tag syntax: {message}"
        );
        assert!(
            message.contains("6 GGUF files"),
            "mmproj companions must not count toward the choice: {message}"
        );
    }

    #[test]
    fn recognized_tag_prefers_the_exact_quant_label() {
        let chosen =
            select_file(&spec("org/repo:Q4_1"), &ternary_bonsai()).expect("unique Q4_1 row");
        assert_eq!(chosen.filename, "Ternary-Bonsai-27B-dspark-Q4_1.gguf");
    }

    #[test]
    fn tags_match_on_token_boundaries_only() {
        // :F16 must not match BF16 (or the excluded mmproj-BF16).
        let chosen = select_file(&spec("org/repo:F16"), &ternary_bonsai()).expect("unique F16");
        assert_eq!(chosen.filename, "Ternary-Bonsai-27B-F16.gguf");
        // :Q2_0 must not match PQ2_0.
        let chosen = select_file(&spec("org/repo:Q2_0"), &ternary_bonsai()).expect("unique Q2_0");
        assert_eq!(chosen.filename, "Ternary-Bonsai-27B-Q2_0.gguf");
        // Unrecognized labels are still addressable, case-insensitively.
        let chosen =
            select_file(&spec("org/repo:q2_g64"), &ternary_bonsai()).expect("unique Q2_g64");
        assert_eq!(chosen.filename, "Ternary-Bonsai-27B-Q2_g64.gguf");
    }

    #[test]
    fn ambiguous_tag_is_an_error() {
        let err = select_file(&spec("org/repo:Q2"), &ternary_bonsai()).unwrap_err();
        assert!(
            err.to_string().contains("matches several files"),
            "got: {err}"
        );
    }

    #[test]
    fn unmatched_tag_lists_the_choices() {
        let err = select_file(&spec("org/repo:Q4_K_M"), &ternary_bonsai()).unwrap_err();
        assert!(err.to_string().contains("no file"), "got: {err}");
    }

    #[test]
    fn mmproj_only_and_shard_only_repos_fail_closed() {
        let mmproj_only = vec![file("model-mmproj-F16.gguf", 10)];
        let err = select_file(&spec("org/repo"), &mmproj_only).unwrap_err();
        assert!(err.to_string().contains("mmproj"), "got: {err}");

        let shards = vec![
            file("model-Q4_0-00001-of-00002.gguf", 10),
            file("model-Q4_0-00002-of-00002.gguf", 10),
        ];
        let err = select_file(&spec("org/repo"), &shards).unwrap_err();
        assert!(err.to_string().contains("multi-part"), "got: {err}");
    }

    #[test]
    fn shard_detection_requires_the_full_of_pattern() {
        assert!(is_multipart_shard("m-00001-of-00003.gguf"));
        assert!(is_multipart_shard("m-1-of-2.GGUF"));
        assert!(!is_multipart_shard("m-Q4_0.gguf"));
        assert!(!is_multipart_shard("best-of-nine.gguf"));
        assert!(!is_multipart_shard("m-of-2.gguf"));
    }

    #[test]
    fn tag_matcher_is_boundary_and_case_aware() {
        assert!(tag_matches("Model-Q2_0.gguf", "Q2_0"));
        assert!(tag_matches("model-q2_g64.gguf", "Q2_G64"));
        assert!(!tag_matches("Model-PQ2_0.gguf", "Q2_0"));
        assert!(!tag_matches("Model-BF16.gguf", "F16"));
        assert!(tag_matches("Model-BF16.gguf", "BF16"));
        // `_` is a boundary, so a shorter tag stays ambiguous rather than wrong.
        assert!(tag_matches("Model-Q4_K_M.gguf", "Q4_K"));
    }

    /// The standard layout of the largest GGUF publisher: plain labels next to
    /// `_L` / `Q4_0_x_y` superstring variants. The review round found the
    /// original substring labeler collapsed these (Q6_K_L → "Q6_K"), making
    /// plain tags ambiguous and superstring-only repos silently substitutable.
    fn bartowski_layout() -> Vec<HfGgufFile> {
        vec![
            file("Llama-3.2-1B-Instruct-Q4_0.gguf", 100),
            file("Llama-3.2-1B-Instruct-Q4_0_4_4.gguf", 100),
            file("Llama-3.2-1B-Instruct-Q4_0_4_8.gguf", 100),
            file("Llama-3.2-1B-Instruct-Q4_0_8_8.gguf", 100),
            file("Llama-3.2-1B-Instruct-Q6_K.gguf", 100),
            file("Llama-3.2-1B-Instruct-Q6_K_L.gguf", 100),
            file("Llama-3.2-1B-Instruct-Q4_K_M.gguf", 100),
        ]
    }

    #[test]
    fn exact_label_selects_even_with_superstring_variants() {
        let files = bartowski_layout();
        for (tag, want) in [
            ("Q6_K", "Llama-3.2-1B-Instruct-Q6_K.gguf"),
            ("Q6_K_L", "Llama-3.2-1B-Instruct-Q6_K_L.gguf"),
            ("Q4_0", "Llama-3.2-1B-Instruct-Q4_0.gguf"),
            ("Q4_0_8_8", "Llama-3.2-1B-Instruct-Q4_0_8_8.gguf"),
        ] {
            let chosen = select_file(&spec(&format!("org/repo:{tag}")), &files)
                .unwrap_or_else(|err| panic!(":{tag} should select {want}: {err}"));
            assert_eq!(chosen.filename, want, "for tag :{tag}");
        }
    }

    #[test]
    fn recognized_absent_tag_errors_instead_of_substituting() {
        // :Q4_K in a repo shipping only Q4_K_M must never silently download
        // the different quantization the user did not name.
        let only_m = vec![file("model-Q4_K_M.gguf", 100)];
        let err = select_file(&spec("org/repo:Q4_K"), &only_m).unwrap_err();
        assert!(err.to_string().contains("has quant"), "got: {err}");

        let only_l = vec![file("model-Q6_K_L.gguf", 100)];
        let err = select_file(&spec("org/repo:Q6_K"), &only_l).unwrap_err();
        assert!(err.to_string().contains("has quant"), "got: {err}");
    }

    #[test]
    fn every_listed_tag_selects_exactly_its_file() {
        // The listing is the CLI's own suggestion surface: a printed tag that
        // errors (or picks a different file) is a dead end. Invariant-check the
        // fixtures, including a duplicate-quant pair that forces the full-stem
        // fallback.
        let dup_quants = vec![
            file("model-alpha-Q8_0.gguf", 100),
            file("model-beta-Q8_0.gguf", 100),
        ];
        for files in [ternary_bonsai(), bartowski_layout(), dup_quants] {
            let candidates: Vec<&HfGgufFile> = files
                .iter()
                .filter(|f| !is_mmproj(&f.filename) && !is_multipart_shard(&f.filename))
                .collect();
            for (tag, file) in listing_tags(&candidates) {
                let chosen =
                    select_file(&spec(&format!("org/repo:{tag}")), &files).unwrap_or_else(|err| {
                        panic!("listed tag :{tag} must select {}: {err}", file.filename)
                    });
                assert_eq!(chosen.filename, file.filename, "for listed tag :{tag}");
            }
        }
    }

    #[test]
    fn multibyte_specs_error_instead_of_panicking() {
        // A multi-byte char straddling a URL-prefix byte length used to panic
        // on a non-char-boundary slice (found by the security review lens).
        for bad in ["hf.coü/x", "huggingface.c€/org/repo", "hf.c€xyz/a"] {
            assert!(
                HfModelSpec::parse(bad).is_err(),
                "expected {bad:?} to be a clean parse error"
            );
        }
    }

    #[test]
    fn dest_classification_never_overwrites() {
        assert_eq!(classify_dest(None, 100), DestState::Missing);
        assert_eq!(classify_dest(Some(0), 100), DestState::ReplaceEmpty);
        assert_eq!(classify_dest(Some(100), 100), DestState::Complete);
        assert_eq!(classify_dest(Some(7), 0), DestState::KeepUnknownSize);
        assert_eq!(classify_dest(Some(0), 0), DestState::ReplaceEmpty);
        assert_eq!(classify_dest(Some(7), 100), DestState::Conflict { have: 7 });
        assert_eq!(
            classify_dest(Some(200), 100),
            DestState::Conflict { have: 200 }
        );
    }

    #[test]
    fn ceiling_parse_falls_back_on_garbage_and_zero() {
        let default = crate::api::DEFAULT_MAX_DOWNLOAD_BYTES;
        assert_eq!(parse_max_download_bytes(None), default);
        assert_eq!(parse_max_download_bytes(Some("0")), default);
        assert_eq!(parse_max_download_bytes(Some("abc")), default);
        assert_eq!(parse_max_download_bytes(Some("-1")), default);
        assert_eq!(parse_max_download_bytes(Some(" 65536 ")), 65536);
    }

    #[test]
    fn windows_reserved_stems_are_rejected() {
        for reserved in [
            "CON.gguf",
            "con.gguf",
            "NUL.gguf",
            "COM1.gguf",
            "lpt9.gguf",
            "AUX.Q8_0.gguf",
        ] {
            assert!(
                windows_reserved_stem(reserved),
                "{reserved} should be reserved"
            );
        }
        for fine in [
            "CONX.gguf",
            "model-CON.gguf",
            "COM.gguf",
            "COM10.gguf",
            "tinyllama.gguf",
        ] {
            assert!(!windows_reserved_stem(fine), "{fine} should be allowed");
        }
    }

    fn temp_models_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "camelid-hfpull-test-{label}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn download_pre_gates_bail_before_any_bytes_move() {
        let dir = temp_models_dir("gates");

        // Filename gate (defense in depth: the listing filter already skips
        // pathed names, this must hold even if a caller bypasses it).
        let evil = file("e/vil.gguf", 10);
        let err = download(&evil, &dir, 1000).unwrap_err();
        assert!(err.to_string().contains("bare *.gguf"), "got: {err}");

        let reserved = file("CON.gguf", 10);
        let err = download(&reserved, &dir, 1000).unwrap_err();
        assert!(err.to_string().contains("bare *.gguf"), "got: {err}");

        // Ceiling gate.
        let big = file("model-Q8_0.gguf", 2000);
        let err = download(&big, &dir, 1000).unwrap_err();
        assert!(err.to_string().contains("download ceiling"), "got: {err}");

        assert!(!dir.exists(), "pre-gates must not create the models dir");
    }

    #[test]
    fn download_keeps_complete_and_refuses_conflicting_dest() {
        let dir = temp_models_dir("dest");
        std::fs::create_dir_all(&dir).expect("create temp models dir");
        let dest = dir.join("model-Q8_0.gguf");

        // Size-verified skip: no curl involved.
        std::fs::write(&dest, b"12345").expect("seed dest");
        let f = file("model-Q8_0.gguf", 5);
        let got = download(&f, &dir, 1000).expect("complete file is kept");
        assert_eq!(got, dest);

        // Same filename, different bytes: error, never overwrite.
        let conflicting = file("model-Q8_0.gguf", 9);
        let err = download(&conflicting, &dir, 1000).unwrap_err();
        assert!(err.to_string().contains("already exists"), "got: {err}");
        assert_eq!(
            std::fs::read(&dest).expect("dest still readable"),
            b"12345",
            "conflicting pull must not touch the existing file"
        );

        // Unknown upstream size keeps the local copy.
        let unknown = file("model-Q8_0.gguf", 0);
        let got = download(&unknown, &dir, 1000).expect("unknown size keeps local copy");
        assert_eq!(got, dest);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn suggested_tags_cover_unrecognized_quants() {
        assert_eq!(suggested_tag(&file("m-Q4_K_M.gguf", 1)), "Q4_K_M");
        assert_eq!(
            suggested_tag(&file("Ternary-Bonsai-27B-PQ2_0.gguf", 1)),
            "PQ2_0"
        );
        assert_eq!(
            suggested_tag(&file("Ternary-Bonsai-27B-Q2_g64.gguf", 1)),
            "Q2_G64"
        );
    }
}
