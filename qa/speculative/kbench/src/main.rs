// K-quant GEMV microbench: extracts LINEAR_ROW_SHADER from the Camelid worktree at
// RUNTIME, compiles it, and times the single-token + multi-column K-quant GEMVs at
// production shapes. Also cross-checks the mc kernel against k single-token
// dispatches bit-for-bit, so a kernel edit that breaks exactness fails HERE first.
//
// Usage: kbench [q4k|q5k|q6k|q4kv2|q6kv2|q4kmma|q6kmma|q4kv4|q6kv4|q4kv4w|q6kv4w
//                |q4kreg2v4|q6kafragv4|q4kreg2sk4v4|q6kafragsk4v4|...]
//               [--rows N] [--nsb N] [--iters N] [--seed N] [--sane]
//
// Cases with an `oracle` compare the candidate's k=1..=8 output words (as u32
// bits) against that reference kernel over the same zero-padded staging the
// production route uses; KBENCH_REPS (default 8) sets the dispatches per
// command buffer so per-dispatch launch cost can be amortized like production.
// The production register-exact pair is q4kreg2v4 / q6kafragv4 and their
// split-K twins (CAMELID_KQUANT_V4_REGISTER_EXACT_SPLITK) are q4kreg2sk4v4 /
// q6kafragsk4v4.
use metal::*;
use std::cell::Cell;
use std::path::PathBuf;
use std::time::Instant;

fn metal_source() -> PathBuf {
    std::env::var_os("CAMELID_METAL_SOURCE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../src/metal.rs"))
}

fn repo_root() -> PathBuf {
    metal_source()
        .parent()
        .and_then(|p| p.parent())
        .expect("metal.rs must be under <repo>/src")
        .to_path_buf()
}

fn slice_shader(src: &str) -> String {
    let start_tag = "const LINEAR_ROW_SHADER: &str = r#\"";
    let s = src.find(start_tag).expect("shader start") + start_tag.len();
    let e = src[s..].find("\"#;").expect("shader end") + s;
    src[s..e].to_string()
}

fn extract_shader() -> String {
    slice_shader(&std::fs::read_to_string(metal_source()).expect("read metal.rs"))
}

fn extract_v2_shader() -> String {
    let src = std::fs::read_to_string(metal_source()).expect("read metal.rs");
    let start_tag = "const KQUANT_V2_SHADER: &str = r#\"";
    let s = src.find(start_tag).expect("v2 shader start") + start_tag.len();
    let e = src[s..].find("\"#;").expect("v2 shader end") + s;
    let mut text = src[s..e].to_string();
    // KBENCH_EXTRA_METAL=<file>: benchmark-only kernels appended to the strict
    // v2 library (same compile options), so candidates can be timed before
    // they enter metal.rs.
    if let Some(extra) = std::env::var_os("KBENCH_EXTRA_METAL").filter(|v| !v.is_empty()) {
        text.push('\n');
        text.push_str(&std::fs::read_to_string(&extra).expect("read KBENCH_EXTRA_METAL"));
    }
    text
}

fn env_str(name: &str) -> Option<&'static str> {
    std::env::var(name).ok().map(|v| &*Box::leak(v.into_boxed_str()))
}

fn extract_v3_shader() -> String {
    let src = std::fs::read_to_string(metal_source()).expect("read metal.rs");
    let start_tag = "const KQUANT_V3_SHADER: &str = r#\"";
    let s = src.find(start_tag).expect("v3 shader start") + start_tag.len();
    let e = src[s..].find("\"#;").expect("v3 shader end") + s;
    src[s..e].to_string()
}

/// The pristine branch-HEAD shader: the certified reference the edited kernels
/// must stay bit-identical to.
fn extract_ref_shader() -> String {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root())
        .args(["show", "HEAD:src/metal.rs"])
        .output()
        .expect("git show");
    if out.status.success() {
        slice_shader(&String::from_utf8(out.stdout).expect("utf8"))
    } else {
        // Exported benchmark hosts intentionally carry no .git directory.
        // v2 is checked against its own single-token kernel, so the current
        // shader is the correct fallback there.
        slice_shader(&std::fs::read_to_string(metal_source()).expect("read metal.rs"))
    }
}

struct Case {
    name: &'static str,
    block_bytes: usize,
    scratch_ints_per_sb: usize,
    single: &'static str,
    mc: &'static str,
    tiled: &'static str,
    /// Output rows covered by one `mc` threadgroup (grid = ceil(rows / this)).
    mc_rows_per_tg: usize,
    /// Reference kernel (same ABI family as `mc`) whose k=1..=8 output words the
    /// candidate must reproduce bit-for-bit. For v3 cases the oracle is the
    /// single-column kernel dispatched k times.
    oracle: Option<&'static str>,
    /// Threads per `mc` threadgroup (32 = one SIMD group; split-K kernels use more).
    mc_threads: u64,
}

/// splitmix64 finaliser: seed-dependent deterministic test data.
fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// f32 -> IEEE half bits, round-to-nearest-even (normal range only).
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 31) & 1) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    assert!((1..31).contains(&exp), "f16_bits: value {v} outside the normal half range");
    let mut h = (sign << 15) | ((exp as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(|s| s.as_str()).unwrap_or("q4k");
    let getn = |flag: &str, dflt: usize| -> usize {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(dflt)
    };
    // Default shape: Qwen3-4B ffn_up (hidden 2560 -> ffn 9728): rows=9728, n_sb=10.
    let rows = getn("--rows", 9728);
    let n_sb = getn("--nsb", 10);
    let iters = getn("--iters", 30);

    let device = Device::system_default().expect("metal device");
    if args.iter().any(|a| a == "--check") {
        // Compile the strict v2 library (plus KBENCH_EXTRA_METAL) and exit:
        // a fast syntax gate before a long timing batch.
        let strict = CompileOptions::new();
        strict.set_fast_math_enabled(false);
        match device.new_library_with_source(&extract_v2_shader(), &strict) {
            Ok(_) => {
                println!("kbench --check: v2 shader (+extra) compiles");
                return;
            }
            Err(err) => {
                eprintln!("kbench --check: COMPILE ERROR\n{err}");
                std::process::exit(3);
            }
        }
    }
    let queue = device.new_command_queue();
    let options = CompileOptions::new();
    let lib = device
        .new_library_with_source(&extract_shader(), &options)
        .expect("shader compiles");
    let ref_lib = device
        .new_library_with_source(&extract_ref_shader(), &options)
        .expect("ref shader compiles");

    let is_v4_wide = which == "q4kv4w" || which == "q6kv4w";
    let is_v4 = which.ends_with("v4") || is_v4_wide;
    let is_mma = which == "q4kmma" || which == "q6kmma" || is_v4;
    let is_mma_q6 = is_mma && which.starts_with("q6k");
    let is_v3 = which.ends_with("v3");
    let seed = getn("--seed", 0) as u64;
    // --sane: rewrite every block's f16 `d` into a finite, small positive value
    // so no row is masked by NaN/Inf and every word is a live comparison.
    let sane_d = args.iter().any(|a| a == "--sane");
    let is_v2 = which.ends_with("v2") || is_mma;
    let case = match which {
        "q4k" => Case {
            name: "q4k",
            block_bytes: 144,
            scratch_ints_per_sb: 9,
            single: "q4k_linear_simd",
            mc: "q4k_linear_simd_mc",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 1,
            oracle: None,
            mc_threads: 32,
        },
        "q5k" => Case {
            name: "q5k",
            block_bytes: 176,
            scratch_ints_per_sb: 9,
            single: "q5k_linear_simd",
            mc: "q5k_linear_simd_mc",
            tiled: "q5k_linear_tiled",
            mc_rows_per_tg: 1,
            oracle: None,
            mc_threads: 32,
        },
        "q6k" => Case {
            name: "q6k",
            block_bytes: 210,
            scratch_ints_per_sb: 8,
            single: "q6k_linear_simd",
            mc: "q6k_linear_simd_mc",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 1,
            oracle: None,
            mc_threads: 32,
        },
        "q4kv2" => Case {
            name: "q4kv2",
            block_bytes: 144,
            scratch_ints_per_sb: 9,
            single: "q4k_linear_simd_v2",
            mc: "q4k_linear_simd_mc_v2",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 1,
            oracle: None,
            mc_threads: 32,
        },
        "q4kmma" => Case {
            name: "q4kmma",
            block_bytes: 144,
            scratch_ints_per_sb: 9,
            single: "q4k_linear_simd_v2",
            mc: "q4k_linear_mma_mc_v2",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 32,
        },
        "q6kmma" => Case {
            name: "q6kmma",
            block_bytes: 210,
            scratch_ints_per_sb: 8,
            single: "q6k_linear_simd_v2",
            mc: "q6k_linear_mma_mc_v2",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 32,
        },
        "q4kv4" => Case {
            name: "q4kv4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_mma_combined_v4",
            mc: "q4k_linear_mma_combined_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 32,
        },
        "q6kv4" => Case {
            name: "q6kv4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_mma_combined_v4",
            mc: "q6k_linear_mma_combined_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 32,
        },
        "q4kv4w" => Case {
            name: "q4kv4w",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_mma_combined_v4",
            mc: "q4k_linear_mma_combined_w16_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 64,
        },
        "q6kv4w" => Case {
            name: "q6kv4w",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_mma_combined_v4",
            mc: "q6k_linear_mma_combined_w16_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 64,
        },
        "q5kv2" => Case {
            name: "q5kv2",
            block_bytes: 176,
            scratch_ints_per_sb: 9,
            single: "q5k_linear_simd_v2",
            mc: "q5k_linear_simd_mc_v2",
            tiled: "q5k_linear_tiled",
            mc_rows_per_tg: 1,
            oracle: None,
            mc_threads: 32,
        },
        "q6kv2" => Case {
            name: "q6kv2",
            block_bytes: 210,
            scratch_ints_per_sb: 8,
            single: "q6k_linear_simd_v2",
            mc: "q6k_linear_simd_mc_v2",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 1,
            oracle: None,
            mc_threads: 32,
        },
        "q4kv3" => Case {
            name: "q4kv3",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_f32_v3",
            mc: "q4k_linear_f32_mc_v3",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 4,
            oracle: None,
            mc_threads: 32,
        },
        "q6kv3" => Case {
            name: "q6kv3",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_f32_v3",
            mc: "q6k_linear_f32_mc_v3",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 4,
            oracle: None,
            mc_threads: 32,
        },
        "q6kafragv4" => Case {
            name: "q6kafragv4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_mma_combined_afrag_v4",
            mc: "q6k_linear_mma_combined_afrag_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: Some("q6k_linear_mma_combined_direct_v4"),
            mc_threads: 32,
        },
        "q6kafrag2v4" => Case {
            name: "q6kafrag2v4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_mma_combined_afrag2_v4",
            mc: "q6k_linear_mma_combined_afrag2_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: Some("q6k_linear_mma_combined_direct_v4"),
            mc_threads: 32,
        },
        "q6kdirectv4" => Case {
            name: "q6kdirectv4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_mma_combined_direct_v4",
            mc: "q6k_linear_mma_combined_direct_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: Some("q6k_linear_mma_combined_v4"),
            mc_threads: 32,
        },
        "q6kr2v3" => Case {
            name: "q6kr2v3",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_f32_v3",
            mc: "q6k_linear_f32_mc_reg2_v3",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 4,
            oracle: Some("q6k_linear_f32_v3"),
            mc_threads: 32,
        },
        "q6kr4v3" => Case {
            name: "q6kr4v3",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_f32_v3",
            mc: "q6k_linear_f32_mc_reg4_v3",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: Some("q6k_linear_f32_v3"),
            mc_threads: 32,
        },
        "q6kdiagmmav4" => Case {
            name: "q6kdiagmmav4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_afrag_diag_mma_v4",
            mc: "q6k_afrag_diag_mma_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 32,
        },
        "q6kdiagdecv4" => Case {
            name: "q6kdiagdecv4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_afrag_diag_decode_v4",
            mc: "q6k_afrag_diag_decode_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 32,
        },
        "q6kdiagloadv4" => Case {
            name: "q6kdiagloadv4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_afrag_diag_load_v4",
            mc: "q6k_afrag_diag_load_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 32,
        },
        "q6kafragbv4" => Case {
            name: "q6kafragbv4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_mma_combined_afragb_v4",
            mc: "q6k_linear_mma_combined_afragb_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: Some("q6k_linear_mma_combined_direct_v4"),
            mc_threads: 32,
        },
        "q6kdiagmmaonlyv4" => Case {
            name: "q6kdiagmmaonlyv4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_afrag_diag_mmaonly_v4",
            mc: "q6k_afrag_diag_mmaonly_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 32,
        },
        "q6kdiagbloadv4" => Case {
            name: "q6kdiagbloadv4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_afrag_diag_bload_v4",
            mc: "q6k_afrag_diag_bload_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 32,
        },
        "q6kstgv4" => Case {
            name: "q6kstgv4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_mma_combined_stg_v4",
            mc: "q6k_linear_mma_combined_stg_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: Some("q6k_linear_mma_combined_direct_v4"),
            mc_threads: 32,
        },
        "q6kstg2v4" => Case {
            name: "q6kstg2v4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_mma_combined_stg2_v4",
            mc: "q6k_linear_mma_combined_stg2_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: Some("q6k_linear_mma_combined_direct_v4"),
            mc_threads: 32,
        },
        "q6kafragpv4" => Case {
            name: "q6kafragpv4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_mma_combined_afragp_v4",
            mc: "q6k_linear_mma_combined_afragp_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: Some("q6k_linear_mma_combined_direct_v4"),
            mc_threads: 32,
        },
        "q6kafragp2v4" => Case {
            name: "q6kafragp2v4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_mma_combined_afragp2_v4",
            mc: "q6k_linear_mma_combined_afragp2_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: Some("q6k_linear_mma_combined_direct_v4"),
            mc_threads: 32,
        },
        "q6kdiagload8v4" => Case {
            name: "q6kdiagload8v4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_afrag_diag_load8_v4",
            mc: "q6k_afrag_diag_load8_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 32,
        },
        "q6kdiagload32v4" => Case {
            name: "q6kdiagload32v4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_afrag_diag_load32_v4",
            mc: "q6k_afrag_diag_load32_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: None,
            mc_threads: 32,
        },
        "q4kreg2v4" => Case {
            name: "q4kreg2v4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_mma_combined_reg2_v4",
            mc: "q4k_linear_mma_combined_reg2_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: Some("q4k_linear_mma_combined_v4"),
            mc_threads: 32,
        },
        "q4kdirectv4" => Case {
            name: "q4kdirectv4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_mma_combined_direct_v4",
            mc: "q4k_linear_mma_combined_direct_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: Some("q4k_linear_mma_combined_v4"),
            mc_threads: 32,
        },
        "q4kreg2sk2v4" => Case {
            name: "q4kreg2sk2v4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_mma_combined_reg2_sk2_v4",
            mc: "q4k_linear_mma_combined_reg2_sk2_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: Some("q4k_linear_mma_combined_reg2_v4"),
            mc_threads: 64,
        },
        "q4kreg2sk4v4" => Case {
            name: "q4kreg2sk4v4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_mma_combined_reg2_sk4_v4",
            mc: "q4k_linear_mma_combined_reg2_sk4_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: Some("q4k_linear_mma_combined_reg2_v4"),
            mc_threads: 128,
        },
        "q4kreg1sk4v4" => Case {
            name: "q4kreg1sk4v4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_mma_combined_reg1_sk4_v4",
            mc: "q4k_linear_mma_combined_reg1_sk4_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: Some("q4k_linear_mma_combined_reg2_v4"),
            mc_threads: 128,
        },
        "q4kreg1sk2v4" => Case {
            name: "q4kreg1sk2v4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_mma_combined_reg1_sk2_v4",
            mc: "q4k_linear_mma_combined_reg1_sk2_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: Some("q4k_linear_mma_combined_reg2_v4"),
            mc_threads: 64,
        },
        "q6kafragsk2v4" => Case {
            name: "q6kafragsk2v4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_mma_combined_afrag_sk2_v4",
            mc: "q6k_linear_mma_combined_afrag_sk2_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: Some("q6k_linear_mma_combined_afrag_v4"),
            mc_threads: 64,
        },
        "q6kafragsk4v4" => Case {
            name: "q6kafragsk4v4",
            block_bytes: 210,
            scratch_ints_per_sb: 0,
            single: "q6k_linear_mma_combined_afrag_sk4_v4",
            mc: "q6k_linear_mma_combined_afrag_sk4_v4",
            tiled: "q6k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: Some("q6k_linear_mma_combined_afrag_v4"),
            mc_threads: 128,
        },
        "q4kdiagloadv4" => Case {
            name: "q4kdiagloadv4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_reg2_diag_load_v4",
            mc: "q4k_reg2_diag_load_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: None,
            mc_threads: 32,
        },
        "q4kdiagload16v4" => Case {
            name: "q4kdiagload16v4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_reg2_diag_load16_v4",
            mc: "q4k_reg2_diag_load16_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: None,
            mc_threads: 32,
        },
        "q4kdiagmmav4" => Case {
            name: "q4kdiagmmav4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_reg2_diag_mma_v4",
            mc: "q4k_reg2_diag_mma_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: None,
            mc_threads: 32,
        },
        "q4kdiagbmmav4" => Case {
            name: "q4kdiagbmmav4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_reg2_diag_bmma_v4",
            mc: "q4k_reg2_diag_bmma_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: None,
            mc_threads: 32,
        },
        "q4kreg2pv4" => Case {
            name: "q4kreg2pv4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_mma_combined_reg2p_v4",
            mc: "q4k_linear_mma_combined_reg2p_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: Some("q4k_linear_mma_combined_reg2_v4"),
            mc_threads: 32,
        },
        "q4kreg4pv4" => Case {
            name: "q4kreg4pv4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_mma_combined_reg4p_v4",
            mc: "q4k_linear_mma_combined_reg4p_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 32,
            oracle: Some("q4k_linear_mma_combined_reg2_v4"),
            mc_threads: 32,
        },
        "q4kreg4v4" => Case {
            name: "q4kreg4v4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_mma_combined_reg4_v4",
            mc: "q4k_linear_mma_combined_reg4_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 32,
            oracle: Some("q4k_linear_mma_combined_reg2_v4"),
            mc_threads: 32,
        },
        "q4kreg1v4" => Case {
            name: "q4kreg1v4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_linear_mma_combined_reg1_v4",
            mc: "q4k_linear_mma_combined_reg1_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 8,
            oracle: Some("q4k_linear_mma_combined_reg2_v4"),
            mc_threads: 32,
        },
        "q4kdiagmmaonlyv4" => Case {
            name: "q4kdiagmmaonlyv4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_reg2_diag_mmaonly_v4",
            mc: "q4k_reg2_diag_mmaonly_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: None,
            mc_threads: 32,
        },
        "q4kdiagnullv4" => Case {
            name: "q4kdiagnullv4",
            block_bytes: 144,
            scratch_ints_per_sb: 0,
            single: "q4k_reg2_diag_null_v4",
            mc: "q4k_reg2_diag_null_v4",
            tiled: "q4k_linear_tiled",
            mc_rows_per_tg: 16,
            oracle: None,
            mc_threads: 32,
        },
        // Generic register-exact-ABI candidate: KBENCH_CASE_MC (kernel name),
        // KBENCH_CASE_ORACLE (reference kernel, optional), KBENCH_CASE_ROWS_PER_TG
        // (default 16 for Q4, 8 for Q6), KBENCH_CASE_THREADS (default 32).
        "q4kcustomv4" | "q6kcustomv4" => {
            let q6 = which == "q6kcustomv4";
            Case {
                name: if q6 { "q6kcustomv4" } else { "q4kcustomv4" },
                block_bytes: if q6 { 210 } else { 144 },
                scratch_ints_per_sb: 0,
                single: env_str("KBENCH_CASE_MC").expect("KBENCH_CASE_MC"),
                mc: env_str("KBENCH_CASE_MC").expect("KBENCH_CASE_MC"),
                tiled: if q6 { "q6k_linear_tiled" } else { "q4k_linear_tiled" },
                mc_rows_per_tg: std::env::var("KBENCH_CASE_ROWS_PER_TG")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(if q6 { 8 } else { 16 }),
                oracle: env_str("KBENCH_CASE_ORACLE"),
                mc_threads: std::env::var("KBENCH_CASE_THREADS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(32),
            }
        }
        other => panic!("unknown case {other}"),
    };
    let v2_lib = if is_v2 {
        let strict = CompileOptions::new();
        strict.set_fast_math_enabled(false);
        Some(
            device
                .new_library_with_source(&extract_v2_shader(), &strict)
                .expect("v2 shader compiles"),
        )
    } else {
        None
    };
    let v3_lib = if is_v3 {
        let v3_options = CompileOptions::new();
        v3_options.set_fast_math_enabled(false);
        Some(
            device
                .new_library_with_source(&extract_v3_shader(), &v3_options)
                .expect("v3 shader compiles"),
        )
    } else {
        None
    };
    let pipe = |name: &str| {
        let f = lib.get_function(name, None).expect(name);
        device
            .new_compute_pipeline_state_with_function(&f)
            .expect(name)
    };
    let v2_pipe = |name: &str| {
        let f = v2_lib
            .as_ref()
            .unwrap()
            .get_function(name, None)
            .expect(name);
        device
            .new_compute_pipeline_state_with_function(&f)
            .expect(name)
    };
    let (single, mc) = if is_v3 {
        let f_single = v3_lib
            .as_ref()
            .unwrap()
            .get_function(case.single, None)
            .expect(case.single);
        let f_mc = v3_lib
            .as_ref()
            .unwrap()
            .get_function(case.mc, None)
            .expect(case.mc);
        let p_single = device
            .new_compute_pipeline_state_with_function(&f_single)
            .expect(case.single);
        let p_mc = device
            .new_compute_pipeline_state_with_function(&f_mc)
            .expect(case.mc);
        (p_single, p_mc)
    } else if is_v2 {
        (v2_pipe(case.single), v2_pipe(case.mc))
    } else {
        (pipe(case.single), pipe(case.mc))
    };
    // v2 is its own bit-universe: the contract is single_v2 == mc_v2 (checked
    // below), not equality with the HEAD kernel.
    let ref_single = if is_v2 || is_v3 {
        single.clone()
    } else {
        let f = ref_lib.get_function(case.single, None).expect("ref single");
        device
            .new_compute_pipeline_state_with_function(&f)
            .expect("ref single pipe")
    };
    let _ = case.tiled;
    let oracle_pipe = case.oracle.map(|name| {
        if is_v3 {
            let f = v3_lib.as_ref().unwrap().get_function(name, None).expect(name);
            device.new_compute_pipeline_state_with_function(&f).expect(name)
        } else {
            v2_pipe(name)
        }
    });
    println!(
        "{} pipelines: mc maxTotalThreadsPerThreadgroup={} (registers pressure proxy) | seed={} sane_d={}",
        case.name,
        mc.max_total_threads_per_threadgroup(),
        seed,
        sane_d
    );
    let quantize = pipe("quantize_q8k_rows");

    let max_k = 16usize;
    let cols = n_sb * 256;

    // Deterministic activations for max_k columns.
    let mut y_flat = vec![0.0f32; max_k * cols];
    for (i, y) in y_flat.iter_mut().enumerate() {
        let t = i / cols;
        let c = i % cols;
        *y = if seed == 0 {
            ((((t * 131 + c * 17) % 251) as f32) - 125.0) * 0.017 + t as f32 * 0.0011
        } else {
            let h = mix64((i as u64) ^ (seed << 40) ^ 0xA5A5);
            (((h & 0xffff) as f32) / 65535.0 - 0.5) * 4.2 + t as f32 * 0.0011
        };
    }
    let buf_f32 = |data: &[f32]| {
        device.new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let y_buf = buf_f32(&y_flat);
    let scales_buf = device.new_buffer(
        (max_k * n_sb * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let quants_buf =
        device.new_buffer((max_k * cols) as u64, MTLResourceOptions::StorageModeShared);
    let qscalar = device.new_buffer(12, MTLResourceOptions::StorageModeShared);
    unsafe {
        let p = qscalar.contents() as *mut u32;
        *p = n_sb as u32;
        *p.add(1) = rows as u32;
        *p.add(2) = max_k as u32;
    }
    {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(&quantize);
        e.set_buffer(0, Some(&y_buf), 0);
        e.set_buffer(1, Some(&scales_buf), 0);
        e.set_buffer(2, Some(&quants_buf), 0);
        e.set_buffer(3, Some(&qscalar), 0);
        e.set_buffer(4, Some(&qscalar), 8);
        let total = (max_k * n_sb) as u64;
        let w = quantize.thread_execution_width();
        e.dispatch_thread_groups(
            MTLSize {
                width: total.div_ceil(w),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: w,
                height: 1,
                depth: 1,
            },
        );
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Deterministic wire blocks.
    let mut wire = vec![0u8; rows * n_sb * case.block_bytes];
    for (i, b) in wire.iter_mut().enumerate() {
        *b = if seed == 0 {
            ((i * 13 + i / 97 + 5) % 256) as u8
        } else {
            mix64((i as u64) ^ (seed << 40) ^ 0x5A5A) as u8
        };
    }
    if sane_d && case.block_bytes == 210 {
        for blk in 0..rows * n_sb {
            let h = mix64((blk as u64) ^ (seed << 40) ^ 0xD00D);
            let v = 0.002f32 + 0.03f32 * ((h & 0xffff) as f32 / 65535.0);
            let bits = f16_bits(v);
            wire[blk * 210 + 208] = (bits & 0xff) as u8;
            wire[blk * 210 + 209] = (bits >> 8) as u8;
        }
    }
    let w_buf = device.new_buffer_with_data(
        wire.as_ptr() as *const _,
        wire.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_single = device.new_buffer(
        (max_k * rows * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_mc = device.new_buffer(
        (max_k * rows * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v4_single_dispatches = Cell::new(0usize);
    let v4_multi_dispatches = Cell::new(0usize);

    let weight_bytes = (rows * n_sb * case.block_bytes) as f64;

    let scalar_for = |k: usize| {
        let s = device.new_buffer(12, MTLResourceOptions::StorageModeShared);
        unsafe {
            let p = s.contents() as *mut u32;
            *p = n_sb as u32;
            *p.add(1) = rows as u32;
            *p.add(2) = k as u32;
        }
        s
    };

    // Repetitions per command buffer: amortizes the ~0.2 ms commit/wait latency
    // the same way production does (hundreds of dispatches per buffer).
    let reps_env: usize = std::env::var("KBENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    #[allow(non_snake_case)]
    let REPS: usize = reps_env;

    // --- Reference: k single-token dispatches into out_single ------------------
    // Returns amortized ms for ONE set of k dispatches.
    // v2 single: 32 threads (one simdgroup) covering Q4K_V2_ROWS_PER_SG rows.
    let single_tg: u64 = if is_v3 { 64 } else { 32 };
    let rows_per_tg: usize = if is_v3 { 4 } else { 1 };
    let run_single_k = |k: usize, timed_iters: usize| -> f64 {
        let scalar = scalar_for(1);
        let tg_bytes =
            ((case.scratch_ints_per_sb * n_sb * rows_per_tg * 4).next_multiple_of(16)) as u64;
        let mut best = f64::MAX;
        for _ in 0..timed_iters {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            for _rep in 0..REPS {
                for t in 0..k {
                    e.set_compute_pipeline_state(&single);
                    if is_v3 {
                        e.set_buffer(0, Some(&y_buf), (t * cols * 4) as u64);
                    } else {
                        e.set_buffer(0, Some(&scales_buf), (t * n_sb * 4) as u64);
                        e.set_buffer(1, Some(&quants_buf), (t * cols) as u64);
                    }
                    e.set_buffer(2, Some(&w_buf), 0);
                    e.set_buffer(3, Some(&out_single), (t * rows * 4) as u64);
                    e.set_buffer(4, Some(&scalar), 0);
                    e.set_buffer(5, Some(&scalar), 4);
                    if tg_bytes > 0 {
                        e.set_threadgroup_memory_length(0, tg_bytes);
                    }
                    e.dispatch_thread_groups(
                        MTLSize {
                            width: (rows as u64).div_ceil(rows_per_tg as u64),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: single_tg,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
            }
            e.end_encoding();
            let t0 = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64);
        }
        best
    };

    // --- MMA staging pipelines + buffers (q4kmma only) --------------------------
    let mma_aux = if is_mma {
        let stage_y = v2_pipe(if is_mma_q6 && !is_v4 {
            "q6k_mma_stage_y_f32"
        } else {
            "q4k_mma_stage_y"
        });
        let stage_ysums = v2_pipe("q4k_mma_stage_ysums");
        let k_pad_max = 16usize;
        let y_half = device.new_buffer(
            (cols * k_pad_max * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ysums = device.new_buffer(
            (n_sb * 16 * k_pad_max * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Some((stage_y, stage_ysums, y_half, ysums))
    } else {
        None
    };
    // KBENCH_CASE_STAGE_Y / KBENCH_CASE_STAGE_YSUMS: alternate activation
    // staging kernels (same buffer sizes, permuted layout) used for the
    // candidate dispatches only; the oracle keeps the production staging.
    let mma_aux_cand = mma_aux.as_ref().map(|(sy, sys, yh, ys)| {
        let sy2 = env_str("KBENCH_CASE_STAGE_Y").map(|n| v2_pipe(n)).unwrap_or_else(|| sy.clone());
        let sys2 = env_str("KBENCH_CASE_STAGE_YSUMS").map(|n| v2_pipe(n)).unwrap_or_else(|| sys.clone());
        (sy2, sys2, yh.clone(), ys.clone())
    });
    let run_mma = |k: usize, timed_iters: usize| -> f64 {
        let (stage_y, stage_ysums, y_half, ysums) = mma_aux_cand.as_ref().unwrap();
        let k_pad = if is_v4_wide { 16 } else { (k + 7) & !7 };
        let scalar = device.new_buffer(32, MTLResourceOptions::StorageModeShared);
        unsafe {
            let p = scalar.contents() as *mut u32;
            *p = n_sb as u32; // @0  n_sb
            *p.add(1) = rows as u32; // @4  rows
            *p.add(2) = k as u32; // @8  n_tokens
            *p.add(3) = cols as u32; // @12 width
            *p.add(4) = k_pad as u32; // @16 k_pad
        }
        let mut best = f64::MAX;
        for _ in 0..timed_iters {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            for _rep in 0..REPS {
                e.set_compute_pipeline_state(stage_y);
                e.set_buffer(0, Some(&quants_buf), 0);
                e.set_buffer(1, Some(y_half), 0);
                e.set_buffer(2, Some(&scalar), 12);
                e.set_buffer(3, Some(&scalar), 8);
                e.set_buffer(4, Some(&scalar), 16);
                let total = (cols * k_pad) as u64;
                let w = stage_y.thread_execution_width();
                e.dispatch_thread_groups(
                    MTLSize {
                        width: total.div_ceil(w),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: w,
                        height: 1,
                        depth: 1,
                    },
                );
                if !is_mma_q6 {
                    e.set_compute_pipeline_state(stage_ysums);
                    e.set_buffer(0, Some(&quants_buf), 0);
                    e.set_buffer(1, Some(ysums), 0);
                    e.set_buffer(2, Some(&scalar), 0);
                    e.set_buffer(3, Some(&scalar), 8);
                    e.set_buffer(4, Some(&scalar), 16);
                    let total = (n_sb * 16 * k_pad) as u64;
                    let w = stage_ysums.thread_execution_width();
                    e.dispatch_thread_groups(
                        MTLSize {
                            width: total.div_ceil(w),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: w,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                e.set_compute_pipeline_state(&mc);
                e.set_buffer(0, Some(&scales_buf), 0);
                e.set_buffer(2, Some(&w_buf), 0);
                e.set_buffer(3, Some(&out_mc), 0);
                e.set_buffer(4, Some(&scalar), 0);
                e.set_buffer(5, Some(&scalar), 4);
                e.set_buffer(6, Some(&scalar), 8);
                e.set_buffer(7, Some(y_half), 0);
                if !is_mma_q6 {
                    e.set_buffer(8, Some(ysums), 0);
                }
                e.dispatch_thread_groups(
                    MTLSize {
                        width: (rows as u64).div_ceil(case.mc_rows_per_tg as u64),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: case.mc_threads,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            e.end_encoding();
            let t0 = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64);
        }
        if is_v4 {
            let counter = if k == 1 {
                &v4_single_dispatches
            } else {
                &v4_multi_dispatches
            };
            counter.set(counter.get() + REPS * timed_iters);
        }
        best
    };

    // v4's single-column oracle is deliberately the very same padded matrix
    // kernel as its multi-column path.  Stage one source column into an
    // 8-column zero-padded tile, dispatch n_tokens=1, and repeat for each
    // requested column.  That makes the v4 single==multi contract independent
    // of every older scalar arithmetic universe.
    let run_v4_singles = |k: usize, timed_iters: usize| -> f64 {
        assert!(is_v4 && k <= if is_v4_wide { 16 } else { 8 });
        let (stage_y, stage_ysums, y_staged, ysums) = mma_aux_cand.as_ref().unwrap();
        let scalar = device.new_buffer(32, MTLResourceOptions::StorageModeShared);
        unsafe {
            let p = scalar.contents() as *mut u32;
            *p = n_sb as u32; // @0  n_sb
            *p.add(1) = rows as u32; // @4  rows
            *p.add(2) = 1; // @8  n_tokens
            *p.add(3) = cols as u32; // @12 width
            *p.add(4) = 8; // @16 k_pad
        }
        let mut best = f64::MAX;
        for _ in 0..timed_iters {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            for _rep in 0..REPS {
                for t in 0..k {
                    e.set_compute_pipeline_state(stage_y);
                    e.set_buffer(0, Some(&quants_buf), (t * cols) as u64);
                    e.set_buffer(1, Some(y_staged), 0);
                    e.set_buffer(2, Some(&scalar), 12);
                    e.set_buffer(3, Some(&scalar), 8);
                    e.set_buffer(4, Some(&scalar), 16);
                    let total = (cols * 8) as u64;
                    let w = stage_y.thread_execution_width();
                    e.dispatch_thread_groups(
                        MTLSize {
                            width: total.div_ceil(w),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: w,
                            height: 1,
                            depth: 1,
                        },
                    );
                    if !is_mma_q6 {
                        e.set_compute_pipeline_state(stage_ysums);
                        e.set_buffer(0, Some(&quants_buf), (t * cols) as u64);
                        e.set_buffer(1, Some(ysums), 0);
                        e.set_buffer(2, Some(&scalar), 0);
                        e.set_buffer(3, Some(&scalar), 8);
                        e.set_buffer(4, Some(&scalar), 16);
                        let total = (n_sb * 16 * 8) as u64;
                        let w = stage_ysums.thread_execution_width();
                        e.dispatch_thread_groups(
                            MTLSize {
                                width: total.div_ceil(w),
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: w,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    e.set_compute_pipeline_state(&single);
                    e.set_buffer(0, Some(&scales_buf), (t * n_sb * 4) as u64);
                    e.set_buffer(2, Some(&w_buf), 0);
                    e.set_buffer(3, Some(&out_single), (t * rows * 4) as u64);
                    e.set_buffer(4, Some(&scalar), 0);
                    e.set_buffer(5, Some(&scalar), 4);
                    e.set_buffer(6, Some(&scalar), 8);
                    e.set_buffer(7, Some(y_staged), 0);
                    if !is_mma_q6 {
                        e.set_buffer(8, Some(ysums), 0);
                    }
                    e.dispatch_thread_groups(
                        MTLSize {
                            width: (rows as u64).div_ceil(case.mc_rows_per_tg as u64),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: case.mc_threads,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
            }
            e.end_encoding();
            let t0 = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64);
        }
        v4_single_dispatches.set(v4_single_dispatches.get() + REPS * k * timed_iters);
        best
    };

    // --- mc kernel: one dispatch covering k columns ----------------------------
    let mc_tg: u64 = std::env::var("KBENCH_MC_TG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if is_v3 { 64 } else { 32 });
    let run_mc_scalar = |k: usize, timed_iters: usize| -> f64 {
        let scalar = scalar_for(k);
        let scratch_sb = if is_v2 {
            n_sb.min((1440 / (9 * k)).max(1))
        } else {
            n_sb
        };
        let tg_bytes =
            ((case.scratch_ints_per_sb * scratch_sb * k * 4).next_multiple_of(16)) as u64;
        let mut best = f64::MAX;
        for _ in 0..timed_iters {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            for _rep in 0..REPS {
                e.set_compute_pipeline_state(&mc);
                if is_v3 {
                    e.set_buffer(0, Some(&y_buf), 0);
                } else {
                    e.set_buffer(0, Some(&scales_buf), 0);
                    e.set_buffer(1, Some(&quants_buf), 0);
                }
                e.set_buffer(2, Some(&w_buf), 0);
                e.set_buffer(3, Some(&out_mc), 0);
                e.set_buffer(4, Some(&scalar), 0);
                e.set_buffer(5, Some(&scalar), 4);
                e.set_buffer(6, Some(&scalar), 8);
                if tg_bytes > 0 {
                    e.set_threadgroup_memory_length(0, tg_bytes);
                }
                e.dispatch_thread_groups(
                    MTLSize {
                        width: if is_v3 {
                            (rows as u64).div_ceil(case.mc_rows_per_tg as u64)
                        } else {
                            rows as u64
                        },
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: mc_tg,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            e.end_encoding();
            let t0 = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64);
        }
        best
    };

    // Oracle: the pristine branch-HEAD single-token kernel. Any edit to the
    // working-tree simd/mc kernels must still match ITS outputs bit-for-bit.
    let out_ref = device.new_buffer(
        (max_k * rows * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let run_ref = |k: usize| {
        let scalar = scalar_for(1);
        let tg_bytes = ((case.scratch_ints_per_sb * n_sb * 4).next_multiple_of(16)) as u64;
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        for t in 0..k {
            e.set_compute_pipeline_state(&ref_single);
            e.set_buffer(0, Some(&scales_buf), (t * n_sb * 4) as u64);
            e.set_buffer(1, Some(&quants_buf), (t * cols) as u64);
            e.set_buffer(2, Some(&w_buf), 0);
            e.set_buffer(3, Some(&out_ref), (t * rows * 4) as u64);
            e.set_buffer(4, Some(&scalar), 0);
            e.set_buffer(5, Some(&scalar), 4);
            e.set_threadgroup_memory_length(0, tg_bytes);
            e.dispatch_thread_groups(
                MTLSize {
                    width: rows as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    };

    let run_mc = |k: usize, timed_iters: usize| -> f64 {
        if is_mma {
            run_mma(k, timed_iters)
        } else {
            run_mc_scalar(k, timed_iters)
        }
    };

    // ORACLE (MMA ABI): the reference V4-family kernel over the very same staged
    // activation tile (k_pad=8, n_tokens=k), grid ceil(rows/8), 32 threads.
    let run_oracle_mma = |k: usize| {
        let (stage_y, stage_ysums, y_half, ysums) = mma_aux.as_ref().unwrap();
        let oracle = oracle_pipe.as_ref().unwrap();
        let k_pad = 8usize;
        let scalar = device.new_buffer(32, MTLResourceOptions::StorageModeShared);
        unsafe {
            let p = scalar.contents() as *mut u32;
            *p = n_sb as u32;
            *p.add(1) = rows as u32;
            *p.add(2) = k as u32;
            *p.add(3) = cols as u32;
            *p.add(4) = k_pad as u32;
        }
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(stage_y);
        e.set_buffer(0, Some(&quants_buf), 0);
        e.set_buffer(1, Some(y_half), 0);
        e.set_buffer(2, Some(&scalar), 12);
        e.set_buffer(3, Some(&scalar), 8);
        e.set_buffer(4, Some(&scalar), 16);
        let total = (cols * k_pad) as u64;
        let w = stage_y.thread_execution_width();
        e.dispatch_thread_groups(
            MTLSize { width: total.div_ceil(w), height: 1, depth: 1 },
            MTLSize { width: w, height: 1, depth: 1 },
        );
        if !is_mma_q6 {
            // Q4 oracles consume the staged 16-element activation sums too.
            e.set_compute_pipeline_state(stage_ysums);
            e.set_buffer(0, Some(&quants_buf), 0);
            e.set_buffer(1, Some(ysums), 0);
            e.set_buffer(2, Some(&scalar), 0);
            e.set_buffer(3, Some(&scalar), 8);
            e.set_buffer(4, Some(&scalar), 16);
            let total = (n_sb * 16 * k_pad) as u64;
            let w = stage_ysums.thread_execution_width();
            e.dispatch_thread_groups(
                MTLSize { width: total.div_ceil(w), height: 1, depth: 1 },
                MTLSize { width: w, height: 1, depth: 1 },
            );
        }
        e.set_compute_pipeline_state(oracle);
        e.set_buffer(0, Some(&scales_buf), 0);
        e.set_buffer(2, Some(&w_buf), 0);
        e.set_buffer(3, Some(&out_ref), 0);
        e.set_buffer(4, Some(&scalar), 0);
        e.set_buffer(5, Some(&scalar), 4);
        e.set_buffer(6, Some(&scalar), 8);
        e.set_buffer(7, Some(y_half), 0);
        if !is_mma_q6 {
            e.set_buffer(8, Some(ysums), 0);
        }
        e.dispatch_thread_groups(
            MTLSize { width: (rows as u64).div_ceil(8), height: 1, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    };

    let skip_check = std::env::var("KBENCH_SKIP_CHECK").is_ok();
    if !skip_check && oracle_pipe.is_some() {
        // Candidate mc(k) vs ORACLE(k), every output word as u32 bits, k = 1..=8.
        let mut all_ok = true;
        for k in 1..=8usize {
            // Poison both outputs so a word the kernel never writes is caught.
            unsafe {
                std::ptr::write_bytes(out_mc.contents() as *mut u8, 0xEE, max_k * rows * 4);
                std::ptr::write_bytes(out_ref.contents() as *mut u8, 0xDD, max_k * rows * 4);
                std::ptr::write_bytes(out_single.contents() as *mut u8, 0xDD, max_k * rows * 4);
            }
            run_mc(k, 1);
            let o: &[u32] = if is_v3 {
                run_single_k(k, 1);
                unsafe { std::slice::from_raw_parts(out_single.contents() as *const u32, k * rows) }
            } else {
                run_oracle_mma(k);
                unsafe { std::slice::from_raw_parts(out_ref.contents() as *const u32, k * rows) }
            };
            let c = unsafe { std::slice::from_raw_parts(out_mc.contents() as *const u32, k * rows) };
            let nonfinite = o.iter().filter(|w| !f32::from_bits(**w).is_finite()).count();
            let bad: Vec<usize> = (0..k * rows).filter(|&i| o[i] != c[i]).collect();
            if bad.is_empty() {
                println!(
                    "{} ORACLE {} k={} PASS ({} words, {} non-finite oracle words)",
                    case.name, case.oracle.unwrap(), k, k * rows, nonfinite
                );
            } else {
                all_ok = false;
                let i = bad[0];
                println!(
                    "{} ORACLE {} k={} MISMATCH {} of {} words; first row={} col={} oracle={:#010x} ({}) candidate={:#010x} ({})",
                    case.name, case.oracle.unwrap(), k, bad.len(), k * rows,
                    i % rows, i / rows, o[i], f32::from_bits(o[i]), c[i], f32::from_bits(c[i])
                );
            }
        }
        if !all_ok {
            println!("{} ORACLE SUMMARY: MISMATCH", case.name);
            if std::env::var("KBENCH_CONTINUE_ON_MISMATCH").is_err() {
                std::process::exit(2);
            }
        } else {
            println!("{} ORACLE SUMMARY: PASS (k=1..=8, seed {})", case.name, seed);
        }
    }
    // Bit-identity: edited single(k dispatches) vs the HEAD oracle, k = 16 covers all columns.
    // v2 is its own bit-universe AND its own dispatch geometry, so the HEAD-oracle
    // dispatch does not apply there; single_v2 vs mc_v2 below is the real contract.
    if !skip_check && !is_v2 && !is_v3 {
        let k = max_k;
        run_ref(k);
        run_single_k(k, 1);
        let o = unsafe { std::slice::from_raw_parts(out_ref.contents() as *const u32, k * rows) };
        let a =
            unsafe { std::slice::from_raw_parts(out_single.contents() as *const u32, k * rows) };
        let bad = (0..k * rows).filter(|&i| o[i] != a[i]).count();
        if bad > 0 {
            for i in (0..k * rows).filter(|&i| o[i] != a[i]).take(5) {
                let (fo, fa) = (f32::from_bits(o[i]), f32::from_bits(a[i]));
                eprintln!(
                    "  idx {i}: oracle {fo} ({:#x}) vs edited {fa} ({:#x}) rel {:.2e}",
                    o[i],
                    a[i],
                    ((fo - fa) / fo.max(1e-30)).abs()
                );
            }
        }
        assert!(
            bad == 0,
            "{} SINGLE vs HEAD-ORACLE: {} of {} words differ",
            case.name,
            bad,
            k * rows
        );
    }
    if !skip_check {
        let debug = std::env::var("KBENCH_MMA_DEBUG").is_ok();
        let max_check_k = if is_v3 || (is_v4 && !is_v4_wide) {
            8
        } else {
            max_k
        };
        for k in 2..=max_check_k {
            if is_v4 {
                run_v4_singles(k, 1);
            } else {
                run_single_k(k, 1);
            }
            run_mc(k, 1);
            let a = unsafe {
                std::slice::from_raw_parts(out_single.contents() as *const u32, k * rows)
            };
            let b =
                unsafe { std::slice::from_raw_parts(out_mc.contents() as *const u32, k * rows) };
            if debug {
                let bad: Vec<usize> = (0..k * rows).filter(|&i| a[i] != b[i]).collect();
                if !bad.is_empty() {
                    let mut by_rmod = [0usize; 8];
                    let mut by_t = vec![0usize; k];
                    for &i in &bad {
                        let t = i / rows;
                        let r = i % rows;
                        by_rmod[r % 8] += 1;
                        by_t[t] += 1;
                    }
                    println!(
                        "{} k={}: {} of {} mismatch | by r%8 {:?} | by col {:?} | first rows {:?}",
                        case.name,
                        k,
                        bad.len(),
                        k * rows,
                        by_rmod,
                        by_t,
                        bad.iter().take(8).map(|i| i % rows).collect::<Vec<_>>()
                    );
                    for &i in bad.iter().take(3) {
                        println!(
                            "   idx {i}: single {} mma {}",
                            f32::from_bits(a[i]),
                            f32::from_bits(b[i])
                        );
                    }
                } else {
                    println!("{} k={}: CLEAN", case.name, k);
                }
                continue;
            }
            // single writes [t][row] via offset t*rows -- same layout as mc's out[t*rows+row].
            for i in 0..k * rows {
                assert_eq!(a[i], b[i], "{} k={} idx={} MISMATCH", case.name, k, i);
            }
        }
        println!(
            "{} rows={} n_sb={} bit-identity PASS (k=2..={})",
            case.name, rows, n_sb, max_check_k
        );
        if is_v4 {
            if !is_v4_wide {
                assert_eq!(case.single, case.mc, "narrow v4 must use one pipeline name");
            }
            assert!(v4_single_dispatches.get() > 0);
            assert!(v4_multi_dispatches.get() > 0);
            println!(
                "{} structural V4-UNIVERSE PASS (n_tokens=1 dispatches={}, multi dispatches={})",
                case.name,
                v4_single_dispatches.get(),
                v4_multi_dispatches.get()
            );
        }
    } else {
        println!(
            "{} rows={} n_sb={} PERF-ONLY (identity checks SKIPPED)",
            case.name, rows, n_sb
        );
    }

    // Bandwidth probes: same grid (rows x 32 threads / rows x 128), math-free
    // weight streaming — the geometry's achievable GB/s ceiling.
    {
        let probe_src = r#"
#include <metal_stdlib>
using namespace metal;
kernel void probe32(device const uint4* w [[buffer(0)]], device float* out [[buffer(1)]],
                    constant uint& words_per_row [[buffer(2)]], constant uint& rows [[buffer(3)]],
                    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (row >= rows) return;
    device const uint4* p = w + row * words_per_row;
    uint4 acc = uint4(0u);
    for (uint i = lane; i < words_per_row; i += 32) acc += p[i];
    uint s = acc.x + acc.y + acc.z + acc.w;
    s = simd_sum(s);
    if (lane == 0 && s == 0xdeadbeefu) out[row] = 1.0f;
}
kernel void probe128(device const uint4* w [[buffer(0)]], device float* out [[buffer(1)]],
                     constant uint& words_per_row [[buffer(2)]], constant uint& rows [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    if (row >= rows) return;
    device const uint4* p = w + row * words_per_row;
    uint4 acc = uint4(0u);
    for (uint i = tid; i < words_per_row; i += 128) acc += p[i];
    uint s = acc.x + acc.y + acc.z + acc.w;
    s = simd_sum(s);
    if (tid == 0 && s == 0xdeadbeefu) out[row] = 1.0f;
}
"#;
        let plib = device
            .new_library_with_source(probe_src, &options)
            .expect("probe compiles");
        let words_per_row = (n_sb * case.block_bytes / 16) as u32;
        let pscalar = device.new_buffer(8, MTLResourceOptions::StorageModeShared);
        unsafe {
            let p = pscalar.contents() as *mut u32;
            *p = words_per_row;
            *p.add(1) = rows as u32;
        }
        for (name, tgw) in [("probe32", 32u64), ("probe128", 128u64)] {
            let f = plib.get_function(name, None).unwrap();
            let pp = device.new_compute_pipeline_state_with_function(&f).unwrap();
            let mut best = f64::MAX;
            for _ in 0..iters {
                let cb = queue.new_command_buffer();
                let e = cb.new_compute_command_encoder();
                for _rep in 0..8 {
                    e.set_compute_pipeline_state(&pp);
                    e.set_buffer(0, Some(&w_buf), 0);
                    e.set_buffer(1, Some(&out_mc), 0);
                    e.set_buffer(2, Some(&pscalar), 0);
                    e.set_buffer(3, Some(&pscalar), 4);
                    e.dispatch_thread_groups(
                        MTLSize {
                            width: rows as u64,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: tgw,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                e.end_encoding();
                let t0 = Instant::now();
                cb.commit();
                cb.wait_until_completed();
                best = best.min(t0.elapsed().as_secs_f64() * 1000.0 / 8.0);
            }
            let usable = (rows * n_sb * (case.block_bytes / 16 * 16)) as f64;
            println!(
                "{name}: {:.3} ms ({:.1} GB/s)",
                best,
                usable / (best / 1000.0) / 1e9
            );
        }
    }

    // Perf: warmup + best-of-iters.
    let t1 = if is_v4 {
        run_mma(1, iters)
    } else {
        run_single_k(1, iters)
    };
    println!(
        "{} single-token: {:.3} ms  ({:.1} GB/s weight stream)",
        case.name,
        t1,
        weight_bytes / (t1 / 1000.0) / 1e9
    );
    if oracle_pipe.is_some() && is_v3 {
        let tc1 = run_mc(1, iters);
        println!(
            "{} candidate k=1 (mc kernel, n_tokens=1): {:.3} ms  ({:.1} GB/s) vs single {:.3} ms",
            case.name, tc1, weight_bytes / (tc1 / 1000.0) / 1e9, t1
        );
    }
    let perf_ks: &[usize] = if is_v4_wide {
        &[8, 16]
    } else if is_v3 || is_v4 {
        &[2, 4, 6, 8]
    } else {
        &[2, 4, 6, 8, 12, 16]
    };
    for &k in perf_ks {
        let tk = run_mc(k, iters);
        let tks = if is_v4 {
            run_v4_singles(k, iters.min(10))
        } else {
            run_single_k(k, iters.min(10))
        };
        println!(
            "{} k={:2}: mc {:.3} ms ({:.2}x single, {:.3} ms/col, {:.1} GB/s) | k-singles {:.3} ms | mc speedup {:.2}x",
            case.name, k, tk, tk / t1, tk / k as f64, weight_bytes / (tk / 1000.0) / 1e9,
            tks, tks / tk
        );
    }
}
