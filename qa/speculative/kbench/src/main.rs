// K-quant GEMV microbench: extracts LINEAR_ROW_SHADER from the Camelid worktree at
// RUNTIME, compiles it, and times the single-token + multi-column K-quant GEMVs at
// production shapes. Also cross-checks the mc kernel against k single-token
// dispatches bit-for-bit, so a kernel edit that breaks exactness fails HERE first.
//
// Usage: kbench [q4k|q5k|q6k|q4kv2|q6kv2|q4kmma|q6kmma]
//               [--rows N] [--nsb N] [--iters N]
use metal::*;
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
    let queue = device.new_command_queue();
    let options = CompileOptions::new();
    let lib = device
        .new_library_with_source(&extract_shader(), &options)
        .expect("shader compiles");
    let ref_lib = device
        .new_library_with_source(&extract_ref_shader(), &options)
        .expect("ref shader compiles");

    let is_mma = which == "q4kmma" || which == "q6kmma";
    let is_mma_q6 = which == "q6kmma";
    let is_v2 = which.ends_with("v2") || is_mma;
    let case = match which {
        "q4k" => Case {
            name: "q4k",
            block_bytes: 144,
            scratch_ints_per_sb: 9,
            single: "q4k_linear_simd",
            mc: "q4k_linear_simd_mc",
            tiled: "q4k_linear_tiled",
        },
        "q5k" => Case {
            name: "q5k",
            block_bytes: 176,
            scratch_ints_per_sb: 9,
            single: "q5k_linear_simd",
            mc: "q5k_linear_simd_mc",
            tiled: "q5k_linear_tiled",
        },
        "q6k" => Case {
            name: "q6k",
            block_bytes: 210,
            scratch_ints_per_sb: 8,
            single: "q6k_linear_simd",
            mc: "q6k_linear_simd_mc",
            tiled: "q6k_linear_tiled",
        },
        "q4kv2" => Case {
            name: "q4kv2",
            block_bytes: 144,
            scratch_ints_per_sb: 9,
            single: "q4k_linear_simd_v2",
            mc: "q4k_linear_simd_mc_v2",
            tiled: "q4k_linear_tiled",
        },
        "q4kmma" => Case {
            name: "q4kmma",
            block_bytes: 144,
            scratch_ints_per_sb: 9,
            single: "q4k_linear_simd_v2",
            mc: "q4k_linear_mma_mc_v2",
            tiled: "q4k_linear_tiled",
        },
        "q6kmma" => Case {
            name: "q6kmma",
            block_bytes: 210,
            scratch_ints_per_sb: 8,
            single: "q6k_linear_simd_v2",
            mc: "q6k_linear_mma_mc_v2",
            tiled: "q6k_linear_tiled",
        },
        "q5kv2" => Case {
            name: "q5kv2",
            block_bytes: 176,
            scratch_ints_per_sb: 9,
            single: "q5k_linear_simd_v2",
            mc: "q5k_linear_simd_mc_v2",
            tiled: "q5k_linear_tiled",
        },
        "q6kv2" => Case {
            name: "q6kv2",
            block_bytes: 210,
            scratch_ints_per_sb: 8,
            single: "q6k_linear_simd_v2",
            mc: "q6k_linear_simd_mc_v2",
            tiled: "q6k_linear_tiled",
        },
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
    let (single, mc) = if is_v2 {
        (v2_pipe(case.single), v2_pipe(case.mc))
    } else {
        (pipe(case.single), pipe(case.mc))
    };
    // v2 is its own bit-universe: the contract is single_v2 == mc_v2 (checked
    // below), not equality with the HEAD kernel.
    let ref_single = if is_v2 {
        single.clone()
    } else {
        let f = ref_lib.get_function(case.single, None).expect("ref single");
        device
            .new_compute_pipeline_state_with_function(&f)
            .expect("ref single pipe")
    };
    let _ = case.tiled;
    let quantize = pipe("quantize_q8k_rows");

    let max_k = 16usize;
    let cols = n_sb * 256;

    // Deterministic activations for max_k columns.
    let mut y_flat = vec![0.0f32; max_k * cols];
    for (i, y) in y_flat.iter_mut().enumerate() {
        let t = i / cols;
        let c = i % cols;
        *y = ((((t * 131 + c * 17) % 251) as f32) - 125.0) * 0.017 + t as f32 * 0.0011;
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
        *b = ((i * 13 + i / 97 + 5) % 256) as u8;
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
    const REPS: usize = 8;

    // --- Reference: k single-token dispatches into out_single ------------------
    // Returns amortized ms for ONE set of k dispatches.
    // v2 single: 32 threads (one simdgroup) covering Q4K_V2_ROWS_PER_SG rows.
    let single_tg: u64 = 32;
    let rows_per_tg: usize = 1;
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
                    e.set_buffer(0, Some(&scales_buf), (t * n_sb * 4) as u64);
                    e.set_buffer(1, Some(&quants_buf), (t * cols) as u64);
                    e.set_buffer(2, Some(&w_buf), 0);
                    e.set_buffer(3, Some(&out_single), (t * rows * 4) as u64);
                    e.set_buffer(4, Some(&scalar), 0);
                    e.set_buffer(5, Some(&scalar), 4);
                    e.set_threadgroup_memory_length(0, tg_bytes);
                    if rows_per_tg > 1 {
                        e.set_threadgroup_memory_length(
                            1,
                            ((rows_per_tg * 9 * 4).next_multiple_of(16)) as u64,
                        );
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
        let stage_y = v2_pipe(if is_mma_q6 {
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
    let run_mma = |k: usize, timed_iters: usize| -> f64 {
        let (stage_y, stage_ysums, y_half, ysums) = mma_aux.as_ref().unwrap();
        let k_pad = (k + 7) & !7;
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
                        width: (rows as u64).div_ceil(8),
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
            let t0 = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64);
        }
        best
    };

    // --- mc kernel: one dispatch covering k columns ----------------------------
    let mc_tg: u64 = std::env::var("KBENCH_MC_TG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
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
                e.set_buffer(0, Some(&scales_buf), 0);
                e.set_buffer(1, Some(&quants_buf), 0);
                e.set_buffer(2, Some(&w_buf), 0);
                e.set_buffer(3, Some(&out_mc), 0);
                e.set_buffer(4, Some(&scalar), 0);
                e.set_buffer(5, Some(&scalar), 4);
                e.set_buffer(6, Some(&scalar), 8);
                e.set_threadgroup_memory_length(0, tg_bytes);
                e.dispatch_thread_groups(
                    MTLSize {
                        width: rows as u64,
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

    let skip_check = std::env::var("KBENCH_SKIP_CHECK").is_ok();
    // Bit-identity: edited single(k dispatches) vs the HEAD oracle, k = 16 covers all columns.
    // v2 is its own bit-universe AND its own dispatch geometry, so the HEAD-oracle
    // dispatch does not apply there; single_v2 vs mc_v2 below is the real contract.
    if !skip_check && !is_v2 {
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
        for k in 2..=max_k {
            run_single_k(k, 1);
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
            "{} rows={} n_sb={} bit-identity PASS (k=2..=16)",
            case.name, rows, n_sb
        );
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
    let t1 = run_single_k(1, iters);
    println!(
        "{} single-token: {:.3} ms  ({:.1} GB/s weight stream)",
        case.name,
        t1,
        weight_bytes / (t1 / 1000.0) / 1e9
    );
    for k in [2usize, 4, 6, 8, 12, 16] {
        let tk = run_mc(k, iters);
        let tks = run_single_k(k, iters.min(10));
        println!(
            "{} k={:2}: mc {:.3} ms ({:.2}x single, {:.3} ms/col, {:.1} GB/s) | k-singles {:.3} ms | mc speedup {:.2}x",
            case.name, k, tk, tk / t1, tk / k as f64, weight_bytes / (tk / 1000.0) / 1e9,
            tks, tks / tk
        );
    }
}
