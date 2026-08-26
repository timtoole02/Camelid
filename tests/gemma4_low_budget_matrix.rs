//! Fine-Grained Low-Budget Cache Matrix & Three-Tier Memory Telemetry Benchmark
//! Budgets: 8, 16, 24, 32, and 40 slots/layer on Genuine Gemma 4 26B-A4B.

mod support;

use camelid::gemma4_runtime::Gemma4Runtime;
use std::{path::PathBuf, process::Command, time::Instant};

#[derive(Debug, Clone, Default)]
struct VmStats {
    page_size: u64,
    pages_free: u64,
    pages_active: u64,
    pages_inactive: u64,
    pages_wired: u64,
    pages_compressed: u64,
    file_backed_pages: u64,
    compressor_pages: u64,
}

impl VmStats {
    fn capture() -> Self {
        let output = Command::new("vm_stat").output().ok();
        let mut stats = Self {
            page_size: 16384,
            ..Default::default()
        };
        let Some(output) = output else {
            return stats;
        };
        let text = String::from_utf8_lossy(&output.stdout);
        for line in text.lines() {
            let line = line.trim();
            if line.contains("page size of") {
                if let Some(pos) = line.find("page size of ") {
                    let rest = &line[pos + 13..];
                    if let Some(end) = rest.find(' ') {
                        stats.page_size = rest[..end].parse().unwrap_or(16384);
                    }
                }
            } else if let Some((k, v)) = line.split_once(':') {
                let v = v.trim().trim_end_matches('.').parse::<u64>().unwrap_or(0);
                match k.trim() {
                    "Pages free" => stats.pages_free = v,
                    "Pages active" => stats.pages_active = v,
                    "Pages inactive" => stats.pages_inactive = v,
                    "Pages wired down" => stats.pages_wired = v,
                    "Pages occupied by compressor" => stats.pages_compressed = v,
                    "Pages stored in compressor" => stats.compressor_pages = v,
                    "File-backed pages" => stats.file_backed_pages = v,
                    _ => {}
                }
            }
        }
        stats
    }

    fn free_gib(&self) -> f64 {
        (self.pages_free * self.page_size) as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    fn file_backed_gib(&self) -> f64 {
        (self.file_backed_pages * self.page_size) as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    fn compressed_gib(&self) -> f64 {
        (self.compressor_pages * self.page_size) as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

#[allow(deprecated)]
fn get_process_rss_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut info: libc::mach_task_basic_info = std::mem::zeroed();
        let mut count = (std::mem::size_of::<libc::mach_task_basic_info>()
            / std::mem::size_of::<libc::natural_t>())
            as libc::mach_msg_type_number_t;
        let kret = libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as *mut libc::integer_t,
            &mut count,
        );
        if kret == libc::KERN_SUCCESS {
            info.resident_size
        } else {
            0
        }
    }
    #[cfg(not(target_os = "macos"))]
    0
}

fn get_rusage_faults() -> (u64, u64) {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            (ru.ru_minflt as u64, ru.ru_majflt as u64)
        } else {
            (0, 0)
        }
    }
    #[cfg(not(target_os = "macos"))]
    (0, 0)
}

fn get_vm_pressure_level() -> u32 {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut level: u32 = 0;
        let mut size = std::mem::size_of::<u32>();
        let name = std::ffi::CString::new("kern.memorystatus_vm_pressure_level").unwrap();
        if libc::sysctlbyname(
            name.as_ptr(),
            &mut level as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        ) == 0
        {
            level
        } else {
            1
        }
    }
    #[cfg(not(target_os = "macos"))]
    1
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct FineBudgetResult {
    slots_per_layer: usize,
    metal_resident_gib: f64,
    process_rss_gib: f64,
    system_free_gib: f64,
    file_backed_gib: f64,
    compressed_gib: f64,
    vm_pressure: u32,
    k5_round_ms: f64,
    candidate_tok_s: f64,
    emitted_tok_s: f64,
    major_faults: u64,
    minor_faults: u64,
    parity_pass: bool,
}

#[test]
fn test_genuine_gemma4_low_budget_matrix() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() || !cghost_path.is_file() {
        eprintln!("SKIP: 26B MoE model/cghost not found");
        return;
    }

    let test_prompt = "<|turn>user\nExplain the concept of quantum entanglement and its potential applications in cryptography in simple terms.<turn|>\n<|turn>model\n";
    let token_budget = 16;
    let slot_budgets = [8, 16, 24, 32, 40];
    let mut results = Vec::new();

    println!("==========================================================================================================");
    println!("GENUINE GEMMA 4 26B-A4B LOW-BUDGET MATRIX & THREE-TIER MEMORY BENCHMARK");
    println!("Budgets: 8, 16, 24, 32, 40 slots/layer");
    println!("Objective: Minimize physical SSD reads and exposed I/O by preserving macOS filesystem page cache");
    println!("==========================================================================================================\n");

    for &slots in &slot_budgets {
        println!(">>> TESTING LOW BUDGET: {} slots/layer <<<", slots);
        std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS", "1");
        std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_SLOTS_FAST", "1");
        std::env::set_var(
            "CAMELID_GEMMA4_GHOST_METAL_SLOTS_PER_LAYER",
            slots.to_string(),
        );
        std::env::set_var("CAMELID_GEMMA4_GHOST_METAL", "1");
        std::env::set_var("CAMELID_GEMMA4_GHOST_METAL_STATS", "1");

        let runtime = Gemma4Runtime::load_ghost_moe(&model_path, &cghost_path, 4096, false)
            .expect("load ghost moe");

        let metal_resident_gib = (30.0 * slots as f64 * 3_345_408.0) / (1024.0 * 1024.0 * 1024.0);

        // 1. K=1 Baseline Greedy
        std::env::set_var("CAMELID_SPEC_DECODE", "0");
        std::env::set_var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS", "0");
        let (_k1_text, k1_tokens) = runtime
            .generate_greedy(test_prompt, token_budget)
            .expect("k1 decode");

        // 2. K=5 Speculative with 3-tier telemetry
        std::env::set_var("CAMELID_SPEC_DECODE", "1");
        std::env::set_var("CAMELID_GEMMA4_SPEC_DRAFT_TOKENS", "4");

        let (minflt_before, majflt_before) = get_rusage_faults();
        let t_k5 = Instant::now();
        let (_k5_text, k5_tokens) = runtime
            .generate_greedy_speculative(test_prompt, token_budget)
            .expect("k5 decode");
        let k5_dur = t_k5.elapsed().as_secs_f64();
        let (minflt_after, majflt_after) = get_rusage_faults();

        let vm_after = VmStats::capture();
        let rss_after = get_process_rss_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
        let vm_pressure = get_vm_pressure_level();

        let rounds = (k5_tokens.len() as f64 / 2.5).ceil().max(1.0) as usize;
        let k5_round_ms = (k5_dur * 1000.0) / rounds as f64;
        let candidate_tok_s = 5000.0 / k5_round_ms.max(0.001);
        let emitted_tok_s = k5_tokens.len() as f64 / k5_dur.max(0.001);
        let parity_pass = k1_tokens == k5_tokens;

        let major_faults = majflt_after.saturating_sub(majflt_before);
        let minor_faults = minflt_after.saturating_sub(minflt_before);

        println!("  Slots: {:2} | Metal: {:.2} GiB | RSS: {:.2} GiB | PageCache: {:.2} GiB | Free: {:.2} GiB | Pressure: {}",
            slots, metal_resident_gib, rss_after, vm_after.file_backed_gib(), vm_after.free_gib(), vm_pressure);
        println!(
            "  Major Faults (Physical NVMe): {} | Minor Faults (Page Cache): {}",
            major_faults, minor_faults
        );
        println!(
            "  K=5 Round: {:.1} ms | Cand: {:.2} tok/s | Emitted: {:.2} tok/s | Parity: {}\n",
            k5_round_ms,
            candidate_tok_s,
            emitted_tok_s,
            if parity_pass { "PASS" } else { "FAIL" }
        );

        results.push(FineBudgetResult {
            slots_per_layer: slots,
            metal_resident_gib,
            process_rss_gib: rss_after,
            system_free_gib: vm_after.free_gib(),
            file_backed_gib: vm_after.file_backed_gib(),
            compressed_gib: vm_after.compressed_gib(),
            vm_pressure,
            k5_round_ms,
            candidate_tok_s,
            emitted_tok_s,
            major_faults,
            minor_faults,
            parity_pass,
        });
    }

    println!("\n==========================================================================================================");
    println!("LOW-BUDGET (8–40 SLOTS) CACHE & THREE-TIER MEMORY MATRIX");
    println!("==========================================================================================================");
    println!(
        "{:>6} | {:>9} | {:>8} | {:>9} | {:>9} | {:>9} | {:>10} | {:>10} | {:>10} | {:>6}",
        "Slots",
        "Metal GiB",
        "RSS GiB",
        "PageCache",
        "NVMe(Maj)",
        "RAM(Min)",
        "K=5 Rnd ms",
        "Cand tok/s",
        "Emit tok/s",
        "Parity"
    );
    println!("----------------------------------------------------------------------------------------------------------");
    for r in &results {
        println!("{:>6} | {:>9.2} | {:>8.2} | {:>9.2} | {:>9} | {:>9} | {:>10.1} | {:>10.2} | {:>10.2} | {:>6}",
            r.slots_per_layer,
            r.metal_resident_gib,
            r.process_rss_gib,
            r.file_backed_gib,
            r.major_faults,
            r.minor_faults,
            r.k5_round_ms,
            r.candidate_tok_s,
            r.emitted_tok_s,
            if r.parity_pass { "PASS" } else { "FAIL" },
        );
    }
    println!("==========================================================================================================\n");
}
