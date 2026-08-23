use std::{env, path::PathBuf, process::Command};

fn main() {
    embed_build_provenance();
    ensure_web_ui_placeholder();
    println!("cargo:rerun-if-changed=src/x86_amx_q8.c");
    println!("cargo:rerun-if-env-changed=CAMELID_BUILD_X86_AMX_SHIM");
    println!("cargo:rustc-check-cfg=cfg(camelid_x86_amx_shim)");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        // Accelerate provides the refmath vDSP / __sincosf_stret system bindings
        // (a framework binding, not compiled C/C++). The DiffusionGemma expert
        // argsort is now pure Rust (see src/diffusion_gemma.rs) — no C++ shim.
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    if target_os == "windows" {
        // CUDA is part of the DEFAULT Windows build: turn on the `cuda` cfg so the GPU
        // backend compiles without an explicit `--features cuda`. The Windows-only
        // cudarc dependency in Cargo.toml is always linked; the driver/NVRTC load
        // dynamically at runtime (no CUDA SDK needed to build; no-ops without a GPU).
        // Other platforms keep CUDA opt-in via the `cuda` feature.
        println!("cargo:rustc-cfg=feature=\"cuda\"");

        // Export the Optimus / Enduro hints so a laptop's hybrid-graphics driver
        // routes this process to the discrete NVIDIA (or AMD) GPU instead of the
        // integrated Intel one. Reading these exported DWORDs at process start is
        // the documented mechanism; combined with the per-app GPU preference the
        // binary sets at runtime, Windows attributes the app to the dGPU.
        //
        // Scope the /EXPORT to the `camelid` bin only: the backing statics live in
        // src/main.rs, so exporting them from sibling bins (e.g. repack-ghost)
        // would be an unresolved external (LNK2001).
        println!("cargo:rustc-link-arg-bin=camelid=/EXPORT:NvOptimusEnablement,DATA");
        println!(
            "cargo:rustc-link-arg-bin=camelid=/EXPORT:AmdPowerXpressRequestHighPerformance,DATA"
        );
        // Windows' default 1 MiB stack overflows while Clap builds this binary's CLI.
        println!("cargo:rustc-link-arg-bin=camelid=/STACK:8388608");
    }
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    // MESA: CUDA is part of the DEFAULT build on x86_64 Linux too (mirrors the Windows
    // block above). Pairs with the non-optional x86_64-linux `cudarc` in Cargo.toml so a
    // bare `cargo build` compiles the GPU backend with no `--features cuda`. Driver/NVRTC
    // load dynamically at runtime (no CUDA SDK to build; no-ops without a device). aarch64
    // Linux (Pi) is deliberately excluded here and stays opt-in. No Optimus/`/STACK` link
    // args — those are Windows-only and stay in the windows block above.
    if target_os == "linux" && target_arch == "x86_64" {
        println!("cargo:rustc-cfg=feature=\"cuda\"");
    }
    if target_os != "linux" || target_arch != "x86_64" {
        return;
    }
    let require_amx_shim = env_flag_enabled("CAMELID_BUILD_X86_AMX_SHIM");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let obj = out_dir.join("x86_amx_q8.o");
    let lib = out_dir.join("libcamelid_x86_amx_q8.a");

    let status = Command::new("gcc")
        .args([
            "-O3",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-mavx512f",
            "-mfma",
            "-mamx-tile",
            "-mamx-int8",
            "-c",
            "src/x86_amx_q8.c",
            "-o",
        ])
        .arg(&obj)
        .status();
    let Ok(status) = status else {
        if require_amx_shim {
            panic!("failed to run gcc for x86 AMX Q8 kernel");
        }
        println!("cargo:warning=skipping optional x86 AMX Q8 shim because gcc could not be run");
        return;
    };
    if !status.success() {
        if require_amx_shim {
            panic!("gcc failed building x86 AMX Q8 kernel");
        }
        println!(
            "cargo:warning=skipping optional x86 AMX Q8 shim because gcc rejected the AMX flags"
        );
        return;
    }

    let status = Command::new("ar").arg("crus").arg(&lib).arg(&obj).status();
    let Ok(status) = status else {
        if require_amx_shim {
            panic!("failed to run ar for x86 AMX Q8 kernel");
        }
        println!("cargo:warning=skipping optional x86 AMX Q8 shim because ar could not be run");
        return;
    };
    if !status.success() {
        if require_amx_shim {
            panic!("ar failed building x86 AMX Q8 kernel");
        }
        println!("cargo:warning=skipping optional x86 AMX Q8 shim because ar failed");
        return;
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=camelid_x86_amx_q8");
    println!("cargo:rustc-cfg=camelid_x86_amx_shim");
}

// Embed git provenance so a running binary reports its own version/commit
// (used by parity receipts) without shelling out at request time. Builds
// without a git checkout simply omit the env vars; the receipt module falls
// back to the crate version.
fn embed_build_provenance() {
    // Resolve paths through Git rather than assuming `.git` is a directory.
    // Linked worktrees use a `.git` pointer file, and their HEAD/index live in
    // the main checkout's worktrees directory. The symbolic branch ref is the
    // path that actually changes when a worktree commit advances.
    for path in ["HEAD", "index", "packed-refs"] {
        emit_git_rerun_path(path);
    }
    if let Some(head_ref) = git_stdout(&["symbolic-ref", "-q", "HEAD"]) {
        emit_git_rerun_path(&head_ref);
    }
    if let Some(commit) = git_stdout(&["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=CAMELID_GIT_COMMIT={commit}");
    }
    if let Some(describe) = git_stdout(&["describe", "--tags", "--dirty"]) {
        println!("cargo:rustc-env=CAMELID_GIT_DESCRIBE={describe}");
    }
}

fn emit_git_rerun_path(git_path: &str) {
    let Some(path) = git_stdout(&["rev-parse", "--git-path", git_path]) else {
        return;
    };
    let path = PathBuf::from(path);
    let path = if path.is_absolute() {
        path
    } else {
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR")).join(path)
    };
    println!("cargo:rerun-if-changed={}", path.display());
}

// The web UI (frontend/dist) is embedded into the binary via rust-embed, which
// fails to compile if the folder has no index.html. A fresh checkout has not
// run `npm run build` yet, so write a placeholder index.html when one is
// missing — a real `npm run build` overwrites it. This keeps `cargo build`
// working with no Node toolchain while still embedding the real UI in release
// builds that run the frontend build first.
fn ensure_web_ui_placeholder() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let dist = manifest_dir.join("frontend").join("dist");
    let index = dist.join("index.html");
    // Re-embed whenever the built UI changes (or the placeholder is replaced).
    println!("cargo:rerun-if-changed={}", dist.display());
    if index.exists() {
        return;
    }
    if let Err(err) = std::fs::create_dir_all(&dist) {
        println!(
            "cargo:warning=could not create {}: {err}; web UI will be unavailable",
            dist.display()
        );
        return;
    }
    let placeholder = "<!doctype html><!-- placeholder: run `cd frontend && npm run build` to embed the real UI -->\n";
    if let Err(err) = std::fs::write(&index, placeholder) {
        println!(
            "cargo:warning=could not write {}: {err}; web UI will be unavailable",
            index.display()
        );
    }
}

fn git_stdout(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let value = value.trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn env_flag_enabled(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("enabled")
                || value.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}
