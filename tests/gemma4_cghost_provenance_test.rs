mod support;

use camelid::gguf::read_metadata;
use camelid::ghost::{write_cghost_moe, GhostFile};
use camelid::model::{Gemma4Binding, LlamaModelConfig};
use camelid::tensor::TensorStore;
use std::path::{Path, PathBuf};

/// Destructive repair is opt-in: without this env the test never writes a
/// byte. A plain `cargo test --all-targets` once regenerated the live 12.9 GB
/// .cghost from the sparse hot shadow the default GGUF had become — a
/// well-formed, identity-valid, 99.65%-zeros artifact installed over the
/// operator's real one.
const INSTALL_OPT_IN_ENV: &str = "CAMELID_TEST_INSTALL_CGHOST";

fn install_opt_in() -> bool {
    std::env::var(INSTALL_OPT_IN_ENV).is_ok_and(|value| value == "1")
}

fn validate_pair(
    cghost_path: &Path,
    model_path: &Path,
    binding: &Gemma4Binding,
    n_experts: usize,
) -> Result<(), String> {
    let ghost =
        GhostFile::open(cghost_path).map_err(|err| format!("open {cghost_path:?}: {err}"))?;
    if !ghost.has_sampled_source_identity() {
        return Err("legacy .cghost without cryptographic source identity".into());
    }
    ghost
        .validate_moe_source_identity(model_path, binding, n_experts)
        .map_err(|err| format!("identity/density validation: {err}"))
}

#[test]
fn test_cghost_sha256_provenance() {
    let model_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.gguf");
    let cghost_path = PathBuf::from(support::model_root()).join("gemma-4-26B_q4_0-it.cghost");

    if !model_path.is_file() {
        eprintln!("Model not found: {:?}", model_path);
        return;
    }

    println!("Opening GGUF model metadata: {:?}", model_path);
    let gguf = read_metadata(&model_path).expect("read gguf metadata");
    let config = LlamaModelConfig::from_gguf(&gguf).expect("metadata");
    let binding = Gemma4Binding::bind(&gguf, &config).expect("bind gemma4");
    let n_experts = config.moe.as_ref().expect("has moe").expert_count as usize;

    let verdict = cghost_path
        .is_file()
        .then(|| validate_pair(&cghost_path, &model_path, &binding, n_experts));
    match (&verdict, install_opt_in()) {
        (Some(Ok(())), _) => {
            println!(
                "[PROVENANCE OK] .cghost identity and payload density match {:?}",
                model_path
            );
            return;
        }
        (None, false) => {
            eprintln!(
                "No .cghost at {cghost_path:?}; nothing to validate. Set {INSTALL_OPT_IN_ENV}=1 to build one from the full GGUF."
            );
            return;
        }
        (Some(Err(reason)), false) => {
            panic!(
                "cghost pair failed validation: {reason}\n\
                 This test is read-only by default and will NOT touch {cghost_path:?}.\n\
                 To repack and install a fresh .cghost (only with the FULL source GGUF in \
                 place — the repack refuses sparse hot shadows), rerun with {INSTALL_OPT_IN_ENV}=1."
            );
        }
        (_, true) => {
            println!(
                ">>> {INSTALL_OPT_IN_ENV}=1: repacking a fresh .cghost from {:?} ({})",
                model_path,
                verdict
                    .as_ref()
                    .and_then(|v| v.as_ref().err().map(String::as_str))
                    .unwrap_or("no existing .cghost")
            );
        }
    }

    // Opt-in repair path. All writes go to a unique temp sibling; the default
    // path is only ever replaced by an atomic same-directory rename of a fully
    // verified artifact.
    let store = TensorStore::open(&model_path, &gguf);
    let source_name = model_path
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let temp_cghost =
        cghost_path.with_extension(format!("cghost.repack.{}.part", std::process::id()));
    let result = (|| -> Result<(), String> {
        let t0 = std::time::Instant::now();
        write_cghost_moe(&store, &binding, &config, &source_name, &temp_cghost, None)
            .map_err(|err| format!("repack refused or failed: {err}"))?;
        println!(
            "Repack write completed in {:.1}s",
            t0.elapsed().as_secs_f64()
        );

        validate_pair(&temp_cghost, &model_path, &binding, n_experts)?;
        let ghost = GhostFile::open(&temp_cghost).map_err(|err| err.to_string())?;
        ghost
            .verify_moe_expert_payload_against_source(&model_path, &binding, n_experts)
            .map_err(|err| format!("full payload verification: {err}"))?;

        std::fs::rename(&temp_cghost, &cghost_path)
            .map_err(|err| format!("install rename: {err}"))?;
        println!(
            "[SUCCESS] Installed fresh fully-verified .cghost at {:?}",
            cghost_path
        );
        Ok(())
    })();
    if result.is_err() {
        std::fs::remove_file(&temp_cghost).ok();
    }
    result.unwrap_or_else(|reason| panic!("{reason}"));
}
