//! Exercises the production executable, whose library is built without cfg(test).
#[cfg(not(target_os = "macos"))]
#[test]
fn eagle3_server_reports_unimplemented_backend_before_loading_models() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_camelid"))
        .args(["serve", "--addr", "127.0.0.1:0"])
        .env("CAMELID_SPEC_DECODE", "eagle3")
        .env_remove("CAMELID_EAGLE3_MODEL")
        .output()
        .expect("launch production server executable");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr
            .contains("CUDA learned draft head and EAGLE cache orchestration are not implemented"),
        "{stderr}"
    );
    assert!(stderr.contains("Unset CAMELID_SPEC_DECODE"), "{stderr}");
}
