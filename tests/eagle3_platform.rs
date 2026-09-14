//! Exercises the production executable, whose library is built without cfg(test).
#[test]
fn eagle3_server_validates_context_before_loading_models() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_camelid"))
        .args(["serve", "--addr", "127.0.0.1:0"])
        .env("CAMELID_SPEC_DECODE", "eagle3")
        .env("CAMELID_EAGLE3_LOGICAL_TOKENS", "17")
        .env_remove("CAMELID_EAGLE3_MODEL")
        .output()
        .expect("launch production server executable");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("must be 2048 or 4096")
            || stderr.contains("requires a Metal or CUDA build"),
        "{stderr}"
    );
}
