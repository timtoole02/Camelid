fn main() {
    // The app's own commands must be DECLARED here for tauri-build to generate
    // their `allow-*` / `deny-*` ACL permissions. Local webviews (the splash)
    // may call app commands implicitly, but the loopback sidecar UI is a
    // REMOTE origin to Tauri's ACL, and a remote origin can only invoke a
    // command that a capability explicitly grants (capabilities/sidecar.json).
    // Without this manifest those grants have nothing to reference and every
    // invoke from the embedded UI fails with "not allowed by ACL".
    let attributes =
        tauri_build::Attributes::new().app_manifest(tauri_build::AppManifest::new().commands(&[
            "startup_snapshot",
            "choose_models_directory",
            "reset_models_directory",
        ]));
    tauri_build::try_build(attributes).expect("failed to run tauri-build");
}
