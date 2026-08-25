use std::{ffi::OsString, path::PathBuf};

/// Resolve the operator's model root without embedding a workstation-specific
/// home directory in integration-test binaries or repository fixtures.
pub fn model_root() -> OsString {
    std::env::var_os("CAMELID_MODEL_ROOT")
        .filter(|root| !root.is_empty())
        .unwrap_or_else(|| {
            operator_home()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("models")
                .into_os_string()
        })
}

/// Resolve the current operator home on Unix and Windows hosts.
pub fn operator_home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
}

#[allow(dead_code)]
pub fn is_canonical_path_within_operator_home(path: &std::path::Path) -> bool {
    let Some(home) = operator_home() else {
        return false;
    };
    if !home.is_absolute() || home.parent().is_none() {
        return false;
    }
    let Ok(canonical_home) = home.canonicalize() else {
        return false;
    };
    if canonical_home.parent().is_none() {
        return false;
    }
    path.canonicalize()
        .is_ok_and(|canonical_path| canonical_path.starts_with(canonical_home))
}
