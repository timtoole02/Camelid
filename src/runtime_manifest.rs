//! Machine-readable status and evidence registry for runtime projects.
//!
//! This is deliberately separate from model support.  A runtime optimization
//! can be implemented or benchmark-gated without accidentally promoting a
//! model/quantization support row.

use std::{collections::HashSet, sync::OnceLock};

use serde::{Deserialize, Serialize};

const RUNTIME_CAPABILITIES_JSON: &str = include_str!("../config/runtime-capabilities.json");

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeCapabilityManifest {
    pub schema_version: u32,
    pub projects: Vec<RuntimeProjectCapability>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeProjectCapability {
    pub id: String,
    pub project: u32,
    pub status: String,
    pub default_enabled: bool,
    pub configuration: Vec<String>,
    pub dependencies: Vec<String>,
    pub evidence: Vec<String>,
    pub safety_contract: String,
}

static MANIFEST: OnceLock<RuntimeCapabilityManifest> = OnceLock::new();

pub fn runtime_capability_manifest() -> &'static RuntimeCapabilityManifest {
    MANIFEST.get_or_init(|| {
        let manifest: RuntimeCapabilityManifest = serde_json::from_str(RUNTIME_CAPABILITIES_JSON)
            .expect("config/runtime-capabilities.json must parse");
        validate_manifest(&manifest).expect("config/runtime-capabilities.json must be valid");
        manifest
    })
}

fn validate_manifest(manifest: &RuntimeCapabilityManifest) -> Result<(), String> {
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported runtime capability schema {}",
            manifest.schema_version
        ));
    }
    if manifest.projects.is_empty() {
        return Err("runtime capability project list is empty".to_string());
    }
    let mut ids = HashSet::new();
    let mut projects = HashSet::new();
    for item in &manifest.projects {
        if item.id.trim().is_empty() || item.status.trim().is_empty() {
            return Err("runtime capability id/status cannot be empty".to_string());
        }
        if !ids.insert(item.id.as_str()) {
            return Err(format!("duplicate runtime capability id {}", item.id));
        }
        if !projects.insert(item.project) {
            return Err(format!(
                "duplicate runtime capability project {}",
                item.project
            ));
        }
        if item.safety_contract.trim().is_empty() {
            return Err(format!("{} has no safety contract", item.id));
        }
        if item.evidence.is_empty() {
            return Err(format!("{} has no evidence target", item.id));
        }
    }
    for item in &manifest.projects {
        for dependency in &item.dependencies {
            if !ids.contains(dependency.as_str()) {
                return Err(format!(
                    "{} refers to unknown dependency {dependency}",
                    item.id
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_runtime_manifest_is_valid_and_covers_selected_projects() {
        let manifest = runtime_capability_manifest();
        assert_eq!(manifest.schema_version, 1);
        let projects: HashSet<_> = manifest.projects.iter().map(|item| item.project).collect();
        assert_eq!(projects, HashSet::from([2, 3, 4, 5, 6, 7, 8]));
    }

    #[test]
    fn validator_rejects_unknown_dependencies() {
        let mut manifest = runtime_capability_manifest().clone();
        manifest.projects[0]
            .dependencies
            .push("does-not-exist".to_string());
        assert!(validate_manifest(&manifest)
            .unwrap_err()
            .contains("unknown dependency"));
    }
}
