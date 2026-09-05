//! Explicit diagnostics for offline assistant replay; never selected by default.
use super::*;
use std::io::Write;
use std::path::PathBuf;

pub(crate) const ENV: &str = "CAMELID_MTP12_DUMP_FINAL_KV";

pub(crate) fn enabled_path() -> Option<&'static Path> {
    static PATH: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| std::env::var_os(ENV).filter(|p| !p.is_empty()).map(PathBuf::from))
        .as_deref()
}

fn diagnostic_error(error: impl std::fmt::Display) -> BackendError {
    BackendError::InvalidModelMetadata(format!("MTP12 offline snapshot: {error}"))
}

/// A seed sidecar uses the existing query path plus `.seeds`: 16-byte LE header
/// (anchor token, absolute query anchor, target prefix length, draft count), then
/// 3,840 f32 values before the existing gather's BF16 rounding. Both selectors
/// must be set. The caller has already synchronized and validated the view.
pub(crate) fn record_initial_seed(
    query_path: &Path, anchor: u32, position: usize, prefix: usize, drafts: usize,
    values: &[f32],
) -> Result<()> {
    if enabled_path().is_none() { return Ok(()); }
    if values.len() != 3_840 || values.iter().any(|x| !x.is_finite()) {
        return Err(diagnostic_error("invalid initial recurrent seed"));
    }
    let mut name = query_path.as_os_str().to_os_string();
    name.push(".seeds");
    let mut bytes = Vec::with_capacity(16 + values.len() * 4);
    for value in [u64::from(anchor), position as u64, prefix as u64, drafts as u64] {
        bytes.extend_from_slice(&u32::try_from(value).map_err(diagnostic_error)?.to_le_bytes());
    }
    for value in values { bytes.extend_from_slice(&value.to_le_bytes()); }
    std::fs::OpenOptions::new().create(true).append(true).open(PathBuf::from(name))
        .and_then(|mut file| file.write_all(&bytes)).map_err(diagnostic_error)
}

fn packed_prefix(values: &[f32], heads: usize, capacity: usize, dim: usize, prefix: usize) -> Result<Vec<u8>> {
    let elements = heads.checked_mul(capacity).and_then(|n| n.checked_mul(dim))
        .ok_or_else(|| diagnostic_error("KV geometry overflow"))?;
    if prefix == 0 || prefix > capacity || values.len() != elements {
        return Err(diagnostic_error("KV prefix or buffer extent is invalid"));
    }
    let mut bytes = Vec::with_capacity(heads * prefix * dim * 4);
    for head in 0..heads {
        let base = head * capacity * dim;
        for value in &values[base..base + prefix * dim] {
            if !value.is_finite() { return Err(diagnostic_error("non-finite committed KV")); }
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}

impl Gemma4GpuRuntime {
    pub(super) fn maybe_dump_mtp12_final_kv(
        &self, prompt: &str, generated: &[u32], position: usize, stats: &Gemma4Mtp12MetalStats,
    ) -> Result<()> {
        let Some(root) = enabled_path() else { return Ok(()); };
        let sequence = self.verifier_state.lock().map_err(diagnostic_error)?;
        if sequence.lane != Gemma4DenseSequenceLane::OrderedVerifier
            || sequence.pending.is_some() || sequence.logical_len != position {
            return Err(diagnostic_error("snapshot requires the final committed ordered prefix"));
        }
        let prompt_ids = self.tokenizer.encode(prompt, true, true)?;
        if position != prompt_ids.len() + stats.committed_input_rows as usize {
            return Err(diagnostic_error("final prefix does not match committed row count"));
        }
        std::fs::create_dir_all(root).map_err(diagnostic_error)?;
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
            .map_err(diagnostic_error)?.as_nanos();
        let directory = root.join(format!("p{}-n{}-{stamp}", prompt_ids.len(), generated.len()));
        std::fs::create_dir(&directory).map_err(diagnostic_error)?;
        let mut files = Vec::new();
        self.model.with_kv_device_views(&[46, 47], |views| -> Result<()> {
            for view in views {
                let expected = if view.source_layer == 46 { (8, 256) } else { (1, 512) };
                if (view.kv_heads, view.head_dim) != expected || position > view.max_positions {
                    return Err(diagnostic_error("source layer geometry mismatch"));
                }
                for (kind, buffer) in [("key", view.key), ("value", view.value)] {
                    let values = unsafe {
                        std::slice::from_raw_parts(
                            buffer.contents().cast::<u8>().add(view.byte_offset as usize).cast::<f32>(),
                            view.byte_len as usize / 4,
                        )
                    };
                    let bytes = packed_prefix(values, view.kv_heads, view.max_positions, view.head_dim, position)?;
                    let name = format!("layer{}-{kind}.f32le", view.source_layer);
                    std::fs::write(directory.join(&name), &bytes).map_err(diagnostic_error)?;
                    files.push(serde_json::json!({"file":name,"layer":view.source_layer,"kind":kind,
                        "kv_heads":view.kv_heads,"head_dim":view.head_dim,"kv_len":position,
                        "bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}));
                }
            }
            Ok(())
        }).ok_or_else(|| diagnostic_error("source KV views unavailable"))??;
        let environment = std::env::vars().filter(|(k, _)| k.starts_with("CAMELID_"))
            .collect::<std::collections::BTreeMap<_, _>>();
        let metadata = serde_json::json!({"format":"camelid-mtp12-final-kv-v1","diagnostic_only":true,
            "target_sha256":crate::metal::GEMMA4_12B_QAT_Q4_0_TARGET_SHA256,
            "assistant_sha256":crate::metal::GEMMA4_12B_MTP_ASSISTANT_SHA256,
            "prompt_token_ids":prompt_ids,"generated_token_ids":generated,"committed_prefix":position,
            "committed_input_rows":stats.committed_input_rows,"rounds":stats.rounds,
            "environment":environment,"files":files});
        std::fs::write(directory.join("snapshot.json"), serde_json::to_vec_pretty(&metadata).map_err(diagnostic_error)?)
            .map_err(diagnostic_error)?;
        eprintln!("[gemma4-mtp12] final committed KV snapshot: {}", directory.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn offline_snapshot_packs_only_committed_rows_in_head_order() {
        let values = [1.0, 2.0, 3.0, 4.0, f32::NAN, f32::NAN,
                      5.0, 6.0, 7.0, 8.0, f32::NAN, f32::NAN];
        let bytes = packed_prefix(&values, 2, 3, 2, 2).unwrap();
        let result = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        assert_eq!(result, vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        assert!(packed_prefix(&values, 2, 3, 2, 3).is_err());
        assert!(packed_prefix(&values, 2, 3, 2, 4).is_err());
        assert!(packed_prefix(&values[..11], 2, 3, 2, 2).is_err());
    }
}
