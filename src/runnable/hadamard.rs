//! Activation/lookup transforms declared by the Bonsai 2 GGUF contract.
//! Weights stay packed. Only an activation or a looked-up embedding is rotated.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::gguf::GgufFile;
use crate::{BackendError, Result};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Transform {
    block: usize,
    signs: Arc<[f32]>,
    // Source feature order is [repeat, key_head, channel]; stored ssm_out
    // expects [key_head, repeat, channel] when gdn_v_grouped is set.
    permutation: Option<(usize, usize, usize)>,
}

impl Transform {
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_parts(&self) -> (usize, &[f32], Option<(usize, usize, usize)>) {
        (self.block, &self.signs, self.permutation)
    }

    #[cfg(test)]
    pub(crate) fn test_rotation(width: usize) -> Self {
        Self::test_rotation_geometry(width, 128, None)
    }

    #[cfg(test)]
    pub(crate) fn test_rotation_geometry(
        width: usize,
        block: usize,
        permutation: Option<(usize, usize, usize)>,
    ) -> Self {
        Self {
            block,
            signs: (0..width)
                .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
                .collect(),
            permutation,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut out = input.to_vec();
        if let Some((hd, nk, rep)) = self.permutation {
            for k in 0..nk {
                for r in 0..rep {
                    let src = (r * nk + k) * hd;
                    let dst = (k * rep + r) * hd;
                    out[dst..dst + hd].copy_from_slice(&input[src..src + hd]);
                }
            }
        }
        self.sign(&mut out);
        self.rotate(&mut out);
        out
    }

    pub fn inverse(&self, values: &mut [f32]) {
        self.rotate(values);
        self.sign(values);
    }

    fn sign(&self, values: &mut [f32]) {
        if !self.signs.is_empty() {
            debug_assert_eq!(values.len(), self.signs.len());
            for (x, s) in values.iter_mut().zip(self.signs.iter()) {
                *x *= s;
            }
        }
    }

    fn rotate(&self, values: &mut [f32]) {
        debug_assert_eq!(values.len() % self.block, 0);
        let scale = (self.block as f32).sqrt().recip();
        for block in values.chunks_exact_mut(self.block) {
            let mut stride = 1;
            while stride < self.block {
                for chunk in block.chunks_exact_mut(stride * 2) {
                    for i in 0..stride {
                        let (a, b) = (chunk[i], chunk[i + stride]);
                        chunk[i] = a + b;
                        chunk[i + stride] = a - b;
                    }
                }
                stride *= 2;
            }
            for x in block {
                *x *= scale;
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct Hadamard {
    pub forward: BTreeMap<String, Arc<Transform>>,
    pub inverse: BTreeMap<String, Arc<Transform>>,
}

fn invalid(message: impl std::fmt::Display) -> BackendError {
    BackendError::InvalidModelMetadata(format!("prism.hadamard: {message}"))
}

impl Hadamard {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        if !gguf
            .metadata
            .keys()
            .any(|k| k.starts_with("prism.hadamard."))
        {
            return Ok(Self::default());
        }
        if gguf.architecture() != Some("qwen35") {
            return Err(invalid("only the qwen35 graph implements folded weights"));
        }
        let version = gguf.metadata_u32("prism.hadamard.version");
        let tied = gguf
            .metadata_bool("prism.hadamard.tied_output")
            .unwrap_or(false);
        if !matches!(version, Some(1 | 2)) || (version == Some(2)) != tied {
            return Err(invalid(
                "version 1 requires an explicit head; version 2 requires tied_output=true",
            ));
        }
        let block = gguf.metadata_u32("prism.hadamard.block_size").unwrap_or(0) as usize;
        if !block.is_power_of_two()
            || gguf.metadata_string("prism.hadamard.transform")
                != Some("normalized-sylvester-walsh-hadamard")
            || gguf.metadata_string("prism.hadamard.axis") != Some("input-last-dimension")
        {
            return Err(invalid("invalid block size, transform or axis"));
        }
        let mut signs = BTreeMap::<usize, Arc<[f32]>>::new();
        match gguf.metadata_string("prism.hadamard.sign_mode") {
            Some("identity") => (),
            Some("explicit") => {
                let widths = gguf
                    .metadata_array_i32_optional("prism.hadamard.sign_widths")?
                    .unwrap_or_default();
                let values = gguf
                    .metadata_array_i32_optional("prism.hadamard.sign_values")?
                    .unwrap_or_default();
                if widths.is_empty() || values.iter().any(|v| !matches!(v, -1 | 1)) {
                    return Err(invalid("explicit signs require widths and +/-1 values"));
                }
                let mut offset: usize = 0;
                for width in widths {
                    let w = usize::try_from(width).map_err(|_| invalid("negative sign width"))?;
                    let end = offset
                        .checked_add(w)
                        .ok_or_else(|| invalid("sign width overflow"))?;
                    if w == 0 || w % block != 0 || end > values.len() || signs.contains_key(&w) {
                        return Err(invalid("invalid or duplicate sign width"));
                    }
                    signs.insert(w, values[offset..end].iter().map(|v| *v as f32).collect());
                    offset = end;
                }
                if offset != values.len() {
                    return Err(invalid("sign_values length mismatch"));
                }
            }
            _ => return Err(invalid("unknown or missing sign_mode")),
        }
        let names = gguf.metadata_array_strings("prism.hadamard.weight_names")?;
        let inverse = gguf
            .metadata_array_strings_optional("prism.hadamard.inverse_weight_names")?
            .unwrap_or_default();
        let explicit_head = gguf.tensors.iter().any(|t| t.name == "output.weight");
        if names.is_empty()
            || (tied && (explicit_head || !inverse.iter().any(|n| n == "token_embd.weight")))
            || (!tied && (!explicit_head || !names.iter().any(|n| n == "output.weight")))
        {
            return Err(invalid("invalid folded/tied output contract"));
        }
        let grouped = gguf
            .metadata_bool("prism.hadamard.gdn_v_grouped")
            .unwrap_or(false);
        let mut result = Self::default();
        let mut seen = BTreeSet::new();
        for (list, is_inverse) in [(&names, false), (&inverse, true)] {
            for name in list {
                let allowed = if is_inverse {
                    name == "token_embd.weight"
                } else if name == "output.weight" {
                    true
                } else {
                    let parts: Vec<_> = name.split('.').collect();
                    parts.len() == 4
                        && parts[0] == "blk"
                        && parts[1]
                            .parse::<u32>()
                            .is_ok_and(|i| i < gguf.metadata_u32("qwen35.block_count").unwrap_or(0))
                        && parts[3] == "weight"
                        && matches!(
                            parts[2],
                            "attn_q"
                                | "attn_k"
                                | "attn_v"
                                | "attn_qkv"
                                | "attn_gate"
                                | "attn_output"
                                | "ssm_out"
                                | "ffn_gate"
                                | "ffn_up"
                                | "ffn_down"
                        )
                };
                if !allowed || !seen.insert(name) {
                    return Err(invalid(format!(
                        "unsupported or duplicate folded tensor {name}"
                    )));
                }
                let tensor = gguf
                    .tensors
                    .iter()
                    .find(|t| &t.name == name)
                    .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
                let width = tensor.dimensions.first().copied().unwrap_or(0) as usize;
                if tensor.dimensions.len() != 2 || width == 0 || !width.is_multiple_of(block) {
                    return Err(invalid(format!("block size does not divide {name}")));
                }
                let sign = if signs.is_empty() {
                    Arc::from([])
                } else {
                    signs
                        .get(&width)
                        .cloned()
                        .ok_or_else(|| invalid(format!("missing signs for width {width}")))?
                };
                let permutation = if grouped && name.ends_with(".ssm_out.weight") {
                    let nk = gguf.metadata_u32("qwen35.ssm.group_count").unwrap_or(0) as usize;
                    let nv = gguf.metadata_u32("qwen35.ssm.time_step_rank").unwrap_or(0) as usize;
                    if nk == 0 || nv == 0 || !nv.is_multiple_of(nk) || !width.is_multiple_of(nv) {
                        return Err(invalid("invalid grouped GDN geometry"));
                    }
                    Some((width / nv, nk, nv / nk))
                } else {
                    None
                };
                let transform = Arc::new(Transform {
                    block,
                    signs: sign,
                    permutation,
                });
                if is_inverse {
                    result.inverse.insert(name.clone(), transform);
                } else {
                    result.forward.insert(name.clone(), transform);
                }
            }
        }
        if tied {
            result.forward.insert(
                "token_embd.weight".into(),
                result.inverse["token_embd.weight"].clone(),
            );
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufMetadataValue as V, GgufTensorDescriptor, GgufTensorType};

    fn fixture() -> GgufFile {
        GgufFile {
            path: "fixture.gguf".into(),
            version: 3,
            tensor_count: 1,
            metadata_count: 0,
            alignment: 32,
            data_start_offset: 0,
            metadata: BTreeMap::from([
                ("general.architecture".into(), V::String("qwen35".into())),
                ("qwen35.block_count".into(), V::U32(1)),
                ("prism.hadamard.version".into(), V::U32(2)),
                ("prism.hadamard.tied_output".into(), V::Bool(true)),
                ("prism.hadamard.block_size".into(), V::U32(128)),
                (
                    "prism.hadamard.transform".into(),
                    V::String("normalized-sylvester-walsh-hadamard".into()),
                ),
                (
                    "prism.hadamard.axis".into(),
                    V::String("input-last-dimension".into()),
                ),
                (
                    "prism.hadamard.sign_mode".into(),
                    V::String("explicit".into()),
                ),
                (
                    "prism.hadamard.sign_widths".into(),
                    V::Array(vec![V::I32(128)]),
                ),
                (
                    "prism.hadamard.sign_values".into(),
                    V::Array(vec![V::I32(-1); 128]),
                ),
                (
                    "prism.hadamard.weight_names".into(),
                    V::Array(vec![V::String("blk.0.ffn_up.weight".into())]),
                ),
                (
                    "prism.hadamard.inverse_weight_names".into(),
                    V::Array(vec![V::String("token_embd.weight".into())]),
                ),
            ]),
            tensors: ["token_embd.weight", "blk.0.ffn_up.weight"]
                .into_iter()
                .map(|name| GgufTensorDescriptor {
                    name: name.into(),
                    dimensions: vec![128, 128],
                    tensor_type: GgufTensorType::Ptq1_0,
                    relative_offset: 0,
                    absolute_offset: 0,
                    n_bytes: 3584,
                })
                .collect(),
        }
    }

    #[test]
    fn tied_head_and_malformed_contracts() {
        let gguf = fixture();
        let h = Hadamard::from_gguf(&gguf).unwrap();
        assert!(Arc::ptr_eq(
            &h.forward["token_embd.weight"],
            &h.inverse["token_embd.weight"]
        ));
        for (key, value) in [
            ("prism.hadamard.version", V::U32(1)),
            ("prism.hadamard.version", V::U32(3)),
            ("prism.hadamard.block_size", V::U32(256)),
            ("prism.hadamard.block_size", V::U32(0)),
            ("prism.hadamard.sign_values", V::Array(vec![V::I32(0); 128])),
            ("prism.hadamard.sign_widths", V::Array(vec![V::I32(64)])),
            ("prism.hadamard.sign_values", V::Array(vec![V::I32(1); 127])),
            ("prism.hadamard.inverse_weight_names", V::Array(vec![])),
            (
                "prism.hadamard.weight_names",
                V::Array(vec![V::String("blk.0.ssm_alpha.weight".into())]),
            ),
        ] {
            let mut bad = gguf.clone();
            bad.metadata.insert(key.into(), value);
            assert!(Hadamard::from_gguf(&bad).is_err(), "{key}");
        }
        let mut explicit = gguf.clone();
        explicit
            .metadata
            .insert("prism.hadamard.version".into(), V::U32(1));
        explicit
            .metadata
            .insert("prism.hadamard.tied_output".into(), V::Bool(false));
        let mut head = explicit.tensors[0].clone();
        head.name = "output.weight".into();
        explicit.tensors.push(head);
        assert!(Hadamard::from_gguf(&explicit).is_err());
        explicit.metadata.insert(
            "prism.hadamard.weight_names".into(),
            V::Array(vec![
                V::String("blk.0.ffn_up.weight".into()),
                V::String("output.weight".into()),
            ]),
        );
        assert!(Hadamard::from_gguf(&explicit).is_ok());
    }
    #[test]
    fn forward_and_lookup_inverse_preserve_dot_product() {
        let t = Transform {
            block: 4,
            signs: Arc::from([1., -1., -1., 1., -1., 1., -1., 1.]),
            permutation: None,
        };
        let x = [1., 2., 3., 4., 5., -6., 7., 8.];
        let w = [2., -1., 0., 3., -2., 4., 1., 0.];
        let mut restored = t.forward(&x);
        t.inverse(&mut restored);
        assert_eq!(restored, x);
        let transformed = t.forward(&x);
        let mut unrotated = w;
        t.inverse(&mut unrotated);
        assert_eq!(
            transformed.iter().zip(w).map(|(a, b)| a * b).sum::<f32>(),
            x.iter().zip(unrotated).map(|(a, b)| a * b).sum::<f32>()
        );
    }
    #[test]
    fn grouped_ssm_features_are_permuted_before_rotation() {
        let t = Transform {
            block: 2,
            signs: Arc::from([]),
            permutation: Some((2, 2, 3)),
        };
        let x = (0..12).map(|v| v as f32).collect::<Vec<_>>();
        let mut y = t.forward(&x);
        t.inverse(&mut y);
        for (actual, expected) in y
            .iter()
            .zip([0., 1., 4., 5., 8., 9., 2., 3., 6., 7., 10., 11.])
        {
            assert!((actual - expected).abs() < 0.00001);
        }
    }
}
