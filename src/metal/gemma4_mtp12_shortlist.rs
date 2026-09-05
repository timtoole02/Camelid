//! Reader for the optional, draft-only overlapping-cluster head index.
//! The source identity in this file does not certify draft accuracy. Every
//! proposed token must still pass the unchanged target verifier.

use std::{io::Read, path::Path};

use crate::{BackendError, Result};

const CLUSTERS: usize = 2048;
const HIDDEN: usize = 1024;
const VOCAB: usize = 262144;
const HEADER_BYTES: usize = 56;
const CENTROID_BYTES: usize = CLUSTERS * HIDDEN * 4;
const FILE_BYTES: usize = HEADER_BYTES + CENTROID_BYTES + VOCAB * 4 * 2;
const ASSISTANT_SHA256: [u8; 32] = [
    0x67, 0xf1, 0x42, 0x0c, 0xf2, 0x4a, 0xa5, 0x06,
    0x50, 0x89, 0xaa, 0xed, 0x17, 0x52, 0x23, 0xf7,
    0xc2, 0x45, 0xcc, 0xfd, 0xa1, 0x61, 0x11, 0xb6,
    0xc5, 0x67, 0x65, 0xaf, 0xd7, 0x28, 0x0d, 0xb6,
];

pub(super) struct Mtp12ShortlistData {
    pub(super) centroids: Vec<f32>,
    /// Three cluster IDs and a zero padding element per vocabulary row.
    pub(super) token_clusters: Vec<u16>,
}

fn invalid(detail: &str) -> BackendError {
    BackendError::InvalidTensorData(format!("Gemma 4 MTP12 shortlist: {detail}"))
}

pub(super) fn read_sidecar(path: &Path) -> Result<Mtp12ShortlistData> {
    let file = std::fs::File::open(path)
        .map_err(|e| invalid(&format!("cannot open {}: {e}", path.display())))?;
    // Bound allocation even for a corrupt or accidentally selected model file.
    let mut bytes = Vec::with_capacity(FILE_BYTES + 1);
    file.take((FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| invalid(&format!("cannot read {}: {e}", path.display())))?;
    parse_sidecar(&bytes)
}

fn parse_sidecar(bytes: &[u8]) -> Result<Mtp12ShortlistData> {
    if bytes.len() != FILE_BYTES {
        return Err(invalid("wrong file length"));
    }
    let word = |offset: usize| {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    };
    if &bytes[..4] != b"C4SL" || word(4) != 1 {
        return Err(invalid("unsupported magic or version"));
    }
    if bytes[8..40] != ASSISTANT_SHA256 {
        return Err(invalid("assistant source identity does not match"));
    }
    if [word(40), word(44), word(48), word(52)]
        != [CLUSTERS as u32, HIDDEN as u32, VOCAB as u32, 3]
    {
        return Err(invalid("expected dimensions 2048 x 1024, vocab 262144, overlap 3"));
    }
    let centroids: Vec<f32> = bytes[HEADER_BYTES..HEADER_BYTES + CENTROID_BYTES]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if centroids.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite centroid"));
    }
    let token_clusters: Vec<u16> = bytes[HEADER_BYTES + CENTROID_BYTES..]
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect();
    for row in token_clusters.chunks_exact(4) {
        if row[..3].iter().any(|&c| usize::from(c) >= CLUSTERS) {
            return Err(invalid("cluster index out of range"));
        }
        if row[3] != 0 || row[0] == row[1] || row[0] == row[2] || row[1] == row[2] {
            return Err(invalid("expected three distinct clusters and zero padding"));
        }
    }
    Ok(Mtp12ShortlistData { centroids, token_clusters })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        let mut b = vec![0; FILE_BYTES];
        b[..4].copy_from_slice(b"C4SL");
        b[4..8].copy_from_slice(&1u32.to_le_bytes());
        b[8..40].copy_from_slice(&ASSISTANT_SHA256);
        for (offset, value) in [(40, 2048u32), (44, 1024), (48, 262144), (52, 3)] {
            b[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        b[HEADER_BYTES..HEADER_BYTES + 4].copy_from_slice(&1.25f32.to_le_bytes());
        for row in b[HEADER_BYTES + CENTROID_BYTES..].chunks_exact_mut(8) {
            row.copy_from_slice(&[0, 0, 1, 0, 255, 7, 0, 0]);
        }
        b
    }

    #[test]
    fn shortlist_parses_little_endian_payload() {
        let parsed = parse_sidecar(&fixture()).unwrap();
        assert_eq!(parsed.centroids.len(), CLUSTERS * HIDDEN);
        assert_eq!(parsed.centroids[0].to_bits(), 1.25f32.to_bits());
        assert_eq!(&parsed.token_clusters[..4], &[0, 1, 2047, 0]);
    }

    #[test]
    fn shortlist_rejects_bad_identity_shape_and_payload() {
        let valid = fixture();
        for offset in [0, 4, 8, 40, 44, 48, 52] {
            let mut b = valid.clone();
            b[offset] ^= 1;
            assert!(parse_sidecar(&b).is_err(), "header offset {offset}");
        }
        assert!(parse_sidecar(&valid[..FILE_BYTES - 1]).is_err());
        let mut b = valid.clone();
        b.push(0);
        assert!(parse_sidecar(&b).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut b = valid.clone();
            b[HEADER_BYTES..HEADER_BYTES + 4].copy_from_slice(&value.to_le_bytes());
            assert!(parse_sidecar(&b).is_err());
        }
        for (offset, value) in [(0, 2048u16), (2, 0u16), (6, 1u16)] {
            let mut b = valid.clone();
            let start = HEADER_BYTES + CENTROID_BYTES + offset;
            b[start..start + 2].copy_from_slice(&value.to_le_bytes());
            assert!(parse_sidecar(&b).is_err());
        }
    }
}
