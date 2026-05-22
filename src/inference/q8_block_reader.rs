use std::{
    fs::File,
    io::{Error as IoError, ErrorKind, Result as IoResult},
    os::unix::fs::FileExt,
};

use super::f16_bits_to_f32;
use crate::tensor::record_q8_0_file_read;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q8BlockReader {
    pub(super) offset: u64,
    pub(super) num_blocks: usize,
}

impl Q8BlockReader {
    pub const BLOCK_SIZE_BYTES: usize = 34;
    pub const WEIGHTS_PER_BLOCK: usize = 32;

    pub fn new(offset: u64, num_blocks: usize) -> Self {
        Self { offset, num_blocks }
    }

    pub fn dequantize_block_to_slice(
        &self,
        file: &File,
        block_idx: usize,
        dest: &mut [f32],
    ) -> IoResult<()> {
        if block_idx >= self.num_blocks {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "block index out of bounds",
            ));
        }

        let dest_offset = block_idx
            .checked_mul(Self::WEIGHTS_PER_BLOCK)
            .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "destination offset overflow"))?;
        if dest_offset + Self::WEIGHTS_PER_BLOCK > dest.len() {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "destination buffer too small",
            ));
        }

        let block_offset = self
            .offset
            .checked_add((block_idx * Self::BLOCK_SIZE_BYTES) as u64)
            .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "block offset overflow"))?;
        let mut block_data = [0u8; Self::BLOCK_SIZE_BYTES];
        file.read_exact_at(&mut block_data, block_offset)?;
        record_q8_0_file_read(block_data.len());

        let scale_bits = u16::from_le_bytes(block_data[0..2].try_into().expect("2-byte scale"));
        let scale = f16_bits_to_f32(scale_bits);
        for i in 0..Self::WEIGHTS_PER_BLOCK {
            dest[dest_offset + i] = f32::from(block_data[2 + i] as i8) * scale;
        }
        Ok(())
    }
}
