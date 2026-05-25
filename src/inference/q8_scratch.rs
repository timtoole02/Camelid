use std::{cell::RefCell, mem};

use crate::{
    tensor::{parse_byte_count_env, Q8_0Block},
    Result,
};

thread_local! {
    static Q8_0_FILE_READER_ROW_CHUNK: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static Q8_0_FILE_READER_CHUNK_SCALES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static Q8_0_FILE_READER_QUANTIZED_INPUTS: RefCell<Vec<Q8_0Block>> = const { RefCell::new(Vec::new()) };
    static Q8_0_FILE_READER_OUTPUT_CHUNK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn q8_0_file_reader_retained_scratch_bytes() -> usize {
    const DEFAULT_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES: usize = 64 * 1024 * 1024;
    parse_byte_count_env("CAMELID_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES")
        .unwrap_or(DEFAULT_Q8_0_FILE_READER_RETAINED_SCRATCH_BYTES)
}

fn q8_0_file_reader_retained_scratch_entries<T>() -> usize {
    q8_0_file_reader_retained_scratch_bytes() / mem::size_of::<T>().max(1)
}

pub(super) fn cap_q8_0_file_reader_scratch<T>(scratch: &mut Vec<T>, retained_len: usize) {
    let retained_entries = q8_0_file_reader_retained_scratch_entries::<T>();
    if scratch.capacity() > retained_entries {
        *scratch = Vec::with_capacity(retained_len.min(retained_entries));
    } else if scratch.len() > retained_len {
        scratch.truncate(retained_len);
    }
}

pub(super) fn with_q8_0_file_reader_row_chunk<T>(
    len: usize,
    f: impl FnOnce(&mut [u8]) -> Result<T>,
) -> Result<T> {
    Q8_0_FILE_READER_ROW_CHUNK.with(|cell| {
        let mut row_chunk = cell.borrow_mut();
        if row_chunk.len() < len {
            row_chunk.resize(len, 0);
        }
        let result = f(&mut row_chunk[..len]);
        cap_q8_0_file_reader_scratch(&mut row_chunk, len);
        result
    })
}

pub(super) fn with_q8_0_file_reader_chunk_scales<T>(
    len: usize,
    f: impl FnOnce(&mut [f32]) -> Result<T>,
) -> Result<T> {
    Q8_0_FILE_READER_CHUNK_SCALES.with(|cell| {
        let mut scales = cell.borrow_mut();
        if scales.len() < len {
            scales.resize(len, 0.0);
        }
        let result = f(&mut scales[..len]);
        cap_q8_0_file_reader_scratch(&mut scales, len);
        result
    })
}

pub(super) fn with_q8_0_file_reader_quantized_inputs<T>(
    f: impl FnOnce(&mut Vec<Q8_0Block>) -> Result<T>,
) -> Result<T> {
    Q8_0_FILE_READER_QUANTIZED_INPUTS.with(|cell| {
        let mut quantized_inputs = cell.borrow_mut();
        let result = f(&mut quantized_inputs);
        // Keep the allocation as reusable scratch capacity, but do not leave the
        // previous activation blocks logically live between file-backed Q8 calls.
        quantized_inputs.clear();
        cap_q8_0_file_reader_scratch(&mut quantized_inputs, 0);
        result
    })
}

pub(super) fn with_q8_0_file_reader_output_chunk<T>(
    len: usize,
    f: impl FnOnce(&mut [f32]) -> Result<T>,
) -> Result<T> {
    Q8_0_FILE_READER_OUTPUT_CHUNK.with(|cell| {
        let mut output_chunk = cell.borrow_mut();
        if output_chunk.len() < len {
            output_chunk.resize(len, 0.0);
        }
        let result = f(&mut output_chunk[..len]);
        cap_q8_0_file_reader_scratch(&mut output_chunk, len);
        result
    })
}

#[cfg(test)]
pub(super) fn q8_0_file_reader_scratch_capacities() -> (usize, usize, usize, usize) {
    let row_chunk = Q8_0_FILE_READER_ROW_CHUNK.with(|cell| cell.borrow().capacity());
    let chunk_scales = Q8_0_FILE_READER_CHUNK_SCALES.with(|cell| cell.borrow().capacity());
    let quantized_inputs = Q8_0_FILE_READER_QUANTIZED_INPUTS.with(|cell| cell.borrow().capacity());
    let output_chunk = Q8_0_FILE_READER_OUTPUT_CHUNK.with(|cell| cell.borrow().capacity());
    (row_chunk, chunk_scales, quantized_inputs, output_chunk)
}
