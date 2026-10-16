//! Integer PTQ1_0 projections for the speculative drafter only.
//! Group scaling follows Prism's PTQ1_0 x Q8_0 contraction. The target keeps
//! its f32 path and verifies every proposal, including any quantization changes.
//! Reference: PrismML-Eng/llama.cpp prism-b10777-e3de715,
//! ggml/src/ggml-cpu/quants.c, ggml_vec_dot_ptq1_0_q8_0_generic.
use crate::tensor::{f16_bits_to_f32, Q8_0Block};

fn unpack(block: &[u8]) -> [i8; 128] {
    let mut values = [0_i8; 128];
    let mut offset = 0;
    for (start, width, count) in [(0, 16, 5), (16, 8, 5), (24, 2, 4)] {
        for power in [1_u8, 3, 9, 27, 81].iter().take(count) {
            for &code in &block[start..start + width] {
                values[offset] = ((u16::from(code.wrapping_mul(*power)) * 3) >> 8) as i8 - 1;
                offset += 1;
            }
        }
    }
    values
}

#[inline]
fn block_dot(weights: &[i8; 128], activation: &[Q8_0Block]) -> f32 {
    let mut sum = 0.0_f32;
    for (weights, input) in weights.chunks_exact(32).zip(activation) {
        // Integer addition can vectorize across features without changing its
        // result. Widen before multiplying, including the i8::MIN case.
        let dot: i32 = weights
            .iter()
            .zip(&input.quants)
            .map(|(&weight, &input)| i32::from(weight) * i32::from(input))
            .sum();
        sum += input.scale * dot as f32;
    }
    sum
}

pub(super) fn dot_row(row: &[u8], input: &[Q8_0Block]) -> f32 {
    debug_assert_eq!(row.len() / 28 * 4, input.len());
    let mut sum = 0.0_f32;
    for (block, activation) in row.chunks_exact(28).zip(input.chunks_exact(4)) {
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[26], block[27]]));
        sum += scale * block_dot(&unpack(block), activation);
    }
    sum
}

pub(super) fn dot_rows(row: &[u8], inputs: &[Vec<Q8_0Block>]) -> Vec<f32> {
    let mut sums = vec![0.0_f32; inputs.len()];
    for (index, block) in row.chunks_exact(28).enumerate() {
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[26], block[27]]));
        let weights = unpack(block);
        for (sum, input) in sums.iter_mut().zip(inputs) {
            *sum += scale * block_dot(&weights, &input[index * 4..index * 4 + 4]);
        }
    }
    sums
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_unpack_matches_the_independent_weight_decoder_for_every_byte() {
        for code in 0..=255_u8 {
            let mut block = [code; 28];
            block[26..].copy_from_slice(&0x3c00_u16.to_le_bytes());
            let decoded = crate::runnable::dequant::dequantize(
                crate::gguf::GgufTensorType::Ptq1_0,
                &block,
                128,
                "test",
            )
            .unwrap();
            assert_eq!(unpack(&block).map(f32::from).as_slice(), decoded);
        }
    }

    #[test]
    fn batch_and_single_integer_dots_match_scalar_groups_with_signed_extremes() {
        let mut row = Vec::new();
        for seed in [0_u8, 71, 129] {
            row.extend((0..26).map(|i| seed.wrapping_add(i * 7)));
            row.extend(0x3800_u16.to_le_bytes());
        }
        let inputs: Vec<Vec<Q8_0Block>> = (0..9)
            .map(|p| {
                (0..12)
                    .map(|b| Q8_0Block {
                        scale: (b + 1) as f32 / 64.0,
                        quants: std::array::from_fn(|i| {
                            ((i * 31 + p * 17 + b * 7) % 256) as u8 as i8
                        }),
                    })
                    .collect()
            })
            .collect();
        let decoded = crate::runnable::dequant::dequantize(
            crate::gguf::GgufTensorType::Ptq1_0,
            &row,
            384,
            "test",
        )
        .unwrap();
        let expected: Vec<f32> = inputs
            .iter()
            .map(|input| {
                let mut sum = 0.0_f32;
                for (weights, activations) in decoded.chunks_exact(128).zip(input.chunks_exact(4)) {
                    let mut inner = 0.0_f32;
                    for (weights, activation) in weights.chunks_exact(32).zip(activations) {
                        let mut integer = 0_i32;
                        for (&weight, &quant) in weights.iter().zip(&activation.quants) {
                            integer += (weight * 2.0) as i32 * i32::from(quant);
                        }
                        inner += activation.scale * integer as f32;
                    }
                    sum += 0.5 * inner;
                }
                sum
            })
            .collect();
        assert_eq!(dot_rows(&row, &inputs), expected);
        for (input, expected) in inputs.iter().zip(expected) {
            assert_eq!(dot_row(&row, input), expected);
        }
    }
}
