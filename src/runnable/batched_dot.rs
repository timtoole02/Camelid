//! Vectorize across independent tokens, keeping each dot's reduction order.
//! Reducing feature lanes in parallel would change rounding and draft decisions.
pub(super) struct BatchedInputs {
    tiles: Vec<[f32; 8]>,
    width: usize,
    count: usize,
}

impl BatchedInputs {
    pub(super) fn new(inputs: &[Vec<f32>], width: usize) -> Self {
        let mut tiles = vec![[0.0; 8]; inputs.len().div_ceil(8) * width];
        for (p, input) in inputs.iter().enumerate() {
            assert_eq!(input.len(), width);
            for (i, &value) in input.iter().enumerate() {
                tiles[p / 8 * width + i][p % 8] = value;
            }
        }
        Self {
            tiles,
            width,
            count: inputs.len(),
        }
    }

    pub(super) fn dot_row(&self, row: &[f32]) -> Vec<f32> {
        assert_eq!(row.len(), self.width);
        let mut output = Vec::with_capacity(self.count.div_ceil(8) * 8);
        for tile in self.tiles.chunks_exact(self.width) {
            let mut sums = [0.0_f32; 8];
            for (&weight, inputs) in row.iter().zip(tile) {
                for (sum, input) in sums.iter_mut().zip(inputs) {
                    *sum += weight * input;
                }
            }
            output.extend(sums);
        }
        output.truncate(self.count);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_preserve_sequential_f32_dots_including_partial_tiles() {
        for count in [1, 2, 3, 4, 7, 8, 9, 31, 64, 128] {
            for width in [1, 7, 128, 256, 4096] {
                let inputs: Vec<Vec<f32>> = (0..count)
                    .map(|p| {
                        (0..width)
                            .map(|i| ((i * 7 + p * 13) as f32).sin())
                            .collect()
                    })
                    .collect();
                let packed = BatchedInputs::new(&inputs, width);
                for r in 0..3 {
                    let row: Vec<_> = (0..width)
                        .map(|i| ((i * 17 + r * 3) as f32).cos())
                        .collect();
                    let expected: Vec<f32> = inputs
                        .iter()
                        .map(|x| row.iter().zip(x).map(|(a, b)| a * b).sum())
                        .collect();
                    assert_eq!(
                        packed.dot_row(&row),
                        expected,
                        "count={count}, width={width}"
                    );
                }
            }
        }
    }
}
