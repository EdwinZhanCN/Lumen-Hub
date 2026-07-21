//! Tiny deterministic Burn model for the QA harness.
//!
//! Real weights, real burnpack load path, real forward pass — just small
//! (3·32·32 → 16). Weights are derived from integer arithmetic so every
//! platform generates bit-identical fixtures; no RNG, no seed plumbing.

use burn::nn::Linear;
use burn::prelude::*;

use crate::backend::{Backend, Device};

pub const QA_INPUT_CHANNELS: usize = 3;
pub const QA_INPUT_SIZE: usize = 32;
pub const QA_INPUT_NUMEL: usize = QA_INPUT_CHANNELS * QA_INPUT_SIZE * QA_INPUT_SIZE;
pub const QA_EMBED_DIM: usize = 16;

/// One linear projection over the flattened image, L2-normalized per row.
/// `nn::Linear` on purpose: the runtime Q8 quantizer only quantizes params
/// inside weight-bearing containers, and the QA model must exercise it.
#[derive(Module, Debug)]
pub struct QaNet<B: burn::prelude::Backend> {
    pub proj: Linear<B>,
}

impl QaNet<Backend> {
    /// Builds the net with the canonical deterministic weights.
    pub fn deterministic(device: &Device) -> Self {
        let weight: Vec<f32> = (0..QA_INPUT_NUMEL * QA_EMBED_DIM)
            .map(|index| {
                let row = index / QA_EMBED_DIM;
                let col = index % QA_EMBED_DIM;
                deterministic_value(row, col)
            })
            .collect();
        let bias: Vec<f32> = (0..QA_EMBED_DIM)
            .map(|col| deterministic_value(usize::MAX / 2, col))
            .collect();

        let weight = Tensor::<Backend, 1>::from_floats(weight.as_slice(), device)
            .reshape([QA_INPUT_NUMEL, QA_EMBED_DIM]);
        let bias = Tensor::<Backend, 1>::from_floats(bias.as_slice(), device);

        Self {
            proj: Linear {
                weight: burn::module::Param::from_tensor(weight),
                bias: Some(burn::module::Param::from_tensor(bias)),
            },
        }
    }

    /// Embeds `batch` flattened images (`pixels.len() == batch * QA_INPUT_NUMEL`)
    /// into L2-normalized rows, returned row-major.
    pub fn embed(&self, pixels: &[f32], batch: usize, device: &Device) -> Vec<f32> {
        assert_eq!(pixels.len(), batch * QA_INPUT_NUMEL, "pixel buffer size");
        let input =
            Tensor::<Backend, 1>::from_floats(pixels, device).reshape([batch, QA_INPUT_NUMEL]);
        let output = self.proj.forward(input);
        let norm = output
            .clone()
            .powi_scalar(2)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-12);
        let normalized = output / norm;
        normalized.into_data().to_vec::<f32>().expect("f32 output")
    }
}

/// Integer-derived weight in roughly [-0.05, 0.05]; bit-identical everywhere.
fn deterministic_value(row: usize, col: usize) -> f32 {
    let hash = (row.wrapping_mul(31).wrapping_add(col.wrapping_mul(17))) % 197;
    (hash as f32 / 197.0 - 0.5) * 0.1
}
