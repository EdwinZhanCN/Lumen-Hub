//! Burn-backed SigLIP text and vision encoders.
//!
//! A SigLIP repository ships one concrete architecture (hidden size, depth,
//! patch size, image/sequence length). Each architecture is a distinct
//! generated module under [`crate::model_arch`]; this file dispatches the
//! `model` configured in `model_info.json` to the right one behind the
//! [`SiglipTextArch`] / [`SiglipVisionArch`] traits, so adding a variant such as
//! `siglip2-so400m-patch14-384` is: drop in the generated module + add one match
//! arm here. Precision (fp32/fp16) reuses the same module — only the `.bpk`
//! differs.

use burn::tensor::{Int, Tensor, TensorData};

use crate::backend::{Backend, Device};
use crate::model_arch::siglip2_base_patch16_224 as base_patch16_224;
use crate::model_arch::siglip2_so400m_patch14_384 as so400m_patch14_384;

/// A loaded SigLIP text encoder for a specific architecture.
trait SiglipTextArch: Send + Sync {
    fn encode(&self, token_ids: &[i64]) -> Vec<f32>;
}

/// A loaded SigLIP vision encoder for a specific architecture.
trait SiglipVisionArch: Send + Sync {
    fn encode(&self, pixels: Vec<f32>, batch: usize, height: usize, width: usize) -> Vec<f32>;
}

/// SigLIP text encoder (architecture-erased).
pub struct SiglipTextModel {
    inner: Box<dyn SiglipTextArch>,
    /// Fixed token sequence length expected by the encoder's positional table.
    pub seq_len: usize,
}

impl SiglipTextModel {
    /// Loads the text encoder for `model_name` from its `.bpk` weights.
    pub fn load(model_name: &str, path: &str, device: Device) -> Result<Self, String> {
        let (inner, seq_len): (Box<dyn SiglipTextArch>, usize) = match model_name {
            "siglip2-base-patch16-224" => (
                Box::new(BasePatch16Text {
                    model: base_patch16_224::text::Model::<Backend>::from_file(path, &device),
                    device,
                }),
                64,
            ),
            "siglip2-so400m-patch14-384" => (
                Box::new(So400mText {
                    model: so400m_patch14_384::text::Model::<Backend>::from_file(path, &device),
                    device,
                }),
                64,
            ),
            other => return Err(unsupported(other)),
        };
        Ok(Self { inner, seq_len })
    }

    pub fn encode(&self, token_ids: &[i64]) -> Vec<f32> {
        self.inner.encode(token_ids)
    }
}

/// SigLIP vision encoder (architecture-erased).
pub struct SiglipVisionModel {
    inner: Box<dyn SiglipVisionArch>,
}

impl SiglipVisionModel {
    /// Loads the vision encoder for `model_name` from its `.bpk` weights.
    pub fn load(model_name: &str, path: &str, device: Device) -> Result<Self, String> {
        let inner: Box<dyn SiglipVisionArch> = match model_name {
            "siglip2-base-patch16-224" => Box::new(BasePatch16Vision {
                model: base_patch16_224::vision::Model::<Backend>::from_file(path, &device),
                device,
            }),
            "siglip2-so400m-patch14-384" => Box::new(So400mVision {
                model: so400m_patch14_384::vision::Model::<Backend>::from_file(path, &device),
                device,
            }),
            other => return Err(unsupported(other)),
        };
        Ok(Self { inner })
    }

    pub fn encode(&self, pixels: Vec<f32>, batch: usize, height: usize, width: usize) -> Vec<f32> {
        self.inner.encode(pixels, batch, height, width)
    }
}

fn unsupported(model_name: &str) -> String {
    format!(
        "no Burn SigLIP architecture registered for model `{model_name}`; \
         add it under model_arch/ and register it in models::siglip::model"
    )
}

// --- siglip2-base-patch16-224 ---------------------------------------------

struct BasePatch16Text {
    model: base_patch16_224::text::Model<Backend>,
    device: Device,
}

impl SiglipTextArch for BasePatch16Text {
    fn encode(&self, token_ids: &[i64]) -> Vec<f32> {
        let data = TensorData::new(token_ids.to_vec(), [1, token_ids.len()]);
        let input = Tensor::<Backend, 2, Int>::from_data(data, &self.device);
        tensor_to_f32(self.model.forward(input))
    }
}

struct BasePatch16Vision {
    model: base_patch16_224::vision::Model<Backend>,
    device: Device,
}

impl SiglipVisionArch for BasePatch16Vision {
    fn encode(&self, pixels: Vec<f32>, batch: usize, height: usize, width: usize) -> Vec<f32> {
        let data = TensorData::new(pixels, [batch, 3, height, width]);
        let input = Tensor::<Backend, 4>::from_data(data, &self.device);
        tensor_to_f32(self.model.forward(input))
    }
}

// --- siglip2-so400m-patch14-384 -------------------------------------------
//
// The so400m export returns `(last_hidden_state, pooled_features)`; the pooled
// `[batch, embedding_dim]` tensor (`.1`) is the text/image feature.

struct So400mText {
    model: so400m_patch14_384::text::Model<Backend>,
    device: Device,
}

impl SiglipTextArch for So400mText {
    fn encode(&self, token_ids: &[i64]) -> Vec<f32> {
        let data = TensorData::new(token_ids.to_vec(), [1, token_ids.len()]);
        let input = Tensor::<Backend, 2, Int>::from_data(data, &self.device);
        let (_hidden, pooled) = self.model.forward(input);
        tensor_to_f32(pooled)
    }
}

struct So400mVision {
    model: so400m_patch14_384::vision::Model<Backend>,
    device: Device,
}

impl SiglipVisionArch for So400mVision {
    fn encode(&self, pixels: Vec<f32>, batch: usize, height: usize, width: usize) -> Vec<f32> {
        let data = TensorData::new(pixels, [batch, 3, height, width]);
        let input = Tensor::<Backend, 4>::from_data(data, &self.device);
        let (_hidden, pooled) = self.model.forward(input);
        tensor_to_f32(pooled)
    }
}

fn tensor_to_f32<const D: usize>(tensor: Tensor<Backend, D>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("burn tensor data convertible to f32")
}
