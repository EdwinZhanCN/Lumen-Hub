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
use crate::model_arch::aesthetic_head;
use crate::model_arch::siglip2_base_patch16_224 as base_patch16_224;
use crate::model_arch::siglip2_so400m_patch14_384 as so400m_patch14_384;
use crate::model_arch::{load_aesthetic_head, load_burnpack};

/// A loaded SigLIP text encoder for a specific architecture.
trait SiglipTextArch: Send + Sync {
    fn encode(&self, token_ids: &[i64]) -> Vec<f32>;
}

/// A loaded SigLIP vision encoder for a specific architecture.
trait SiglipVisionArch: Send + Sync {
    fn encode(&self, pixels: Vec<f32>, batch: usize, height: usize, width: usize) -> Vec<f32>;
}

/// A loaded aesthetic-scoring head for a specific architecture.
///
/// Consumes the vision encoder's **raw** pooled features (`image_features`); the
/// L2-normalization is baked into the head graph, so it must see the un-normalized
/// pooled output. Returns one scalar score per batch row.
trait AestheticHeadArch: Send + Sync {
    fn score(&self, pooled: &[f32], batch: usize) -> Vec<f32>;
}

/// SigLIP text encoder (architecture-erased).
pub struct SiglipTextModel {
    inner: Box<dyn SiglipTextArch>,
    /// Fixed token sequence length expected by the encoder's positional table.
    pub seq_len: usize,
}

impl SiglipTextModel {
    /// Loads the text encoder for `model_name` from its `.bpk` weights.
    pub fn load(
        model_name: &str,
        path: &str,
        precision: &str,
        device: Device,
    ) -> Result<Self, String> {
        let (inner, seq_len): (Box<dyn SiglipTextArch>, usize) = match model_name {
            "siglip2-base-patch16-224" => (
                Box::new(BasePatch16Text {
                    model: load_burnpack(
                        base_patch16_224::text::Model::<Backend>::new(&device),
                        path,
                        precision,
                    )?,
                    device,
                }),
                64,
            ),
            "siglip2-so400m-patch14-384" => (
                Box::new(So400mText {
                    model: load_burnpack(
                        so400m_patch14_384::text::Model::<Backend>::new(&device),
                        path,
                        precision,
                    )?,
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

/// SigLIP vision encoder (architecture-erased), optionally bundled with an
/// aesthetic-scoring head sharing the same backbone.
pub struct SiglipVisionModel {
    inner: Box<dyn SiglipVisionArch>,
    head: Option<Box<dyn AestheticHeadArch>>,
}

impl SiglipVisionModel {
    /// Loads the vision encoder for `model_name` from its `.bpk` weights.
    ///
    /// When `head_path` is `Some`, the matching aesthetic head is loaded from that
    /// burnpack and runs on the same forward pass — `encode` then also returns a
    /// per-row score. The head is loaded fp16 (never Q8) even under `fp16q8`.
    pub fn load(
        model_name: &str,
        path: &str,
        precision: &str,
        device: Device,
        head_path: Option<&str>,
    ) -> Result<Self, String> {
        let inner: Box<dyn SiglipVisionArch> = match model_name {
            "siglip2-base-patch16-224" => Box::new(BasePatch16Vision {
                model: load_burnpack(
                    base_patch16_224::vision::Model::<Backend>::new(&device),
                    path,
                    precision,
                )?,
                device: device.clone(),
            }),
            "siglip2-so400m-patch14-384" => Box::new(So400mVision {
                model: load_burnpack(
                    so400m_patch14_384::vision::Model::<Backend>::new(&device),
                    path,
                    precision,
                )?,
                device: device.clone(),
            }),
            other => return Err(unsupported(other)),
        };
        let head = match head_path {
            Some(head_path) => Some(load_head(model_name, head_path, precision, device)?),
            None => None,
        };
        Ok(Self { inner, head })
    }

    /// Runs the vision encoder, returning raw pooled features and, when a head is
    /// loaded, the per-row aesthetic score computed in the same call.
    pub fn encode(
        &self,
        pixels: Vec<f32>,
        batch: usize,
        height: usize,
        width: usize,
    ) -> (Vec<f32>, Option<Vec<f32>>) {
        let pooled = self.inner.encode(pixels, batch, height, width);
        let scores = self.head.as_ref().map(|head| head.score(&pooled, batch));
        (pooled, scores)
    }
}

/// Loads the aesthetic head matching `model_name` (fp16, never Q8).
fn load_head(
    model_name: &str,
    path: &str,
    precision: &str,
    device: Device,
) -> Result<Box<dyn AestheticHeadArch>, String> {
    let head: Box<dyn AestheticHeadArch> = match model_name {
        "siglip2-base-patch16-224" => Box::new(BasePatch16Head {
            model: load_aesthetic_head(
                aesthetic_head::siglip2_base_patch16_224::Model::<Backend>::new(&device),
                path,
                precision,
            )?,
            device,
        }),
        "siglip2-so400m-patch14-384" => Box::new(So400mHead {
            model: load_aesthetic_head(
                aesthetic_head::siglip2_so400m_patch14_384::Model::<Backend>::new(&device),
                path,
                precision,
            )?,
            device,
        }),
        other => return Err(unsupported(other)),
    };
    Ok(head)
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

// --- aesthetic heads ------------------------------------------------------
//
// Each head takes raw pooled `[batch, dim]` features and returns `[batch]`
// scores (the head graph L2-normalizes its input internally).

struct BasePatch16Head {
    model: aesthetic_head::siglip2_base_patch16_224::Model<Backend>,
    device: Device,
}

impl AestheticHeadArch for BasePatch16Head {
    fn score(&self, pooled: &[f32], batch: usize) -> Vec<f32> {
        run_head(&self.device, batch, pooled, |input| {
            self.model.forward(input)
        })
    }
}

struct So400mHead {
    model: aesthetic_head::siglip2_so400m_patch14_384::Model<Backend>,
    device: Device,
}

impl AestheticHeadArch for So400mHead {
    fn score(&self, pooled: &[f32], batch: usize) -> Vec<f32> {
        run_head(&self.device, batch, pooled, |input| {
            self.model.forward(input)
        })
    }
}

/// Shapes raw pooled features into a `[batch, dim]` tensor, runs the head
/// `forward` (`[batch, dim] -> [batch]`), and flattens the scores.
fn run_head<F>(device: &Device, batch: usize, pooled: &[f32], forward: F) -> Vec<f32>
where
    F: FnOnce(Tensor<Backend, 2>) -> Tensor<Backend, 1>,
{
    let dim = pooled.len() / batch.max(1);
    let data = TensorData::new(pooled.to_vec(), [batch, dim]);
    let input = Tensor::<Backend, 2>::from_data(data, device);
    tensor_to_f32(forward(input))
}

fn tensor_to_f32<const D: usize>(tensor: Tensor<Backend, D>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("burn tensor data convertible to f32")
}
