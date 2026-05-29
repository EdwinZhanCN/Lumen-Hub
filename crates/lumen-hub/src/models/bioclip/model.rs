//! Burn-backed BioCLIP vision encoder.
//!
//! BioCLIP classification only needs the image encoder — the taxon catalog
//! holds precomputed text embeddings (see [`super::dataset`]). The text encoder
//! arch (`model_arch::bioclip2::text`) exists for future open-vocabulary tasks
//! but is not loaded here. Dispatch is keyed on the configured `model` name so
//! new CLIP variants register with one match arm.

use burn::tensor::{Tensor, TensorData};

use crate::backend::{Backend, Device};
use crate::model_arch::bioclip2;

trait BioClipVisionArch: Send + Sync {
    fn encode(&self, pixels: Vec<f32>, batch: usize, height: usize, width: usize) -> Vec<f32>;
}

/// BioCLIP vision encoder (architecture-erased).
pub struct BioClipVisionModel {
    inner: Box<dyn BioClipVisionArch>,
}

impl BioClipVisionModel {
    pub fn load(model_name: &str, path: &str, device: Device) -> Result<Self, String> {
        let inner: Box<dyn BioClipVisionArch> = match model_name {
            "bioclip-2" => Box::new(BioClip2Vision {
                model: bioclip2::vision::Model::<Backend>::from_file(path, &device),
                device,
            }),
            other => {
                return Err(format!(
                    "no Burn BioCLIP vision architecture registered for model `{other}`; \
                     add it under model_arch/ and register it in models::bioclip::model"
                ));
            }
        };
        Ok(Self { inner })
    }

    /// Encodes a batch of preprocessed NCHW pixel tensors, returning raw
    /// `[batch, embedding_dim]` image features flattened row-major.
    pub fn encode(&self, pixels: Vec<f32>, batch: usize, height: usize, width: usize) -> Vec<f32> {
        self.inner.encode(pixels, batch, height, width)
    }
}

struct BioClip2Vision {
    model: bioclip2::vision::Model<Backend>,
    device: Device,
}

impl BioClipVisionArch for BioClip2Vision {
    fn encode(&self, pixels: Vec<f32>, batch: usize, height: usize, width: usize) -> Vec<f32> {
        let data = TensorData::new(pixels, [batch, 3, height, width]);
        let input = Tensor::<Backend, 4>::from_data(data, &self.device);
        let output = self.model.forward(input);
        output
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("burn tensor data convertible to f32")
    }
}
