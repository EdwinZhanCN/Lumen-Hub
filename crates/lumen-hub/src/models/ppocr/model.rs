//! Burn-backed PP-OCR detection (DBNet) and recognition (SVTR/CRNN) encoders.
//!
//! Each OCR model repository is a distinct generated architecture under
//! [`crate::model_arch`]; this file dispatches the configured `model` to the
//! right one behind the [`PpocrDetectionArch`] / [`PpocrRecognitionArch`]
//! traits. Adding e.g. a future `pp-ocrv6` is: drop in the generated module +
//! one match arm here.

use burn::tensor::{Tensor, TensorData};

use crate::backend::{Backend, Device};
use crate::model_arch::pp_ocrv5;

/// A flat tensor result paired with its shape.
pub struct TensorOutput {
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
}

trait PpocrDetectionArch: Send + Sync {
    fn forward(&self, pixels: &[f32], height: usize, width: usize) -> TensorOutput;
}

trait PpocrRecognitionArch: Send + Sync {
    fn forward(&self, pixels: &[f32], c: usize, h: usize, w: usize) -> TensorOutput;
}

/// PP-OCR DBNet detection model (architecture-erased).
pub struct PpocrDetectionModel {
    inner: Box<dyn PpocrDetectionArch>,
}

impl PpocrDetectionModel {
    pub fn load(model_name: &str, path: &str, device: Device) -> Result<Self, String> {
        let inner: Box<dyn PpocrDetectionArch> = match model_name {
            "pp-ocrv5" => Box::new(PpOcrV5Detection {
                model: pp_ocrv5::detection::Model::<Backend>::from_file(path, &device),
                device,
            }),
            other => return Err(unsupported("detection", other)),
        };
        Ok(Self { inner })
    }

    /// Runs detection on a `[1, 3, H, W]` normalized buffer, returning the
    /// probability map `[1, 1, H, W]`.
    pub fn forward(&self, pixels: &[f32], height: usize, width: usize) -> TensorOutput {
        self.inner.forward(pixels, height, width)
    }
}

/// PP-OCR recognition model (architecture-erased).
pub struct PpocrRecognitionModel {
    inner: Box<dyn PpocrRecognitionArch>,
}

impl PpocrRecognitionModel {
    pub fn load(model_name: &str, path: &str, device: Device) -> Result<Self, String> {
        let inner: Box<dyn PpocrRecognitionArch> = match model_name {
            "pp-ocrv5" => Box::new(PpOcrV5Recognition {
                model: pp_ocrv5::recognition::Model::<Backend>::from_file(path, &device),
                device,
            }),
            other => return Err(unsupported("recognition", other)),
        };
        Ok(Self { inner })
    }

    /// Runs recognition on a `[1, C, H, W]` normalized crop, returning CTC
    /// logits `[1, seq_len, num_classes]`.
    pub fn forward(&self, pixels: &[f32], c: usize, h: usize, w: usize) -> TensorOutput {
        self.inner.forward(pixels, c, h, w)
    }
}

fn unsupported(component: &str, model_name: &str) -> String {
    format!(
        "no Burn PP-OCR {component} architecture registered for model `{model_name}`; \
         add it under model_arch/ and register it in models::ppocr::model"
    )
}

// --- pp-ocrv5 --------------------------------------------------------------

struct PpOcrV5Detection {
    model: pp_ocrv5::detection::Model<Backend>,
    device: Device,
}

impl PpocrDetectionArch for PpOcrV5Detection {
    fn forward(&self, pixels: &[f32], height: usize, width: usize) -> TensorOutput {
        let data = TensorData::new(pixels.to_vec(), [1, 3, height, width]);
        let input = Tensor::<Backend, 4>::from_data(data, &self.device);
        let output = self.model.forward(input);
        let shape = output.dims().to_vec();
        TensorOutput {
            values: tensor_to_f32(output),
            shape,
        }
    }
}

struct PpOcrV5Recognition {
    model: pp_ocrv5::recognition::Model<Backend>,
    device: Device,
}

impl PpocrRecognitionArch for PpOcrV5Recognition {
    fn forward(&self, pixels: &[f32], c: usize, h: usize, w: usize) -> TensorOutput {
        let data = TensorData::new(pixels.to_vec(), [1, c, h, w]);
        let input = Tensor::<Backend, 4>::from_data(data, &self.device);
        let output = self.model.forward(input);
        let shape = output.dims().to_vec();
        TensorOutput {
            values: tensor_to_f32(output),
            shape,
        }
    }
}

fn tensor_to_f32<const D: usize>(tensor: Tensor<Backend, D>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("burn tensor data convertible to f32")
}
