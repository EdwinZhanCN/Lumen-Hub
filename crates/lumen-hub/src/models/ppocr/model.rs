//! Burn-backed PP-OCR detection (DBNet) and recognition (SVTR/CRNN) encoders.
//!
//! Each OCR model repository is a distinct generated architecture under
//! [`crate::model_arch`]; this file dispatches the configured `model` to the
//! right one behind the [`PpocrDetectionArch`] / [`PpocrRecognitionArch`]
//! traits. Adding e.g. a future `pp-ocrv6` is: drop in the generated module +
//! one match arm here.

use burn::tensor::{Tensor, TensorData};

use crate::backend::{Backend, Device};
use crate::model_arch::load_burnpack;
use crate::model_arch::{pp_ocrv5, pp_ocrv5_server};

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

trait PpocrClassificationArch: Send + Sync {
    fn forward(&self, pixels: &[f32], c: usize, h: usize, w: usize) -> TensorOutput;
}

/// PP-OCR DBNet detection model (architecture-erased).
pub struct PpocrDetectionModel {
    inner: Box<dyn PpocrDetectionArch>,
}

impl PpocrDetectionModel {
    pub fn load(
        model_name: &str,
        path: &str,
        precision: &str,
        device: Device,
    ) -> Result<Self, String> {
        let inner: Box<dyn PpocrDetectionArch> = match model_name {
            "pp-ocrv5" => Box::new(PpOcrV5Detection {
                model: load_burnpack(
                    pp_ocrv5::detection::Model::<Backend>::new(&device),
                    path,
                    precision,
                )?,
                device,
            }),
            "pp-ocrv5-server" => Box::new(PpOcrV5ServerDetection {
                model: load_burnpack(
                    pp_ocrv5_server::detection::Model::<Backend>::new(&device),
                    path,
                    precision,
                )?,
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
    pub fn load(
        model_name: &str,
        path: &str,
        precision: &str,
        device: Device,
    ) -> Result<Self, String> {
        let inner: Box<dyn PpocrRecognitionArch> = match model_name {
            "pp-ocrv5" => Box::new(PpOcrV5Recognition {
                model: load_burnpack(
                    pp_ocrv5::recognition::Model::<Backend>::new(&device),
                    path,
                    precision,
                )?,
                device,
            }),
            "pp-ocrv5-server" => Box::new(PpOcrV5ServerRecognition {
                model: load_burnpack(
                    pp_ocrv5_server::recognition::Model::<Backend>::new(&device),
                    path,
                    precision,
                )?,
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

/// PP-OCR text-line orientation classification model (architecture-erased).
///
/// Only the server pack ships a classifier; it predicts whether a detected text
/// crop is upright (0°) or upside-down (180°) so the task can rotate it before
/// recognition.
pub struct PpocrClassificationModel {
    inner: Box<dyn PpocrClassificationArch>,
}

impl PpocrClassificationModel {
    pub fn load(
        model_name: &str,
        path: &str,
        precision: &str,
        device: Device,
    ) -> Result<Self, String> {
        let inner: Box<dyn PpocrClassificationArch> = match model_name {
            "pp-ocrv5-server" => Box::new(PpOcrV5ServerClassification {
                model: load_burnpack(
                    pp_ocrv5_server::classification::Model::<Backend>::new(&device),
                    path,
                    precision,
                )?,
                device,
            }),
            other => return Err(unsupported("classification", other)),
        };
        Ok(Self { inner })
    }

    /// Runs orientation classification on a `[1, C, H, W]` normalized crop,
    /// returning class scores `[1, num_classes]` (typically `[p(0°), p(180°)]`).
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

// --- pp-ocrv5-server -------------------------------------------------------

struct PpOcrV5ServerDetection {
    model: pp_ocrv5_server::detection::Model<Backend>,
    device: Device,
}

impl PpocrDetectionArch for PpOcrV5ServerDetection {
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

struct PpOcrV5ServerRecognition {
    model: pp_ocrv5_server::recognition::Model<Backend>,
    device: Device,
}

impl PpocrRecognitionArch for PpOcrV5ServerRecognition {
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

struct PpOcrV5ServerClassification {
    model: pp_ocrv5_server::classification::Model<Backend>,
    device: Device,
}

impl PpocrClassificationArch for PpOcrV5ServerClassification {
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
