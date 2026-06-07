//! Burn-backed InsightFace SCRFD detection and ArcFace recognition models.
//!
//! Each InsightFace pack (antelopev2, buffalo_l, ...) ships its own SCRFD
//! detector and ArcFace recognizer graphs, so each is a distinct generated
//! architecture under [`crate::model_arch`]. This file dispatches the configured
//! `model` to the right one behind the [`InsightFaceDetectionArch`] /
//! [`InsightFaceRecognitionArch`] traits; the shared preprocessing/anchor
//! decoding lives in the task layer and is driven by
//! [`super::metadata::pack_spec`]. Adding `buffalo_l` is: drop in the generated
//! modules + one match arm here.

use burn::tensor::{Tensor, TensorData};

use crate::backend::{Backend, Device};
use crate::model_arch::antelopev2;
use crate::model_arch::load_burnpack;

/// A flat tensor result paired with its shape.
pub struct TensorOutput {
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
}

trait InsightFaceDetectionArch: Send + Sync {
    fn forward(&self, pixels: &[f32], height: usize, width: usize) -> Vec<TensorOutput>;
}

trait InsightFaceRecognitionArch: Send + Sync {
    fn forward(&self, pixels: &[f32], height: usize, width: usize) -> Vec<f32>;
}

/// SCRFD face detection model (architecture-erased).
///
/// Produces nine output heads in ONNX-declared order:
/// `[score_s8, score_s16, score_s32, bbox_s8, bbox_s16, bbox_s32,
///   kps_s8, kps_s16, kps_s32]`.
pub struct InsightFaceDetectionModel {
    inner: Box<dyn InsightFaceDetectionArch>,
}

impl InsightFaceDetectionModel {
    pub fn load(
        model_name: &str,
        path: &str,
        precision: &str,
        device: Device,
    ) -> Result<Self, String> {
        let inner: Box<dyn InsightFaceDetectionArch> = match model_name {
            "antelopev2" => Box::new(AntelopeV2Detection {
                model: load_burnpack(
                    antelopev2::detection::Model::<Backend>::new(&device),
                    path,
                    precision,
                )?,
                device,
            }),
            other => return Err(unsupported("detection", other)),
        };
        Ok(Self { inner })
    }

    pub fn forward(&self, pixels: &[f32], height: usize, width: usize) -> Vec<TensorOutput> {
        self.inner.forward(pixels, height, width)
    }
}

/// ArcFace recognition (embedding) model (architecture-erased).
pub struct InsightFaceRecognitionModel {
    inner: Box<dyn InsightFaceRecognitionArch>,
}

impl InsightFaceRecognitionModel {
    pub fn load(
        model_name: &str,
        path: &str,
        precision: &str,
        device: Device,
    ) -> Result<Self, String> {
        let inner: Box<dyn InsightFaceRecognitionArch> = match model_name {
            "antelopev2" => Box::new(AntelopeV2Recognition {
                model: load_burnpack(
                    antelopev2::recognition::Model::<Backend>::new(&device),
                    path,
                    precision,
                )?,
                device,
            }),
            other => return Err(unsupported("recognition", other)),
        };
        Ok(Self { inner })
    }

    /// Runs recognition on a `[1, 3, H, W]` aligned face, returning the raw
    /// embedding row.
    pub fn forward(&self, pixels: &[f32], height: usize, width: usize) -> Vec<f32> {
        self.inner.forward(pixels, height, width)
    }
}

fn unsupported(component: &str, model_name: &str) -> String {
    format!(
        "no Burn InsightFace {component} architecture registered for model `{model_name}` \
         (e.g. buffalo_l is not bundled yet); add it under model_arch/ and register it in \
         models::insightface::model"
    )
}

// --- antelopev2 ------------------------------------------------------------

struct AntelopeV2Detection {
    model: antelopev2::detection::Model<Backend>,
    device: Device,
}

impl InsightFaceDetectionArch for AntelopeV2Detection {
    fn forward(&self, pixels: &[f32], height: usize, width: usize) -> Vec<TensorOutput> {
        let data = TensorData::new(pixels.to_vec(), [1, 3, height, width]);
        let input = Tensor::<Backend, 4>::from_data(data, &self.device);
        let (o0, o1, o2, o3, o4, o5, o6, o7, o8) = self.model.forward(input);
        vec![
            tensor_output(o0),
            tensor_output(o1),
            tensor_output(o2),
            tensor_output(o3),
            tensor_output(o4),
            tensor_output(o5),
            tensor_output(o6),
            tensor_output(o7),
            tensor_output(o8),
        ]
    }
}

struct AntelopeV2Recognition {
    model: antelopev2::recognition::Model<Backend>,
    device: Device,
}

impl InsightFaceRecognitionArch for AntelopeV2Recognition {
    fn forward(&self, pixels: &[f32], height: usize, width: usize) -> Vec<f32> {
        let data = TensorData::new(pixels.to_vec(), [1, 3, height, width]);
        let input = Tensor::<Backend, 4>::from_data(data, &self.device);
        tensor_to_f32(self.model.forward(input))
    }
}

fn tensor_output(tensor: Tensor<Backend, 2>) -> TensorOutput {
    let shape = tensor.dims().to_vec();
    TensorOutput {
        values: tensor_to_f32(tensor),
        shape,
    }
}

fn tensor_to_f32<const D: usize>(tensor: Tensor<Backend, D>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("burn tensor data convertible to f32")
}
