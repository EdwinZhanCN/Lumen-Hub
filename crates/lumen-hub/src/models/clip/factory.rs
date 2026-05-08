use std::{fs, path::PathBuf, sync::Arc};

use lumen_schema::{ModelInfo, Runtime};
use lumnn::{
    candle::node::CandleOnnxNode,
    core::{context::MLContext, node::MLNode},
    ort::node::OrtNode,
};

use crate::service::{ServiceError, ServiceResult};

/// Resolves model artifacts using the Lumen model repository convention.
///
/// Repository layout under `{cache_dir}/{model_name}/`:
///
/// ```text
/// model_info.json                  # ModelInfo schema
/// tokenizer.json                   # HuggingFace tokenizer (text models)
/// {runtime}/text.{precision}.{ext} # e.g. onnx/text.fp16.onnx
/// {runtime}/vision.{precision}.{ext}
/// ...
/// ```
///
/// The extension is derived from the runtime: `.onnx` for ONNX, `.rknn` for RKNN.
pub struct ClipModelFactory {
    cache_dir: String,
}

impl ClipModelFactory {
    pub fn new(cache_dir: impl Into<String>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
        }
    }

    /// Returns the root directory for a given model.
    pub fn model_dir(&self, model_name: &str) -> PathBuf {
        PathBuf::from(&self.cache_dir).join(model_name)
    }

    /// Loads and validates the `model_info.json` for a model.
    pub fn load_model_info(&self, model_name: &str) -> ServiceResult<ModelInfo> {
        let path = self.model_dir(model_name).join("model_info.json");
        let contents = fs::read_to_string(&path).map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "failed to read model_info.json at {}: {e}",
                path.display()
            ))
        })?;
        ModelInfo::from_json_str(&contents).map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "invalid model_info.json at {}: {e}",
                path.display()
            ))
        })
    }

    /// Resolves the path for a specific component artifact.
    ///
    /// Convention: `{cache_dir}/{model_name}/{runtime}/{component}.{precision}.{ext}`
    pub fn resolve_component_path(
        &self,
        model_name: &str,
        runtime: Runtime,
        component: &str,
        precision: &str,
    ) -> PathBuf {
        let runtime_dir = match runtime {
            Runtime::Onnx | Runtime::CandleOnnx => "onnx",
            Runtime::Rknn => "rknn",
        };
        let ext = match runtime {
            Runtime::Onnx | Runtime::CandleOnnx => "onnx",
            Runtime::Rknn => "rknn",
        };
        let filename = format!("{component}.{precision}.{ext}");
        self.model_dir(model_name).join(runtime_dir).join(filename)
    }

    /// Creates a model-forward node for a specific component.
    pub fn create_component(
        &self,
        model_name: &str,
        runtime: Runtime,
        component: &str,
        precision: &str,
        context: &Arc<MLContext>,
    ) -> ServiceResult<Box<dyn MLNode>> {
        let model_path = self.resolve_component_path(model_name, runtime, component, precision);
        let path_str = model_path.to_str().ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model path is not valid UTF-8: {}",
                model_path.display()
            ))
        })?;
        let name = format!("{model_name}_{component}");
        match runtime {
            Runtime::Onnx => OrtNode::new(context.as_ref(), path_str, name)
                .map(|node| Box::new(node) as Box<dyn MLNode>)
                .map_err(ServiceError::Internal),
            Runtime::CandleOnnx => CandleOnnxNode::new(context.as_ref(), path_str, name)
                .map(|node| Box::new(node) as Box<dyn MLNode>)
                .map_err(ServiceError::Internal),
            Runtime::Rknn => Err(ServiceError::InvalidArgument(
                "CLIP RKNN runtime is not implemented yet".to_owned(),
            )),
        }
    }

    /// Loads the HuggingFace tokenizer from the model directory root.
    pub fn load_tokenizer(&self, model_name: &str) -> ServiceResult<tokenizers::Tokenizer> {
        let tokenizer_path = self.model_dir(model_name).join("tokenizer.json");
        tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            ))
        })
    }
}
