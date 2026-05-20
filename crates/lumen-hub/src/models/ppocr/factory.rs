use std::{fs, path::PathBuf, sync::Arc};

use lumen_schema::{ModelInfo, Runtime};
#[cfg(feature = "mnn")]
use lumnn::mnn::MnnNode;
use lumnn::{
    core::{context::MLContext, node::MLNode},
    ort::node::OrtNode,
};

use crate::service::{ServiceError, ServiceResult};

/// Resolves PP-OCR model artifacts using the Lumen model repository convention.
///
/// Repository layout under `{cache_dir}/{model_name}/`:
///
/// ```text
/// model_info.json                      # ModelInfo schema
/// ppocrv5_dict.txt                     # Character vocabulary
/// {runtime}/detection.{precision}.onnx  # DBNet detection model
/// {runtime}/recognition.{precision}.onnx # SVTR/CRNN recognition model
/// ```
pub struct PpocrModelFactory {
    cache_dir: String,
}

impl PpocrModelFactory {
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
            Runtime::Mnn => "mnn",
            Runtime::MnnLlm => "mnn-llm",
        };
        let ext = match runtime {
            Runtime::Onnx | Runtime::CandleOnnx => "onnx",
            Runtime::Rknn => "rknn",
            Runtime::Mnn => "mnn",
            Runtime::MnnLlm => "json",
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
            #[cfg(feature = "mnn")]
            Runtime::Mnn => MnnNode::new(context.as_ref(), path_str, name)
                .map(|node| Box::new(node) as Box<dyn MLNode>)
                .map_err(ServiceError::Internal),
            #[cfg(not(feature = "mnn"))]
            Runtime::Mnn => Err(ServiceError::InvalidArgument(
                "PP-OCR MNN runtime is not enabled in this lumen-hub build".to_owned(),
            )),
            Runtime::CandleOnnx => Err(ServiceError::InvalidArgument(
                "PP-OCR Candle ONNX runtime is not implemented yet; use runtime=onnx".to_owned(),
            )),
            Runtime::Rknn => Err(ServiceError::InvalidArgument(
                "PP-OCR RKNN runtime is not implemented yet".to_owned(),
            )),
            Runtime::MnnLlm => Err(ServiceError::InvalidArgument(
                "PP-OCR MNN-LLM runtime is not supported".to_owned(),
            )),
        }
    }

    /// Loads the character vocabulary from the model directory root.
    ///
    /// The dictionary path is configured in `model_info.json`
    /// `task_metadata.tasks.ocr.recognition.character_dict_path`.
    /// Falls back to `ppocrv5_dict.txt` in the model root.
    pub fn load_vocab(
        &self,
        model_name: &str,
        dict_filename: &str,
        use_space_char: bool,
    ) -> ServiceResult<Vec<String>> {
        let vocab_path = self.model_dir(model_name).join(dict_filename);
        let contents = fs::read_to_string(&vocab_path).map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "failed to read character dictionary at {}: {e}",
                vocab_path.display()
            ))
        })?;
        let mut chars: Vec<String> = contents
            .lines()
            .map(|line| line.trim_end_matches('\r').to_owned())
            .filter(|line| !line.is_empty())
            .collect();
        if chars.is_empty() {
            return Err(ServiceError::InvalidArgument(format!(
                "character dictionary at {} is empty",
                vocab_path.display()
            )));
        }
        if use_space_char {
            chars.push(" ".to_owned());
        }
        Ok(chars)
    }
}
