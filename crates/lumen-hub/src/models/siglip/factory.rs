use std::{fs, path::PathBuf};

use lumen_schema::{ModelInfo, Runtime};
use tokenizers::{TruncationDirection, TruncationParams, TruncationStrategy};

use super::model::{SiglipTextModel, SiglipVisionModel};
use crate::backend::Device;
use crate::service::{ServiceError, ServiceResult};

/// Resolves SigLIP model artifacts using the Lumen model repository convention.
///
/// Repository layout under `{cache_dir}/{model_name}/`:
///
/// ```text
/// model_info.json                       # ModelInfo schema
/// tokenizer.json                        # HuggingFace tokenizer
/// burn/text.{precision}.bpk             # Burn text encoder weights
/// burn/vision.{precision}.bpk           # Burn vision encoder weights
/// ```
pub struct SiglipModelFactory {
    cache_dir: String,
}

impl SiglipModelFactory {
    pub fn new(cache_dir: impl Into<String>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
        }
    }

    pub fn model_dir(&self, model_name: &str) -> PathBuf {
        PathBuf::from(&self.cache_dir).join(model_name)
    }

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

    /// Convention: `{cache_dir}/{model_name}/burn/{component}.{precision}.bpk`.
    pub fn resolve_component_path(
        &self,
        model_name: &str,
        runtime: Runtime,
        component: &str,
        precision: &str,
    ) -> ServiceResult<PathBuf> {
        ensure_burn_runtime(runtime)?;
        Ok(self
            .model_dir(model_name)
            .join("burn")
            .join(format!("{component}.{precision}.bpk")))
    }

    fn component_path_str(
        &self,
        model_name: &str,
        runtime: Runtime,
        component: &str,
        precision: &str,
    ) -> ServiceResult<String> {
        let path = self.resolve_component_path(model_name, runtime, component, precision)?;
        if !path.exists() {
            return Err(ServiceError::InvalidArgument(format!(
                "SigLIP `{component}` weights not found at {}",
                path.display()
            )));
        }
        path.to_str().map(str::to_owned).ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model path is not valid UTF-8: {}",
                path.display()
            ))
        })
    }

    pub fn create_text_model(
        &self,
        model_name: &str,
        runtime: Runtime,
        precision: &str,
        device: &Device,
    ) -> ServiceResult<SiglipTextModel> {
        let path = self.component_path_str(model_name, runtime, "text", precision)?;
        SiglipTextModel::load(model_name, &path, precision, device.clone())
            .map_err(ServiceError::InvalidArgument)
    }

    pub fn create_vision_model(
        &self,
        model_name: &str,
        runtime: Runtime,
        precision: &str,
        device: &Device,
    ) -> ServiceResult<SiglipVisionModel> {
        let path = self.component_path_str(model_name, runtime, "vision", precision)?;
        SiglipVisionModel::load(model_name, &path, precision, device.clone())
            .map_err(ServiceError::InvalidArgument)
    }

    /// Loads the tokenizer and enforces truncation to the encoder's fixed
    /// sequence length so long inputs cannot overflow the positional table.
    pub fn load_tokenizer(
        &self,
        model_name: &str,
        seq_len: usize,
    ) -> ServiceResult<tokenizers::Tokenizer> {
        let tokenizer_path = self.model_dir(model_name).join("tokenizer.json");
        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            ))
        })?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: seq_len,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|e| {
                ServiceError::Internal(format!("failed to configure tokenizer truncation: {e}"))
            })?;
        Ok(tokenizer)
    }
}

fn ensure_burn_runtime(runtime: Runtime) -> ServiceResult<()> {
    match runtime {
        Runtime::Burn => Ok(()),
    }
}
