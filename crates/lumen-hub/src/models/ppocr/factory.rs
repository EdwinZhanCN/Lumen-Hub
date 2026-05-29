use std::{fs, path::PathBuf};

use lumen_schema::{ModelInfo, Runtime};

use super::model::{PpocrDetectionModel, PpocrRecognitionModel};
use crate::backend::Device;
use crate::service::{ServiceError, ServiceResult};

/// Resolves PP-OCR model artifacts using the Lumen model repository convention.
///
/// Repository layout under `{cache_dir}/{model_name}/`:
///
/// ```text
/// model_info.json                      # ModelInfo schema
/// ppocrv5_dict.txt                     # Character vocabulary
/// burn/detection.{precision}.bpk        # DBNet detection model
/// burn/recognition.{precision}.bpk      # SVTR/CRNN recognition model
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

    fn component_path_str(
        &self,
        model_name: &str,
        runtime: Runtime,
        component: &str,
        precision: &str,
    ) -> ServiceResult<String> {
        match runtime {
            Runtime::Burn => {}
        }
        let path = self
            .model_dir(model_name)
            .join("burn")
            .join(format!("{component}.{precision}.bpk"));
        if !path.exists() {
            return Err(ServiceError::InvalidArgument(format!(
                "PP-OCR `{component}` weights not found at {}",
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

    pub fn create_detection_model(
        &self,
        model_name: &str,
        runtime: Runtime,
        component: &str,
        precision: &str,
        device: &Device,
    ) -> ServiceResult<PpocrDetectionModel> {
        let path = self.component_path_str(model_name, runtime, component, precision)?;
        PpocrDetectionModel::load(model_name, &path, device.clone())
            .map_err(ServiceError::InvalidArgument)
    }

    pub fn create_recognition_model(
        &self,
        model_name: &str,
        runtime: Runtime,
        component: &str,
        precision: &str,
        device: &Device,
    ) -> ServiceResult<PpocrRecognitionModel> {
        let path = self.component_path_str(model_name, runtime, component, precision)?;
        PpocrRecognitionModel::load(model_name, &path, device.clone())
            .map_err(ServiceError::InvalidArgument)
    }

    /// Loads the character vocabulary from the model directory root.
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
