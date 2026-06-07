use std::{fs, path::PathBuf};

use lumen_schema::{ModelInfo, Runtime};

use super::model::{InsightFaceDetectionModel, InsightFaceRecognitionModel};
use crate::backend::Device;
use crate::service::{ServiceError, ServiceResult};

/// Resolves InsightFace model artifacts using the Lumen model repository
/// convention: `{cache_dir}/{model_name}/burn/{component}.{precision}.bpk`.
pub struct InsightFaceModelFactory {
    cache_dir: String,
}

impl InsightFaceModelFactory {
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
                "InsightFace `{component}` weights not found at {}",
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
    ) -> ServiceResult<InsightFaceDetectionModel> {
        let path = self.component_path_str(model_name, runtime, component, precision)?;
        InsightFaceDetectionModel::load(model_name, &path, precision, device.clone())
            .map_err(ServiceError::InvalidArgument)
    }

    pub fn create_recognition_model(
        &self,
        model_name: &str,
        runtime: Runtime,
        component: &str,
        precision: &str,
        device: &Device,
    ) -> ServiceResult<InsightFaceRecognitionModel> {
        let path = self.component_path_str(model_name, runtime, component, precision)?;
        InsightFaceRecognitionModel::load(model_name, &path, precision, device.clone())
            .map_err(ServiceError::InvalidArgument)
    }
}
