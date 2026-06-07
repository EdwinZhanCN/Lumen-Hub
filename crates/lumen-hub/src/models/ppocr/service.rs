use std::{collections::BTreeMap, sync::Arc};

use lumen_schema::ServiceConfig;
use serde::Deserialize;

use super::factory::PpocrModelFactory;
use super::task::{PpocrClsConfig, PpocrDetConfig, PpocrRecConfig, PpocrTask};
use crate::backend::{BACKEND_NAME, Device};
use crate::service::{
    InferenceService, ServiceCapability, ServiceError, ServiceResult, TaskRegistry,
};

/// PP-OCR inference service backed by Burn.
pub struct PpocrService {
    name: String,
    tasks: TaskRegistry,
    model_ids: Vec<String>,
}

/// Parsed PP-OCR `task_metadata` from `model_info.json`.
#[derive(Debug, Clone, Deserialize)]
struct PpocrTaskMetadata {
    tasks: BTreeMap<String, PpocrTaskConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct PpocrTaskConfig {
    detection: PpocrDetConfig,
    recognition: PpocrRecConfig,
    /// Optional text-line orientation classifier (server pack only).
    #[serde(default)]
    classification: Option<PpocrClsConfig>,
}

impl PpocrService {
    pub fn from_config(
        service_name: &str,
        service_config: &ServiceConfig,
        cache_dir: &str,
        device: Arc<Device>,
    ) -> ServiceResult<Self> {
        let factory = PpocrModelFactory::new(cache_dir);
        let mut tasks = TaskRegistry::new();
        let mut model_ids = Vec::new();

        for model_config in service_config.models.values() {
            let model_name = &model_config.model;
            let runtime = model_config.runtime;
            let precision = model_config.precision.as_deref().unwrap_or("fp32");

            let model_info = factory.load_model_info(model_name)?;
            let raw_meta =
                serde_json::to_value(model_info.task_metadata.clone().unwrap_or_default())
                    .map_err(|e| {
                        ServiceError::InvalidArgument(format!(
                            "model `{model_name}` task_metadata serialization failed: {e}"
                        ))
                    })?;
            let ppocr_meta: PpocrTaskMetadata = serde_json::from_value(raw_meta).map_err(|e| {
                ServiceError::InvalidArgument(format!(
                    "model `{model_name}` task_metadata is not valid PP-OCR metadata: {e}"
                ))
            })?;

            model_ids.push(model_name.to_owned());

            for (task_key, task_config) in &ppocr_meta.tasks {
                let det_model = factory.create_detection_model(
                    model_name,
                    runtime,
                    &task_config.detection.component,
                    precision,
                    &device,
                )?;
                let rec_model = factory.create_recognition_model(
                    model_name,
                    runtime,
                    &task_config.recognition.component,
                    precision,
                    &device,
                )?;
                let cls_model = task_config
                    .classification
                    .as_ref()
                    .map(|cls| {
                        factory.create_classification_model(
                            model_name,
                            runtime,
                            &cls.component,
                            precision,
                            &device,
                        )
                    })
                    .transpose()?;
                let vocab = factory.load_vocab(
                    model_name,
                    &task_config.recognition.character_dict_path,
                    task_config.recognition.use_space_char,
                )?;

                let task = PpocrTask::new(
                    task_key.clone(),
                    model_name.clone(),
                    task_config.detection.clone(),
                    task_config.recognition.clone(),
                    task_config.classification.clone(),
                    det_model,
                    rec_model,
                    cls_model,
                    vocab,
                )?;

                tasks.register(task)?;
            }
        }

        Ok(Self {
            name: service_name.to_owned(),
            tasks,
            model_ids,
        })
    }
}

impl InferenceService for PpocrService {
    fn name(&self) -> &str {
        &self.name
    }

    fn tasks(&self) -> &TaskRegistry {
        &self.tasks
    }

    fn capability(&self) -> ServiceCapability {
        self.tasks
            .build_capability(&self.name, self.model_ids.clone(), BACKEND_NAME)
    }
}

#[cfg(test)]
mod tests {
    use lumen_schema::ModelInfo;

    use super::PpocrTaskMetadata;

    #[test]
    fn model_info_example_parses_ppocr_metadata() {
        let model_info =
            ModelInfo::from_json_str(include_str!("../../../tools/ppocr/model_info.example.json"))
                .unwrap();
        let raw_meta = serde_json::to_value(model_info.task_metadata.unwrap()).unwrap();
        let ppocr_meta: PpocrTaskMetadata = serde_json::from_value(raw_meta).unwrap();

        let ocr_task = ppocr_meta.tasks.get("ocr").unwrap();
        assert_eq!(ocr_task.detection.component, "detection");
        assert_eq!(ocr_task.recognition.component, "recognition");
        assert_eq!(ocr_task.recognition.blank_id, 0);
        assert_eq!(ocr_task.recognition.image_shape, [3, 48, 320]);
    }
}
