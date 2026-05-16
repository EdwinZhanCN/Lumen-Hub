use std::{collections::BTreeMap, sync::Arc};

use lumen_schema::ServiceConfig;
use lumnn::core::{context::MLContext, node::MLNodeRef};
use serde::Deserialize;

use super::factory::PpocrModelFactory;
use super::nodes::{CtcDecodeNode, DBPostProcessNode};
use super::task::{PpocrDetConfig, PpocrRecConfig, PpocrTask};
use crate::service::{
    InferenceService, ServiceCapability, ServiceError, ServiceResult, TaskRegistry,
};

/// PP-OCR inference service.
///
/// Created from a `ServiceConfig` and `model_info.json`. Each model alias in
/// the config may expose an `ocr` task that performs end-to-end text detection
/// and recognition.
pub struct PpocrService {
    name: String,
    tasks: TaskRegistry,
    model_ids: Vec<String>,
    runtime: String,
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
}

impl PpocrService {
    pub fn from_config(
        service_name: &str,
        service_config: &ServiceConfig,
        cache_dir: &str,
        context: Arc<MLContext>,
    ) -> ServiceResult<Self> {
        let factory = PpocrModelFactory::new(cache_dir);
        let mut tasks = TaskRegistry::new();
        let mut model_ids = Vec::new();
        let mut runtime_str = String::new();

        for (alias, model_config) in &service_config.models {
            let model_name = &model_config.model;
            let precision = model_config.precision.as_deref().unwrap_or("fp32");

            // 1. Load model_info.json
            let model_info = factory.load_model_info(model_name)?;

            // 2. Parse PP-OCR task_metadata
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

            // 3. Record model info
            model_ids.push(model_name.to_owned());
            if runtime_str.is_empty() {
                runtime_str = model_config.runtime.as_str().to_owned();
            }

            // 4. For each task in task_metadata
            for (task_key, task_config) in &ppocr_meta.tasks {
                // --- Create detection ONNX node ---
                let det_node = factory.create_component(
                    model_name,
                    model_config.runtime,
                    &task_config.detection.component,
                    precision,
                    &context,
                )?;
                let det_node: MLNodeRef = Arc::from(det_node);

                // --- Create recognition ONNX node ---
                let rec_node = factory.create_component(
                    model_name,
                    model_config.runtime,
                    &task_config.recognition.component,
                    precision,
                    &context,
                )?;
                let rec_node: MLNodeRef = Arc::from(rec_node);

                // --- Create DB post-process node ---
                let db_node = DBPostProcessNode::new(
                    format!("{}_{}_db_postprocess", alias, task_key),
                    task_config.detection.thresh,
                    task_config.detection.box_thresh,
                    task_config.detection.unclip_ratio,
                );

                // --- Create CTC decode node ---
                let ctc_node = CtcDecodeNode::new(
                    format!("{}_{}_ctc_decode", alias, task_key),
                    task_config.recognition.blank_id,
                );

                // --- Load vocabulary ---
                let vocab = factory.load_vocab(
                    model_name,
                    &task_config.recognition.character_dict_path,
                    task_config.recognition.use_space_char,
                )?;

                // --- Create the task ---
                let task = PpocrTask::new(
                    format!("{}_{}", alias, task_key),
                    Arc::clone(&context),
                    model_name.clone(),
                    task_config.detection.clone(),
                    task_config.recognition.clone(),
                    det_node,
                    rec_node,
                    db_node,
                    ctc_node,
                    vocab,
                )?;

                tasks.register(task)?;
            }
        }

        Ok(Self {
            name: service_name.to_owned(),
            tasks,
            model_ids,
            runtime: runtime_str,
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
            .build_capability(&self.name, self.model_ids.clone(), &self.runtime)
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
        assert_eq!(ocr_task.detection.input_name, "x");
        assert_eq!(ocr_task.detection.output_name, "fetch_name_0");
        assert!((ocr_task.detection.thresh - 0.3).abs() < 1e-6);
        assert_eq!(ocr_task.recognition.component, "recognition");
        assert_eq!(ocr_task.recognition.output_name, "fetch_name_0");
        assert_eq!(ocr_task.recognition.blank_id, 0);
        assert_eq!(ocr_task.recognition.image_shape, [3, 48, 320]);
    }
}
