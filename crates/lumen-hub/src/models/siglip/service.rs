use std::{collections::BTreeMap, sync::Arc};

use lumen_schema::ServiceConfig;
use serde::Deserialize;

use super::factory::SiglipModelFactory;
use super::task::{SiglipImageEmbedTask, SiglipImagePreprocessConfig, SiglipTextEmbedTask};
use crate::backend::{BACKEND_NAME, Device};
use crate::service::{
    InferenceService, ServiceCapability, ServiceError, ServiceResult, TaskRegistry,
};

/// Parsed SigLIP `task_metadata` from `model_info.json`.
#[derive(Debug, Clone, Deserialize)]
struct SiglipTaskMetadata {
    #[allow(dead_code)]
    #[serde(default)]
    embedding_dim: Option<usize>,
    tasks: BTreeMap<String, SiglipTaskConfig>,
}

/// Per-task configuration within SigLIP `task_metadata`.
#[derive(Debug, Clone, Deserialize)]
struct SiglipTaskConfig {
    /// Model component name: `"text"` or `"vision"`.
    component: String,
    /// Image preprocessing metadata for vision tasks.
    #[serde(default)]
    preprocess: Option<SiglipImagePreprocessConfig>,
}

/// SigLIP inference service backed by Burn.
///
/// Created from a `ServiceConfig` and `model_info.json`. Each model alias may
/// expose one or both of `semantic_text_embed` / `semantic_image_embed` tasks.
pub struct SiglipService {
    name: String,
    tasks: TaskRegistry,
    model_ids: Vec<String>,
}

impl SiglipService {
    pub fn from_config(
        service_name: &str,
        service_config: &ServiceConfig,
        cache_dir: &str,
        device: Arc<Device>,
    ) -> ServiceResult<Self> {
        let factory = SiglipModelFactory::new(cache_dir);
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
            let siglip_meta: SiglipTaskMetadata =
                serde_json::from_value(raw_meta).map_err(|e| {
                    ServiceError::InvalidArgument(format!(
                        "model `{model_name}` task_metadata is not valid SigLIP metadata: {e}"
                    ))
                })?;

            model_ids.push(model_name.to_owned());

            for (task_key, task_config) in &siglip_meta.tasks {
                match task_config.component.as_str() {
                    "text" => {
                        let model =
                            factory.create_text_model(model_name, runtime, precision, &device)?;
                        let tokenizer = factory.load_tokenizer(model_name, model.seq_len)?;
                        let task = SiglipTextEmbedTask::new(
                            "semantic_text_embed",
                            Arc::new(model),
                            model_name,
                            Arc::new(tokenizer),
                        );
                        tasks.register(task)?;
                    }
                    "vision" => {
                        let model =
                            factory.create_vision_model(model_name, runtime, precision, &device)?;
                        let preprocess = task_config.preprocess.clone().ok_or_else(|| {
                            ServiceError::InvalidArgument(format!(
                                "model `{model_name}` image task `{task_key}` requires \
                                 `task_metadata.tasks.{task_key}.preprocess`"
                            ))
                        })?;
                        let task = SiglipImageEmbedTask::new(
                            "semantic_image_embed",
                            Arc::new(model),
                            model_name,
                            preprocess,
                        );
                        tasks.register(task)?;
                    }
                    other => {
                        return Err(ServiceError::InvalidArgument(format!(
                            "model `{model_name}` has unknown component `{other}`; \
                             expected `text` or `vision`"
                        )));
                    }
                }
            }
        }

        Ok(Self {
            name: service_name.to_owned(),
            tasks,
            model_ids,
        })
    }
}

impl InferenceService for SiglipService {
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

    use super::SiglipTaskMetadata;

    #[test]
    fn model_info_example_contains_required_siglip_metadata() {
        let model_info = ModelInfo::from_json_str(include_str!(
            "../../../tools/siglip/model_info.example.json"
        ))
        .unwrap();
        let raw_meta = serde_json::to_value(model_info.task_metadata.unwrap()).unwrap();
        let siglip_meta: SiglipTaskMetadata = serde_json::from_value(raw_meta).unwrap();

        let image_task = siglip_meta.tasks.get("semantic_image_embed").unwrap();
        let preprocess = image_task.preprocess.as_ref().unwrap();
        assert_eq!(preprocess.output_shape(), vec![1, 3, 224, 224]);
    }
}
