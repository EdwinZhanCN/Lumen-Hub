use std::{collections::BTreeMap, sync::Arc};

use lumen_schema::ServiceConfig;
use lumnn::core::{context::MLContext, node::MLNodeRef};
use serde::Deserialize;

use super::factory::InsightFaceModelFactory;
use super::metadata::pack_spec;
use super::task::{InsightFaceComponentConfig, InsightFaceTask};
use crate::service::{
    InferenceService, ServiceCapability, ServiceError, ServiceResult, TaskRegistry,
};

pub struct InsightFaceService {
    name: String,
    tasks: TaskRegistry,
    model_ids: Vec<String>,
    runtime: String,
}

#[derive(Debug, Clone, Deserialize)]
struct InsightFaceTaskMetadata {
    tasks: BTreeMap<String, InsightFaceTaskConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct InsightFaceTaskConfig {
    #[serde(default = "default_pack")]
    pack: String,
    detection: InsightFaceComponentConfig,
    recognition: InsightFaceComponentConfig,
}

fn default_pack() -> String {
    "antelopev2".to_owned()
}

impl InsightFaceService {
    pub fn from_config(
        service_name: &str,
        service_config: &ServiceConfig,
        cache_dir: &str,
        context: Arc<MLContext>,
    ) -> ServiceResult<Self> {
        let factory = InsightFaceModelFactory::new(cache_dir);
        let mut tasks = TaskRegistry::new();
        let mut model_ids = Vec::new();
        let mut runtime_str = String::new();

        for (_alias, model_config) in &service_config.models {
            let model_name = &model_config.model;
            let precision = model_config.precision.as_deref().unwrap_or("fp32");
            let model_info = factory.load_model_info(model_name)?;
            let raw_meta =
                serde_json::to_value(model_info.task_metadata.clone().unwrap_or_default())
                    .map_err(|e| {
                        ServiceError::InvalidArgument(format!(
                            "model `{model_name}` task_metadata serialization failed: {e}"
                        ))
                    })?;
            let insightface_meta: InsightFaceTaskMetadata = serde_json::from_value(raw_meta)
                .map_err(|e| {
                    ServiceError::InvalidArgument(format!(
                        "model `{model_name}` task_metadata is not valid InsightFace metadata: {e}"
                    ))
                })?;

            model_ids.push(model_name.to_owned());
            if runtime_str.is_empty() {
                runtime_str = model_config.runtime.as_str().to_owned();
            }

            for (task_key, task_config) in &insightface_meta.tasks {
                let pack = pack_spec(&task_config.pack).ok_or_else(|| {
                    ServiceError::InvalidArgument(format!(
                        "model `{model_name}` task `{task_key}` references unsupported InsightFace pack `{}`",
                        task_config.pack
                    ))
                })?;

                let det_node = factory.create_component(
                    model_name,
                    model_config.runtime,
                    &task_config.detection.component,
                    precision,
                    &context,
                )?;
                let det_node: MLNodeRef = Arc::from(det_node);

                let rec_node = factory.create_component(
                    model_name,
                    model_config.runtime,
                    &task_config.recognition.component,
                    precision,
                    &context,
                )?;
                let rec_node: MLNodeRef = Arc::from(rec_node);

                let task = InsightFaceTask::new(
                    task_key.clone(),
                    Arc::clone(&context),
                    model_name.clone(),
                    pack,
                    task_config.detection.clone(),
                    task_config.recognition.clone(),
                    det_node,
                    rec_node,
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

impl InferenceService for InsightFaceService {
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

    use super::InsightFaceTaskMetadata;

    #[test]
    fn model_info_example_parses_insightface_metadata() {
        let model_info = ModelInfo::from_json_str(include_str!(
            "../../../tools/insightface/model_info.example.json"
        ))
        .unwrap();
        let raw_meta = serde_json::to_value(model_info.task_metadata.unwrap()).unwrap();
        let meta: InsightFaceTaskMetadata = serde_json::from_value(raw_meta).unwrap();

        let task = meta.tasks.get("face_recognition").unwrap();
        assert_eq!(task.pack, "antelopev2");
        assert_eq!(task.detection.component, "detection");
        assert_eq!(task.detection.input_name, "input.1");
        assert_eq!(task.detection.output_names.len(), 9);
        assert_eq!(task.recognition.component, "recognition");
        assert_eq!(task.recognition.output_name.as_deref(), Some("1333"));
    }
}
