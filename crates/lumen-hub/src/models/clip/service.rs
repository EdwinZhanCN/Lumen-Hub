use std::{collections::BTreeMap, sync::Arc};

use lumen_schema::ServiceConfig;
use lumnn::core::{context::MLContext, node::MLNode, packet::MLPacketDataType};
use serde::Deserialize;

use super::factory::{BioClipModelFactory, ClipModelFactory};
use super::pipeline::build_embedding_pipeline;
use super::task::{
    BioClipClassifyTask, ClipImageEmbedTask, ClipImagePreprocessConfig, ClipTextEmbedTask,
};
use crate::service::{
    InferenceService, ServiceCapability, ServiceError, ServiceResult, TaskRegistry,
};

/// Parsed CLIP `task_metadata` from `model_info.json`.
#[derive(Debug, Clone, Deserialize)]
struct ClipTaskMetadata {
    #[allow(dead_code)]
    #[serde(default)]
    embedding_dim: Option<usize>,
    tasks: BTreeMap<String, ClipTaskConfig>,
}

/// Per-task configuration within CLIP `task_metadata`.
#[derive(Debug, Clone, Deserialize)]
struct ClipTaskConfig {
    /// Model component name: `"text"` or `"vision"`.
    component: String,
    /// ONNX model input node names in order (primary input first, then
    /// optional auxiliary inputs such as attention mask).
    input_names: Vec<String>,
    /// ONNX model output node name for the raw forward-pass embedding.
    output_name: String,
    /// Image preprocessing metadata for vision tasks.
    #[serde(default)]
    preprocess: Option<ClipImagePreprocessConfig>,
}

/// CLIP inference service.
///
/// Created from a `ServiceConfig` and `model_info.json`. Each model alias in
/// the config may expose one or both of `semantic_text_embed` / `semantic_image_embed` tasks.
pub struct ClipService {
    name: String,
    tasks: TaskRegistry,
    model_ids: Vec<String>,
    runtime: String,
}

impl ClipService {
    pub fn from_config(
        service_name: &str,
        service_config: &ServiceConfig,
        cache_dir: &str,
        context: Arc<MLContext>,
    ) -> ServiceResult<Self> {
        let factory = ClipModelFactory::new(cache_dir);
        let bioclip_factory = BioClipModelFactory::new(cache_dir);
        let mut tasks = TaskRegistry::new();
        let mut model_ids = Vec::new();
        let mut runtime_str = String::new();

        for (alias, model_config) in &service_config.models {
            let model_name = &model_config.model;
            let precision = model_config.precision.as_deref().unwrap_or("fp32");

            // 1. Load model_info.json
            let model_info = factory.load_model_info(model_name)?;

            // 2. Parse CLIP-specific task_metadata
            let clip_meta = parse_clip_metadata(model_name, &model_info)?;

            // 3. Record model info
            model_ids.push(model_name.to_owned());
            if runtime_str.is_empty() {
                runtime_str = model_config.runtime.as_str().to_owned();
            }

            if let Some(dataset) = &model_config.dataset {
                let (task_key, task_config) = clip_meta
                    .tasks
                    .iter()
                    .find(|(_, task_config)| task_config.component == "vision")
                    .ok_or_else(|| {
                        ServiceError::InvalidArgument(format!(
                            "BioCLIP model `{model_name}` requires a vision task in task_metadata"
                        ))
                    })?;
                let forward_node = bioclip_factory.create_vision_component(
                    model_name,
                    model_config.runtime,
                    precision,
                    &context,
                )?;
                let (input_dtype, preprocess) =
                    validate_vision_task(model_name, task_key, task_config, forward_node.as_ref())?;
                let pipeline = build_embedding_pipeline(
                    format!("{}_{}_{}", service_name, alias, task_key),
                    Arc::clone(&context),
                    forward_node,
                    &task_config.output_name,
                    "embedding",
                )
                .map_err(ServiceError::Internal)?;
                let dataset_paths = bioclip_factory.resolve_dataset_paths(model_name, dataset)?;
                let task = BioClipClassifyTask::new(
                    "bioclip_classify",
                    pipeline,
                    Arc::clone(&context),
                    model_name,
                    task_config.input_names.clone(),
                    input_dtype,
                    "embedding",
                    preprocess,
                    dataset,
                    dataset_paths.embeddings_path,
                    dataset_paths.labels_path,
                    dataset_paths.index_path,
                )?;
                tasks.register(task)?;
                continue;
            }

            // 4. For each task declared in task_metadata, build pipeline + handler
            for (task_key, task_config) in &clip_meta.tasks {
                match task_config.component.as_str() {
                    "text" => {
                        let forward_node = factory.create_component(
                            model_name,
                            model_config.runtime,
                            &task_config.component,
                            precision,
                            &context,
                        )?;

                        let tokenizer = factory.load_tokenizer(model_name)?;

                        let pipeline = build_embedding_pipeline(
                            format!("{}_{}_{}", service_name, alias, task_key),
                            Arc::clone(&context),
                            forward_node,
                            &task_config.output_name,
                            "embedding",
                        )
                        .map_err(ServiceError::Internal)?;

                        let task = ClipTextEmbedTask::new(
                            "semantic_text_embed",
                            pipeline,
                            Arc::clone(&context),
                            model_name,
                            task_config.input_names.clone(),
                            "embedding",
                            tokenizer,
                        );
                        tasks.register(task)?;
                    }
                    "vision" => {
                        let forward_node = factory.create_component(
                            model_name,
                            model_config.runtime,
                            &task_config.component,
                            precision,
                            &context,
                        )?;

                        let (input_dtype, preprocess) = validate_vision_task(
                            model_name,
                            task_key,
                            task_config,
                            forward_node.as_ref(),
                        )?;

                        let pipeline = build_embedding_pipeline(
                            format!("{}_{}_{}", service_name, alias, task_key),
                            Arc::clone(&context),
                            forward_node,
                            &task_config.output_name,
                            "embedding",
                        )
                        .map_err(ServiceError::Internal)?;

                        let task = ClipImageEmbedTask::new(
                            "semantic_image_embed",
                            pipeline,
                            Arc::clone(&context),
                            model_name,
                            task_config.input_names.clone(),
                            input_dtype,
                            "embedding",
                            preprocess,
                        )?;
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
            runtime: runtime_str,
        })
    }
}

fn parse_clip_metadata(
    model_name: &str,
    model_info: &lumen_schema::ModelInfo,
) -> ServiceResult<ClipTaskMetadata> {
    let raw_meta = serde_json::to_value(model_info.task_metadata.clone().unwrap_or_default())
        .map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "model `{model_name}` task_metadata serialization failed: {e}"
            ))
        })?;

    serde_json::from_value(raw_meta).map_err(|e| {
        ServiceError::InvalidArgument(format!(
            "model `{model_name}` task_metadata is not valid CLIP metadata: {e}"
        ))
    })
}

fn validate_vision_task(
    model_name: &str,
    task_key: &str,
    task_config: &ClipTaskConfig,
    forward_node: &dyn MLNode,
) -> ServiceResult<(MLPacketDataType, ClipImagePreprocessConfig)> {
    let input_name = task_config.input_names.first().ok_or_else(|| {
        ServiceError::InvalidArgument(format!(
            "model `{model_name}` image task `{task_key}` has no input names"
        ))
    })?;
    let input_desc = forward_node
        .input_descriptors()
        .get(input_name)
        .ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model `{model_name}` image task `{task_key}` references unknown input `{input_name}`"
            ))
        })?;
    let input_dtype = input_desc.dtype;
    let preprocess = task_config.preprocess.clone().ok_or_else(|| {
        ServiceError::InvalidArgument(format!(
            "model `{model_name}` image task `{task_key}` requires `task_metadata.tasks.{task_key}.preprocess`"
        ))
    })?;
    let expected_shape = preprocess.output_shape();
    if input_desc.shape.len() != expected_shape.len() {
        return Err(ServiceError::InvalidArgument(format!(
            "model `{model_name}` image task `{task_key}` preprocess rank {} does not match input `{input_name}` rank {}",
            expected_shape.len(),
            input_desc.shape.len()
        )));
    }
    if !input_desc.dynamic_batch && input_desc.shape != expected_shape {
        return Err(ServiceError::InvalidArgument(format!(
            "model `{model_name}` image task `{task_key}` preprocess shape {:?} does not match input `{input_name}` shape {:?}",
            expected_shape, input_desc.shape
        )));
    }
    if input_desc.dynamic_batch && input_desc.shape[1..] != expected_shape[1..] {
        return Err(ServiceError::InvalidArgument(format!(
            "model `{model_name}` image task `{task_key}` preprocess shape {:?} does not match dynamic-batch input `{input_name}` shape {:?}",
            expected_shape, input_desc.shape
        )));
    }

    Ok((input_dtype, preprocess))
}

impl InferenceService for ClipService {
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

    use super::ClipTaskMetadata;

    #[test]
    fn model_info_example_contains_required_clip_metadata() {
        let model_info =
            ModelInfo::from_json_str(include_str!("../../../tools/clip/model_info.example.json"))
                .unwrap();
        let raw_meta = serde_json::to_value(model_info.task_metadata.unwrap()).unwrap();
        let clip_meta: ClipTaskMetadata = serde_json::from_value(raw_meta).unwrap();

        assert_eq!(clip_meta.embedding_dim, None);
        let image_task = clip_meta.tasks.get("semantic_image_embed").unwrap();
        let preprocess = image_task.preprocess.as_ref().unwrap();
        assert_eq!(preprocess.output_shape(), vec![1, 3, 224, 224]);
    }
}
