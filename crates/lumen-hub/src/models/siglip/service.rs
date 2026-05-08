use std::{collections::BTreeMap, sync::Arc};

use lumen_schema::ServiceConfig;
use lumnn::core::context::MLContext;
use serde::Deserialize;

use super::factory::SiglipModelFactory;
use super::pipeline::build_embedding_pipeline;
use super::task::{SiglipImageEmbedTask, SiglipImagePreprocessConfig, SiglipTextEmbedTask};
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
    /// ONNX model input node names in order.
    input_names: Vec<String>,
    /// ONNX model output node name for the raw forward-pass embedding.
    output_name: String,
    /// Image preprocessing metadata for vision tasks.
    #[serde(default)]
    preprocess: Option<SiglipImagePreprocessConfig>,
}

/// SigLIP inference service.
///
/// Created from a `ServiceConfig` and `model_info.json`. Each model alias may
/// expose one or both of `semantic_text_embed` / `semantic_image_embed` tasks.
pub struct SiglipService {
    name: String,
    tasks: TaskRegistry,
    model_ids: Vec<String>,
    runtime: String,
}

impl SiglipService {
    pub fn from_config(
        service_name: &str,
        service_config: &ServiceConfig,
        cache_dir: &str,
        context: Arc<MLContext>,
    ) -> ServiceResult<Self> {
        let factory = SiglipModelFactory::new(cache_dir);
        let mut tasks = TaskRegistry::new();
        let mut model_ids = Vec::new();
        let mut runtime_str = String::new();

        for (alias, model_config) in &service_config.models {
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
            let siglip_meta: SiglipTaskMetadata =
                serde_json::from_value(raw_meta).map_err(|e| {
                    ServiceError::InvalidArgument(format!(
                        "model `{model_name}` task_metadata is not valid SigLIP metadata: {e}"
                    ))
                })?;

            model_ids.push(model_name.to_owned());
            if runtime_str.is_empty() {
                runtime_str = model_config.runtime.as_str().to_owned();
            }

            for (task_key, task_config) in &siglip_meta.tasks {
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
                        let task = SiglipTextEmbedTask::new(
                            format!("{}_{}", alias, task_key),
                            pipeline,
                            Arc::clone(&context),
                            model_name,
                            task_config.input_names.clone(),
                            "embedding",
                            tokenizer,
                        )?;
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
                                    "model `{model_name}` image task `{task_key}` references \
                                         unknown ONNX input `{input_name}`"
                                ))
                            })?;
                        let input_dtype = input_desc.dtype;
                        let preprocess = task_config.preprocess.clone().ok_or_else(|| {
                            ServiceError::InvalidArgument(format!(
                                "model `{model_name}` image task `{task_key}` requires \
                                 `task_metadata.tasks.{task_key}.preprocess`"
                            ))
                        })?;
                        validate_preprocess_shape(
                            model_name,
                            task_key,
                            input_name,
                            &preprocess,
                            input_desc,
                        )?;

                        let pipeline = build_embedding_pipeline(
                            format!("{}_{}_{}", service_name, alias, task_key),
                            Arc::clone(&context),
                            forward_node,
                            &task_config.output_name,
                            "embedding",
                        )
                        .map_err(ServiceError::Internal)?;
                        let task = SiglipImageEmbedTask::new(
                            format!("{}_{}", alias, task_key),
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

impl InferenceService for SiglipService {
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

fn validate_preprocess_shape(
    model_name: &str,
    task_key: &str,
    input_name: &str,
    preprocess: &SiglipImagePreprocessConfig,
    input_desc: &lumnn::core::packet::MLPacketDescriptor,
) -> ServiceResult<()> {
    let expected_shape = preprocess.output_shape();
    if input_desc.shape.len() != expected_shape.len() {
        return Err(ServiceError::InvalidArgument(format!(
            "model `{model_name}` image task `{task_key}` preprocess rank {} does not match \
             ONNX input `{input_name}` rank {}",
            expected_shape.len(),
            input_desc.shape.len()
        )));
    }
    if !input_desc.dynamic_batch && input_desc.shape != expected_shape {
        return Err(ServiceError::InvalidArgument(format!(
            "model `{model_name}` image task `{task_key}` preprocess shape {:?} does not match \
             ONNX input `{input_name}` shape {:?}",
            expected_shape, input_desc.shape
        )));
    }
    if input_desc.dynamic_batch && input_desc.shape[1..] != expected_shape[1..] {
        return Err(ServiceError::InvalidArgument(format!(
            "model `{model_name}` image task `{task_key}` preprocess shape {:?} does not match \
             ONNX dynamic-batch input `{input_name}` shape {:?}",
            expected_shape, input_desc.shape
        )));
    }
    Ok(())
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

        assert_eq!(siglip_meta.embedding_dim, Some(768));
        let image_task = siglip_meta.tasks.get("semantic_image_embed").unwrap();
        let preprocess = image_task.preprocess.as_ref().unwrap();
        assert_eq!(preprocess.output_shape(), vec![1, 3, 224, 224]);
    }
}
