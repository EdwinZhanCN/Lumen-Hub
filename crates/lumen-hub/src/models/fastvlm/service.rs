use std::sync::Arc;

use lumen_schema::{RuntimeSpec, ServiceConfig};
use lumnn::core::{context::MLContext, node::MLNodeRef};

use super::{
    factory::FastVlmModelFactory,
    metadata::METADATA,
    task::{FASTVLM_PREPROCESS_ID, FastVlmDecodeTask, FastVlmEmbedsTask},
};
use crate::service::{
    InferenceService, ServiceCapability, ServiceError, ServiceResult, TaskRegistry,
};

const REQUIRED_COMPONENTS: [&str; 3] = ["vision", "embed", "decoder"];

pub struct FastVlmService {
    name: String,
    tasks: TaskRegistry,
    model_ids: Vec<String>,
    runtime: String,
}

impl FastVlmService {
    pub fn from_config(
        service_name: &str,
        service_config: &ServiceConfig,
        cache_dir: &str,
        context: Arc<MLContext>,
    ) -> ServiceResult<Self> {
        let factory = FastVlmModelFactory::new(cache_dir);
        let mut tasks = TaskRegistry::new();
        let mut model_ids = Vec::new();
        let mut runtime_str = String::new();

        for (_alias, model_config) in &service_config.models {
            let model_name = &model_config.model;
            let precision = model_config.precision.as_deref().unwrap_or("fp32");
            let model_info = factory.load_model_info(model_name)?;
            let runtime_key = match model_config.runtime {
                lumen_schema::Runtime::Onnx | lumen_schema::Runtime::CandleOnnx => "onnx",
                lumen_schema::Runtime::Rknn => "rknn",
                lumen_schema::Runtime::Mnn => "mnn",
            };
            let runtime_spec = model_info
                .runtimes
                .as_map()
                .get(runtime_key)
                .ok_or_else(|| {
                    ServiceError::InvalidArgument(format!(
                        "model `{model_name}` does not declare runtime `{runtime_key}`"
                    ))
                })?;
            validate_runtime_components(model_name, runtime_spec)?;

            model_ids.push(model_name.to_owned());
            if runtime_str.is_empty() {
                runtime_str = model_config.runtime.as_str().to_owned();
            }

            let vision_node = factory.create_component(
                model_name,
                model_config.runtime,
                "vision",
                precision,
                &context,
            )?;
            let embed_node = factory.create_component(
                model_name,
                model_config.runtime,
                "embed",
                precision,
                &context,
            )?;
            let decoder_node = factory.create_component(
                model_name,
                model_config.runtime,
                "decoder",
                precision,
                &context,
            )?;

            validate_component_io(
                model_name,
                vision_node.as_ref(),
                embed_node.as_ref(),
                decoder_node.as_ref(),
            )?;

            let vision_node: MLNodeRef = Arc::from(vision_node);
            let embed_node: MLNodeRef = Arc::from(embed_node);
            let decoder_node: MLNodeRef = Arc::from(decoder_node);

            let task = FastVlmEmbedsTask::new(
                "vlm_embeds",
                Arc::clone(&context),
                model_name,
                &model_info.version,
                vision_node,
                Arc::clone(&embed_node),
                factory.load_tokenizer(model_name)?,
            )?;
            tasks.register(task)?;
            let task = FastVlmDecodeTask::new(
                "vlm_decode",
                Arc::clone(&context),
                model_name,
                &model_info.version,
                embed_node,
                decoder_node,
                factory.load_tokenizer(model_name)?,
            )?;
            tasks.register(task)?;
        }

        Ok(Self {
            name: service_name.to_owned(),
            tasks,
            model_ids,
            runtime: runtime_str,
        })
    }
}

impl InferenceService for FastVlmService {
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

fn validate_runtime_components(model_name: &str, runtime: &RuntimeSpec) -> ServiceResult<()> {
    for component in REQUIRED_COMPONENTS {
        if !runtime
            .components
            .iter()
            .any(|declared| declared == component)
        {
            return Err(ServiceError::InvalidArgument(format!(
                "model `{model_name}` FastVLM runtime must declare component `{component}`"
            )));
        }
    }
    Ok(())
}

fn validate_component_io(
    model_name: &str,
    vision_node: &dyn lumnn::core::node::MLNode,
    embed_node: &dyn lumnn::core::node::MLNode,
    decoder_node: &dyn lumnn::core::node::MLNode,
) -> ServiceResult<()> {
    let vision_input = vision_node
        .input_descriptors()
        .get(METADATA.vision_input_name)
        .ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model `{model_name}` vision component missing input `{}`",
                METADATA.vision_input_name
            ))
        })?;
    if !vision_input_matches_expected_shape(vision_input) {
        return Err(ServiceError::InvalidArgument(format!(
            "model `{model_name}` vision input `{}` must be [1,3,448,448], dynamic batch [B,3,448,448], or dynamic spatial [B,3,H,W], got shape {:?} dynamic_batch={} dynamic_axes={:?}",
            METADATA.vision_input_name,
            vision_input.shape,
            vision_input.dynamic_batch,
            vision_input.dynamic_axes
        )));
    }

    vision_node
        .output_descriptors()
        .get(METADATA.vision_output_name)
        .ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model `{model_name}` vision component missing output `{}`",
                METADATA.vision_output_name
            ))
        })?;
    embed_node
        .input_descriptors()
        .get(METADATA.embed_input_name)
        .ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model `{model_name}` embed component missing input `{}`",
                METADATA.embed_input_name
            ))
        })?;
    embed_node
        .output_descriptors()
        .get(METADATA.embed_output_name)
        .ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model `{model_name}` embed component missing output `{}`",
                METADATA.embed_output_name
            ))
        })?;

    let decoder_inputs_embeds = decoder_node
        .input_descriptors()
        .get(METADATA.decoder_input_name)
        .ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model `{model_name}` decoder component missing input `{}`",
                METADATA.decoder_input_name
            ))
        })?;
    if decoder_inputs_embeds.shape.len() != 3
        || decoder_inputs_embeds.shape[2] != METADATA.hidden_size
    {
        return Err(ServiceError::InvalidArgument(format!(
            "model `{model_name}` decoder input `{}` must be [B,S,{}], got {:?}",
            METADATA.decoder_input_name, METADATA.hidden_size, decoder_inputs_embeds.shape
        )));
    }
    decoder_node
        .input_descriptors()
        .get(METADATA.decoder_attention_mask_name)
        .ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model `{model_name}` decoder component missing input `{}`",
                METADATA.decoder_attention_mask_name
            ))
        })?;
    decoder_node
        .input_descriptors()
        .get(METADATA.decoder_position_ids_name)
        .ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model `{model_name}` decoder component missing input `{}`",
                METADATA.decoder_position_ids_name
            ))
        })?;
    decoder_node
        .output_descriptors()
        .get(METADATA.decoder_output_name)
        .ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model `{model_name}` decoder component missing output `{}`",
                METADATA.decoder_output_name
            ))
        })?;

    let past_count = decoder_node
        .input_descriptors()
        .keys()
        .filter(|name| name.starts_with("past_key_values."))
        .count();
    let present_count = decoder_node
        .output_descriptors()
        .keys()
        .filter(|name| name.starts_with("present."))
        .count();
    let expected_kv_count = METADATA.kv_cache.num_layers * 2;
    if past_count != expected_kv_count {
        return Err(ServiceError::InvalidArgument(format!(
            "model `{model_name}` decoder component must expose {expected_kv_count} past_kv inputs, got {past_count}"
        )));
    }
    if present_count != expected_kv_count {
        return Err(ServiceError::InvalidArgument(format!(
            "model `{model_name}` decoder component must expose {expected_kv_count} present outputs, got {present_count}"
        )));
    }

    let _ = FASTVLM_PREPROCESS_ID;
    Ok(())
}

fn vision_input_matches_expected_shape(
    descriptor: &lumnn::core::packet::MLPacketDescriptor,
) -> bool {
    if descriptor.shape.len() != 4 || descriptor.shape[1] != 3 {
        return false;
    }

    let exact = descriptor.shape == [1, 3, 448, 448];
    let dynamic_batch_exact_spatial =
        descriptor.shape[2] == 448 && descriptor.shape[3] == 448 && descriptor.dynamic_batch;
    let dynamic_spatial = descriptor.dynamic_axes.len() == 4
        && descriptor.dynamic_axes[2]
        && descriptor.dynamic_axes[3];

    exact || dynamic_batch_exact_spatial || dynamic_spatial
}

#[cfg(test)]
mod tests {
    use lumen_schema::ModelInfo;

    use super::REQUIRED_COMPONENTS;

    #[test]
    fn model_info_example_declares_required_fastvlm_components() {
        let model_info = ModelInfo::from_json_str(include_str!(
            "../../../tools/fastvlm/model_info.example.json"
        ))
        .unwrap();
        let runtime = model_info.runtimes.as_map().get("onnx").unwrap();

        for component in REQUIRED_COMPONENTS {
            assert!(runtime.components.contains(&component.to_owned()));
        }
    }
}
