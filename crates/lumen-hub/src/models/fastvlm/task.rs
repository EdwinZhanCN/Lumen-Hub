use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use half::f16;
use lumnn::core::{
    context::MLContext,
    node::MLNodeRef,
    packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor},
};

use super::metadata::METADATA;
use crate::service::{
    BatchKey, DEFAULT_TENSOR_MIME, INPUT_KIND_TENSOR, META_INPUT_KIND, META_MODEL_ID,
    META_MODEL_VERSION, ServiceError, ServiceResult, TaskHandler, TaskRequest, TaskResult,
    TaskSpec, TensorDescriptor, TensorValidationOptions, bytes_to_f16_le, bytes_to_f32_le,
    f16_to_le_bytes, f32_to_le_bytes, tensor_response_meta, validate_tensor_request,
};

pub const FASTVLM_PREPROCESS_ID: &str = "fastvlm_image_preprocess_v1";
pub const META_PROMPT: &str = "lumen.prompt";
const INPUT_LAYOUT: &str = "NCHW";
const OUTPUT_LAYOUT: &str = "BSH";

pub struct FastVlmEmbedsTask {
    spec: TaskSpec,
    context: Arc<MLContext>,
    model_id: String,
    model_version: String,
    vision_node: MLNodeRef,
    embed_node: MLNodeRef,
    vision_input_dtype: MLPacketDataType,
    tokenizer: tokenizers::Tokenizer,
    prompt_embeds_cache: Mutex<HashMap<String, CachedPromptEmbeds>>,
}

impl FastVlmEmbedsTask {
    pub fn new(
        task_name: impl Into<String>,
        context: Arc<MLContext>,
        model_id: impl Into<String>,
        model_version: impl Into<String>,
        vision_node: MLNodeRef,
        embed_node: MLNodeRef,
        tokenizer: tokenizers::Tokenizer,
    ) -> ServiceResult<Self> {
        let vision_input_dtype = vision_node
            .input_descriptors()
            .get(METADATA.vision_input_name)
            .ok_or_else(|| {
                ServiceError::InvalidArgument(format!(
                    "FastVLM vision component missing input `{}`",
                    METADATA.vision_input_name
                ))
            })?
            .dtype;

        Ok(Self {
            spec: TaskSpec::new(task_name, "FastVLM image/prompt -> merged input embeddings")
                .with_input_mimes([DEFAULT_TENSOR_MIME])
                .with_output_mime(DEFAULT_TENSOR_MIME),
            context,
            model_id: model_id.into(),
            model_version: model_version.into(),
            vision_node,
            embed_node,
            vision_input_dtype,
            tokenizer,
            prompt_embeds_cache: Mutex::new(HashMap::new()),
        })
    }

    fn prompt<'a>(&self, request: &'a TaskRequest) -> ServiceResult<&'a str> {
        request
            .meta
            .get(META_PROMPT)
            .map(String::as_str)
            .filter(|prompt| !prompt.trim().is_empty())
            .ok_or_else(|| {
                ServiceError::InvalidArgument(format!("missing non-empty metadata `{META_PROMPT}`"))
            })
    }

    fn tensor_descriptor(&self, request: &TaskRequest) -> ServiceResult<TensorDescriptor> {
        let descriptor = validate_tensor_request(
            request,
            TensorValidationOptions {
                dtype: ml_dtype_to_tensor_dtype(self.vision_input_dtype)?,
                layout: INPUT_LAYOUT,
                preprocess_id: FASTVLM_PREPROCESS_ID,
            },
        )?;
        if descriptor.shape != vec![1, 3, 448, 448] {
            return Err(ServiceError::InvalidArgument(format!(
                "FastVLM image tensor shape must be [1, 3, 448, 448], got {:?}",
                descriptor.shape
            )));
        }
        Ok(descriptor)
    }

    fn batch_key_for_request(&self, request: &TaskRequest) -> ServiceResult<BatchKey> {
        let descriptor = self.tensor_descriptor(request)?;
        let prompt = self.prompt(request)?;
        Ok(BatchKey::new(format!(
            "model.id={}\nmodel.version={}\npayload_mime={}\ndtype={}\nshape={:?}\nlayout={}\nformat={}\nbyte_order={}\npreprocess.id={}\nprompt={}",
            request
                .meta
                .get(META_MODEL_ID)
                .map(String::as_str)
                .unwrap_or(&self.model_id),
            request
                .meta
                .get(META_MODEL_VERSION)
                .map(String::as_str)
                .unwrap_or(&self.model_version),
            DEFAULT_TENSOR_MIME,
            descriptor.dtype,
            descriptor.shape,
            descriptor.layout,
            descriptor.format,
            descriptor.byte_order,
            FASTVLM_PREPROCESS_ID,
            prompt
        )))
    }

    fn tokenize_prompt(&self, prompt: &str) -> ServiceResult<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(prompt, METADATA.add_special_tokens_after_template)
            .map_err(|err| ServiceError::Internal(format!("tokenization failed: {err}")))?;
        let ids = encoding
            .get_ids()
            .iter()
            .map(|id| i64::from(*id))
            .collect::<Vec<_>>();
        if ids.is_empty() {
            return Err(ServiceError::InvalidArgument(
                "prompt tokenization produced no tokens".to_owned(),
            ));
        }
        Ok(ids)
    }

    fn batched_image_packet(&self, requests: &[TaskRequest]) -> ServiceResult<MLPacket> {
        let shape = vec![requests.len(), 3, 448, 448];
        let tensor = match self.vision_input_dtype {
            MLPacketDataType::Float32 => {
                let mut values = Vec::new();
                for request in requests {
                    self.tensor_descriptor(request)?;
                    values.extend(bytes_to_f32_le(&request.payload)?);
                }
                HostTensor::Float32(values)
            }
            MLPacketDataType::Float16 => {
                let mut values = Vec::new();
                for request in requests {
                    self.tensor_descriptor(request)?;
                    values.extend(bytes_to_f16_le(&request.payload)?);
                }
                HostTensor::Float16(values)
            }
            other => {
                return Err(ServiceError::Internal(format!(
                    "unsupported FastVLM vision input dtype {other:?}"
                )));
            }
        };
        self.context
            .packet_from_host_tensor(
                MLPacketDescriptor::new(self.vision_input_dtype, shape),
                tensor,
            )
            .map_err(ServiceError::Internal)
    }

    fn prompt_packet(&self, token_ids: Vec<i64>) -> ServiceResult<MLPacket> {
        self.context
            .packet_from_host_tensor(
                MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1, token_ids.len()]),
                HostTensor::Int64(token_ids),
            )
            .map_err(ServiceError::Internal)
    }

    async fn run_vision(&self, image_packet: MLPacket) -> ServiceResult<MLPacket> {
        let mut outputs = self
            .vision_node
            .execute(
                HashMap::from([(METADATA.vision_input_name.to_owned(), image_packet)]),
                self.context.as_ref(),
            )
            .await
            .map_err(ServiceError::Internal)?;
        outputs.remove(METADATA.vision_output_name).ok_or_else(|| {
            ServiceError::Internal(format!(
                "FastVLM vision output missing key `{}`",
                METADATA.vision_output_name
            ))
        })
    }

    async fn run_embed(&self, prompt_packet: MLPacket) -> ServiceResult<MLPacket> {
        let mut outputs = self
            .embed_node
            .execute(
                HashMap::from([(METADATA.embed_input_name.to_owned(), prompt_packet)]),
                self.context.as_ref(),
            )
            .await
            .map_err(ServiceError::Internal)?;
        outputs.remove(METADATA.embed_output_name).ok_or_else(|| {
            ServiceError::Internal(format!(
                "FastVLM embed output missing key `{}`",
                METADATA.embed_output_name
            ))
        })
    }

    async fn cached_prompt_embeds(&self, prompt: &str) -> ServiceResult<CachedPromptEmbeds> {
        if let Some(cached) = self
            .prompt_embeds_cache
            .lock()
            .map_err(|_| {
                ServiceError::Internal("FastVLM prompt embeds cache lock poisoned".to_owned())
            })?
            .get(prompt)
            .cloned()
        {
            return Ok(cached);
        }

        let token_ids = self.tokenize_prompt(prompt)?;
        let prompt_embeds = self.run_embed(self.prompt_packet(token_ids)?).await?;
        let cached = materialize_prompt_embeds(prompt_embeds).await?;

        self.prompt_embeds_cache
            .lock()
            .map_err(|_| {
                ServiceError::Internal("FastVLM prompt embeds cache lock poisoned".to_owned())
            })?
            .insert(prompt.to_owned(), cached.clone());

        Ok(cached)
    }

    async fn merged_batch(&self, requests: &[TaskRequest]) -> ServiceResult<MergedEmbeds> {
        if requests.is_empty() {
            return Err(ServiceError::InvalidArgument(
                "FastVLM batch must contain at least one request".to_owned(),
            ));
        }

        let prompt = self.prompt(&requests[0])?;
        for request in requests {
            if self.prompt(request)? != prompt {
                return Err(ServiceError::InvalidArgument(
                    "FastVLM batch prompts must match exactly".to_owned(),
                ));
            }
        }

        let image_features = self
            .run_vision(self.batched_image_packet(requests)?)
            .await?;
        let prompt_embeds = self.cached_prompt_embeds(prompt).await?;
        merge_image_and_prompt(image_features, &prompt_embeds, requests.len()).await
    }

    fn result_for_row(&self, merged: &MergedEmbeds, row: usize) -> ServiceResult<TaskResult> {
        let shape = vec![1, merged.sequence_length, METADATA.hidden_size];
        let meta = tensor_response_meta(
            merged.dtype.as_str(),
            &shape,
            OUTPUT_LAYOUT,
            FASTVLM_PREPROCESS_ID,
            &self.model_id,
            &self.model_version,
        );
        let row_elements = merged.sequence_length * METADATA.hidden_size;
        let start = row * row_elements;
        let end = start + row_elements;

        let payload = match &merged.values {
            MergedValues::Float32(values) => f32_to_le_bytes(&values[start..end]),
            MergedValues::Float16(values) => f16_to_le_bytes(&values[start..end]),
        };

        let mut result = TaskResult::new(payload, DEFAULT_TENSOR_MIME);
        result.meta = meta;
        Ok(result)
    }
}

#[async_trait]
impl TaskHandler for FastVlmEmbedsTask {
    fn spec(&self) -> &TaskSpec {
        &self.spec
    }

    fn batch_key(&self, request: &TaskRequest) -> ServiceResult<Option<BatchKey>> {
        if normalized_meta(request.meta.get(META_INPUT_KIND)).as_deref() != Some(INPUT_KIND_TENSOR)
        {
            return Ok(None);
        }
        Ok(Some(self.batch_key_for_request(request)?))
    }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
        let merged = self.merged_batch(&[request]).await?;
        self.result_for_row(&merged, 0)
    }

    async fn handle_batch(&self, requests: Vec<TaskRequest>) -> ServiceResult<Vec<TaskResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let merged = self.merged_batch(&requests).await?;
        (0..requests.len())
            .map(|row| self.result_for_row(&merged, row))
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorDType {
    Float32,
    Float16,
}

impl TensorDType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Float32 => "fp32",
            Self::Float16 => "fp16",
        }
    }
}

#[derive(Debug)]
enum MergedValues {
    Float32(Vec<f32>),
    Float16(Vec<f16>),
}

#[derive(Clone, Debug)]
struct CachedPromptEmbeds {
    values: MergedValues,
    prompt_tokens: usize,
}

impl Clone for MergedValues {
    fn clone(&self) -> Self {
        match self {
            Self::Float32(values) => Self::Float32(values.clone()),
            Self::Float16(values) => Self::Float16(values.clone()),
        }
    }
}

struct MergedEmbeds {
    dtype: TensorDType,
    values: MergedValues,
    sequence_length: usize,
}

async fn merge_image_and_prompt(
    image_features: MLPacket,
    prompt_embeds: &CachedPromptEmbeds,
    batch_size: usize,
) -> ServiceResult<MergedEmbeds> {
    let image_shape = image_features.descriptor.shape.clone();
    validate_feature_shape("image_features", &image_shape, batch_size)?;
    let image_tokens = image_shape[1];
    let sequence_length = image_tokens + prompt_embeds.prompt_tokens;

    let image_tensor = image_features
        .to_host_tensor()
        .await
        .map_err(ServiceError::Internal)?;

    match (image_tensor, &prompt_embeds.values) {
        (HostTensor::Float32(image), MergedValues::Float32(prompt)) => {
            let mut values =
                Vec::with_capacity(batch_size * sequence_length * METADATA.hidden_size);
            let image_row = image_tokens * METADATA.hidden_size;
            let prompt_row = prompt_embeds.prompt_tokens * METADATA.hidden_size;
            for batch in 0..batch_size {
                values.extend_from_slice(&image[batch * image_row..(batch + 1) * image_row]);
                values.extend_from_slice(&prompt[..prompt_row]);
            }
            Ok(MergedEmbeds {
                dtype: TensorDType::Float32,
                values: MergedValues::Float32(values),
                sequence_length,
            })
        }
        (HostTensor::Float16(image), MergedValues::Float16(prompt)) => {
            let mut values =
                Vec::with_capacity(batch_size * sequence_length * METADATA.hidden_size);
            let image_row = image_tokens * METADATA.hidden_size;
            let prompt_row = prompt_embeds.prompt_tokens * METADATA.hidden_size;
            for batch in 0..batch_size {
                values.extend_from_slice(&image[batch * image_row..(batch + 1) * image_row]);
                values.extend_from_slice(&prompt[..prompt_row]);
            }
            Ok(MergedEmbeds {
                dtype: TensorDType::Float16,
                values: MergedValues::Float16(values),
                sequence_length,
            })
        }
        (left, right) => Err(ServiceError::Internal(format!(
            "FastVLM vision/embed output dtype mismatch or unsupported tensors: {left:?} vs {right:?}"
        ))),
    }
}

async fn materialize_prompt_embeds(prompt_embeds: MLPacket) -> ServiceResult<CachedPromptEmbeds> {
    let prompt_shape = prompt_embeds.descriptor.shape.clone();
    validate_feature_shape("prompt_embeds", &prompt_shape, 1)?;
    let prompt_tokens = prompt_shape[1];
    let prompt_tensor = prompt_embeds
        .to_host_tensor()
        .await
        .map_err(ServiceError::Internal)?;

    match prompt_tensor {
        HostTensor::Float32(values) => Ok(CachedPromptEmbeds {
            values: MergedValues::Float32(values),
            prompt_tokens,
        }),
        HostTensor::Float16(values) => Ok(CachedPromptEmbeds {
            values: MergedValues::Float16(values),
            prompt_tokens,
        }),
        other => Err(ServiceError::Internal(format!(
            "FastVLM embed output must be fp32/fp16 [1,S,H], got {other:?}"
        ))),
    }
}

fn validate_feature_shape(name: &str, shape: &[usize], batch_size: usize) -> ServiceResult<()> {
    if shape.len() != 3 {
        return Err(ServiceError::Internal(format!(
            "FastVLM `{name}` must be rank 3 [B,S,H], got {shape:?}"
        )));
    }
    if shape[0] != batch_size {
        return Err(ServiceError::Internal(format!(
            "FastVLM `{name}` batch size mismatch: expected {batch_size}, got {}",
            shape[0]
        )));
    }
    if shape[2] != METADATA.hidden_size {
        return Err(ServiceError::Internal(format!(
            "FastVLM `{name}` hidden size mismatch: expected {}, got {}",
            METADATA.hidden_size, shape[2]
        )));
    }
    Ok(())
}

fn ml_dtype_to_tensor_dtype(dtype: MLPacketDataType) -> ServiceResult<&'static str> {
    match dtype {
        MLPacketDataType::Float32 => Ok("fp32"),
        MLPacketDataType::Float16 => Ok("fp16"),
        other => Err(ServiceError::Internal(format!(
            "unsupported FastVLM image tensor dtype {other:?}"
        ))),
    }
}

fn normalized_meta(value: Option<&String>) -> Option<String> {
    value.map(|value| value.trim().to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use async_trait::async_trait;
    use tokenizers::Tokenizer;

    use super::*;
    use crate::service::{
        META_OUTPUT_TENSOR_DTYPE, META_OUTPUT_TENSOR_LAYOUT, META_OUTPUT_TENSOR_SHAPE,
        META_PREPROCESS_ID, META_PREPROCESS_SKIP, META_TENSOR_BYTE_ORDER, META_TENSOR_DTYPE,
        META_TENSOR_FORMAT, META_TENSOR_LAYOUT, META_TENSOR_SHAPE, TENSOR_BYTE_ORDER_LITTLE,
        TENSOR_FORMAT_CONTIGUOUS,
    };
    use lumnn::core::{context::MLContextOptions, node::MLNode};

    #[tokio::test]
    async fn validates_tensor_metadata_and_prompt() {
        let task = test_task();
        let request = tensor_request("describe", vec![1, 3, 448, 448], FASTVLM_PREPROCESS_ID);
        assert!(task.batch_key(&request).unwrap().is_some());

        let wrong_shape = tensor_request("describe", vec![1, 3, 224, 224], FASTVLM_PREPROCESS_ID);
        assert!(task.batch_key(&wrong_shape).is_err());

        let wrong_preprocess = tensor_request("describe", vec![1, 3, 448, 448], "wrong");
        assert!(task.batch_key(&wrong_preprocess).is_err());

        let mut missing_prompt =
            tensor_request("describe", vec![1, 3, 448, 448], FASTVLM_PREPROCESS_ID);
        missing_prompt.meta.remove(META_PROMPT);
        assert!(task.batch_key(&missing_prompt).is_err());
    }

    #[tokio::test]
    async fn returns_merged_inputs_embed_tensor() {
        let task = test_task();
        let result = task
            .handle(tensor_request(
                "describe",
                vec![1, 3, 448, 448],
                FASTVLM_PREPROCESS_ID,
            ))
            .await
            .unwrap();

        assert_eq!(result.payload_mime, DEFAULT_TENSOR_MIME);
        assert_eq!(
            result.meta.get(META_OUTPUT_TENSOR_DTYPE),
            Some(&"fp32".to_owned())
        );
        assert_eq!(
            result.meta.get(META_OUTPUT_TENSOR_LAYOUT),
            Some(&OUTPUT_LAYOUT.to_owned())
        );
        assert_eq!(
            result.meta.get(META_OUTPUT_TENSOR_SHAPE),
            Some(&"[1,3,896]".to_owned())
        );
        let values = bytes_to_f32_le(&result.payload).unwrap();
        assert_eq!(values.len(), 3 * METADATA.hidden_size);
        assert_eq!(values[0], 10.0);
        assert_eq!(values[METADATA.hidden_size], 20.0);
    }

    #[tokio::test]
    async fn handle_batch_preserves_order_and_splits_rows() {
        let task = test_task();
        let results = task
            .handle_batch(vec![
                tensor_request("describe", vec![1, 3, 448, 448], FASTVLM_PREPROCESS_ID),
                tensor_request("describe", vec![1, 3, 448, 448], FASTVLM_PREPROCESS_ID),
            ])
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        let first = bytes_to_f32_le(&results[0].payload).unwrap();
        let second = bytes_to_f32_le(&results[1].payload).unwrap();
        assert_eq!(first[0], 10.0);
        assert_eq!(second[0], 11.0);
        assert_eq!(first[METADATA.hidden_size], 20.0);
        assert_eq!(second[METADATA.hidden_size], 21.0);
        assert_eq!(first[2 * METADATA.hidden_size], 30.0);
        assert_eq!(second[2 * METADATA.hidden_size], 30.0);
    }

    #[tokio::test]
    async fn reuses_cached_prompt_embeds_for_repeated_prompt() {
        let counter = Arc::new(AtomicUsize::new(0));
        let task = test_task_with_counter(Arc::clone(&counter));

        task.handle(tensor_request(
            "describe",
            vec![1, 3, 448, 448],
            FASTVLM_PREPROCESS_ID,
        ))
        .await
        .unwrap();
        task.handle(tensor_request(
            "describe",
            vec![1, 3, 448, 448],
            FASTVLM_PREPROCESS_ID,
        ))
        .await
        .unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn recomputes_prompt_embeds_for_distinct_prompts() {
        let counter = Arc::new(AtomicUsize::new(0));
        let task = test_task_with_counter(Arc::clone(&counter));

        task.handle(tensor_request(
            "describe",
            vec![1, 3, 448, 448],
            FASTVLM_PREPROCESS_ID,
        ))
        .await
        .unwrap();
        task.handle(tensor_request(
            "caption",
            vec![1, 3, 448, 448],
            FASTVLM_PREPROCESS_ID,
        ))
        .await
        .unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn different_prompts_produce_different_batch_keys() {
        let task = test_task();
        let left = task
            .batch_key(&tensor_request(
                "describe",
                vec![1, 3, 448, 448],
                FASTVLM_PREPROCESS_ID,
            ))
            .unwrap()
            .unwrap();
        let right = task
            .batch_key(&tensor_request(
                "caption",
                vec![1, 3, 448, 448],
                FASTVLM_PREPROCESS_ID,
            ))
            .unwrap()
            .unwrap();
        assert_ne!(left, right);
    }

    fn test_task() -> FastVlmEmbedsTask {
        test_task_with_counter(Arc::new(AtomicUsize::new(0)))
    }

    fn test_task_with_counter(counter: Arc<AtomicUsize>) -> FastVlmEmbedsTask {
        let context = MLContext::new(MLContextOptions::default()).unwrap();
        FastVlmEmbedsTask::new(
            "default_vlm_embeds",
            Arc::clone(&context),
            "fastvlm-test",
            "1.0.0",
            Arc::new(MockVisionNode::new()),
            Arc::new(MockEmbedNode::new(counter)),
            test_tokenizer(),
        )
        .unwrap()
    }

    fn test_tokenizer() -> Tokenizer {
        Tokenizer::from_bytes(
            r#"{
              "version": "1.0",
              "truncation": null,
              "padding": null,
              "added_tokens": [],
              "normalizer": null,
              "pre_tokenizer": null,
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": {
                  "[UNK]": 0,
                  "describe": 1,
                  "caption": 2
                },
                "unk_token": "[UNK]"
              }
            }"#,
        )
        .unwrap()
    }

    fn tensor_request(prompt: &str, shape: Vec<usize>, preprocess_id: &str) -> TaskRequest {
        let element_count = shape.iter().product::<usize>();
        TaskRequest::new(
            f32_to_le_bytes(&vec![0.0; element_count]),
            DEFAULT_TENSOR_MIME,
        )
        .with_meta(META_INPUT_KIND, INPUT_KIND_TENSOR)
        .with_meta(META_TENSOR_DTYPE, "fp32")
        .with_meta(META_TENSOR_SHAPE, serde_json::to_string(&shape).unwrap())
        .with_meta(META_TENSOR_LAYOUT, INPUT_LAYOUT)
        .with_meta(META_TENSOR_FORMAT, TENSOR_FORMAT_CONTIGUOUS)
        .with_meta(META_TENSOR_BYTE_ORDER, TENSOR_BYTE_ORDER_LITTLE)
        .with_meta(META_PREPROCESS_ID, preprocess_id)
        .with_meta(META_PREPROCESS_SKIP, "true")
        .with_meta(META_PROMPT, prompt)
    }

    struct MockVisionNode {
        input_descriptors: HashMap<String, MLPacketDescriptor>,
        output_descriptors: HashMap<String, MLPacketDescriptor>,
    }

    impl MockVisionNode {
        fn new() -> Self {
            Self {
                input_descriptors: HashMap::from([(
                    METADATA.vision_input_name.to_owned(),
                    MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 3, 448, 448])
                        .with_dynamic_batch(),
                )]),
                output_descriptors: HashMap::from([(
                    METADATA.vision_output_name.to_owned(),
                    MLPacketDescriptor::new(
                        MLPacketDataType::Float32,
                        vec![1, 2, METADATA.hidden_size],
                    )
                    .with_dynamic_batch(),
                )]),
            }
        }
    }

    #[async_trait]
    impl MLNode for MockVisionNode {
        fn name(&self) -> &str {
            "mock_vision"
        }

        fn input_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> {
            &self.input_descriptors
        }

        fn output_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> {
            &self.output_descriptors
        }

        async fn execute(
            &self,
            mut inputs: HashMap<String, MLPacket>,
            _context: &MLContext,
        ) -> Result<HashMap<String, MLPacket>, String> {
            let packet = inputs.remove(METADATA.vision_input_name).unwrap();
            let (context, descriptor, _payload) = packet.into_parts()?;
            let batch = descriptor.shape[0];
            let mut values = Vec::new();
            for row in 0..batch {
                values.extend(vec![10.0 + row as f32; METADATA.hidden_size]);
                values.extend(vec![20.0 + row as f32; METADATA.hidden_size]);
            }
            let output = MLPacket::from_host_tensor(
                context,
                MLPacketDescriptor::new(
                    MLPacketDataType::Float32,
                    vec![batch, 2, METADATA.hidden_size],
                ),
                HostTensor::Float32(values),
            )?;
            Ok(HashMap::from([(
                METADATA.vision_output_name.to_owned(),
                output,
            )]))
        }
    }

    struct MockEmbedNode {
        input_descriptors: HashMap<String, MLPacketDescriptor>,
        output_descriptors: HashMap<String, MLPacketDescriptor>,
        execute_count: Arc<AtomicUsize>,
    }

    impl MockEmbedNode {
        fn new(execute_count: Arc<AtomicUsize>) -> Self {
            Self {
                input_descriptors: HashMap::from([(
                    METADATA.embed_input_name.to_owned(),
                    MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1, 1])
                        .with_dynamic_axis(1),
                )]),
                output_descriptors: HashMap::from([(
                    METADATA.embed_output_name.to_owned(),
                    MLPacketDescriptor::new(
                        MLPacketDataType::Float32,
                        vec![1, 1, METADATA.hidden_size],
                    )
                    .with_dynamic_axis(1),
                )]),
                execute_count,
            }
        }
    }

    #[async_trait]
    impl MLNode for MockEmbedNode {
        fn name(&self) -> &str {
            "mock_embed"
        }

        fn input_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> {
            &self.input_descriptors
        }

        fn output_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> {
            &self.output_descriptors
        }

        async fn execute(
            &self,
            mut inputs: HashMap<String, MLPacket>,
            _context: &MLContext,
        ) -> Result<HashMap<String, MLPacket>, String> {
            self.execute_count.fetch_add(1, Ordering::SeqCst);
            let packet = inputs.remove(METADATA.embed_input_name).unwrap();
            let (context, descriptor, _payload) = packet.into_parts()?;
            let seq = descriptor.shape[1];
            let values = vec![30.0; seq * METADATA.hidden_size];
            let output = MLPacket::from_host_tensor(
                context,
                MLPacketDescriptor::new(
                    MLPacketDataType::Float32,
                    vec![1, seq, METADATA.hidden_size],
                ),
                HostTensor::Float32(values),
            )?;
            Ok(HashMap::from([(
                METADATA.embed_output_name.to_owned(),
                output,
            )]))
        }
    }
}
