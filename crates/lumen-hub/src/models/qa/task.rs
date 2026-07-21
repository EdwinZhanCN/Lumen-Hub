//! QA embedding task: same raw + tensor-fast-path + batching surface as the
//! production tasks, backed by the tiny deterministic net.

use std::sync::Arc;

use async_trait::async_trait;
use image::imageops::FilterType;
use lumen_schema::EmbeddingV1;

use super::model::{QA_EMBED_DIM, QA_INPUT_NUMEL, QA_INPUT_SIZE, QaNet};
use crate::backend::{Backend, Device};
use crate::inference_worker;
use crate::service::{
    BatchKey, DEFAULT_TENSOR_MIME, INPUT_KIND_TENSOR, META_INPUT_KIND, ServiceError, ServiceResult,
    TaskHandler, TaskRequest, TaskResult, TaskSpec, TensorValidationOptions, bytes_to_f32_le,
    validate_tensor_request,
};

/// QA-namespace preprocess id: 32×32 RGB, NCHW, values scaled to [0, 1].
/// Deliberately absent from the SDK preprocessor registry — tests construct
/// tensor requests directly.
pub const QA_TENSOR_PREPROCESS_ID: &str = "qa.rgb32-image";

/// Ends with `_semantic_image_embed` so startup warmup exercises it.
pub const QA_EMBED_TASK: &str = "qa_semantic_image_embed";

pub struct QaEmbedTask {
    spec: TaskSpec,
    model_id: String,
    net: Arc<QaNet<Backend>>,
    device: Arc<Device>,
}

impl QaEmbedTask {
    pub fn new(model_id: String, net: QaNet<Backend>, device: Arc<Device>) -> Self {
        let spec = TaskSpec::new(
            QA_EMBED_TASK,
            "Deterministic QA embedding over a 32x32 RGB image",
        )
        .with_input_mimes(["image/jpeg", "image/png", DEFAULT_TENSOR_MIME])
        .with_output_mime("application/json;schema=embedding_v1")
        .with_limit("max_hw", QA_INPUT_SIZE.to_string())
        .with_tensor_fast_path(QA_TENSOR_PREPROCESS_ID, true);
        Self {
            spec,
            model_id,
            net: Arc::new(net),
            device,
        }
    }

    fn validate_tensor(&self, request: &TaskRequest) -> ServiceResult<()> {
        let descriptor = validate_tensor_request(
            request,
            TensorValidationOptions {
                dtype: "fp32",
                layout: "NCHW",
                preprocess_id: QA_TENSOR_PREPROCESS_ID,
            },
        )?;
        let expected = vec![1, 3, QA_INPUT_SIZE, QA_INPUT_SIZE];
        if descriptor.shape != expected {
            return Err(ServiceError::InvalidArgument(format!(
                "qa tensor shape must be {expected:?}, got {:?}",
                descriptor.shape
            )));
        }
        Ok(())
    }

    fn decode_image(&self, bytes: &[u8]) -> ServiceResult<Vec<f32>> {
        let image = image::load_from_memory(bytes).map_err(|err| {
            ServiceError::InvalidArgument(format!("failed to decode image: {err}"))
        })?;
        let rgb = image::imageops::resize(
            &image.to_rgb8(),
            QA_INPUT_SIZE as u32,
            QA_INPUT_SIZE as u32,
            FilterType::Triangle,
        );
        // HWC u8 → NCHW f32 in [0, 1].
        let mut pixels = vec![0.0f32; QA_INPUT_NUMEL];
        let plane = QA_INPUT_SIZE * QA_INPUT_SIZE;
        for (index, pixel) in rgb.pixels().enumerate() {
            for channel in 0..3 {
                pixels[channel * plane + index] = f32::from(pixel.0[channel]) / 255.0;
            }
        }
        Ok(pixels)
    }

    async fn embed(&self, pixels: Vec<f32>, batch: usize) -> ServiceResult<Vec<f32>> {
        let net = Arc::clone(&self.net);
        let device = Arc::clone(&self.device);
        inference_worker::run(move || net.embed(&pixels, batch, &device))
            .await
            .map_err(|e| ServiceError::Internal(format!("inference worker failed: {e}")))
    }

    fn result(&self, row: &[f32]) -> ServiceResult<TaskResult> {
        let embedding = EmbeddingV1 {
            vector: row.to_vec(),
            dim: QA_EMBED_DIM,
            model_id: self.model_id.clone(),
            aesthetic_score: None,
        };
        let payload = serde_json::to_vec(&embedding)
            .map_err(|e| ServiceError::Internal(format!("embedding serialization: {e}")))?;
        Ok(
            TaskResult::new(payload, "application/json;schema=embedding_v1")
                .with_result_schema("embedding_v1"),
        )
    }
}

#[async_trait]
impl TaskHandler for QaEmbedTask {
    fn spec(&self) -> &TaskSpec {
        &self.spec
    }

    fn batch_key(&self, request: &TaskRequest) -> ServiceResult<Option<BatchKey>> {
        if request.meta.get(META_INPUT_KIND).map(String::as_str) != Some(INPUT_KIND_TENSOR) {
            return Ok(None);
        }
        self.validate_tensor(request)?;
        Ok(Some(BatchKey::new(format!(
            "qa\nmodel.id={}\npreprocess.id={QA_TENSOR_PREPROCESS_ID}",
            self.model_id
        ))))
    }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
        let pixels =
            if request.meta.get(META_INPUT_KIND).map(String::as_str) == Some(INPUT_KIND_TENSOR) {
                self.validate_tensor(&request)?;
                bytes_to_f32_le(&request.payload)?
            } else {
                self.decode_image(&request.payload)?
            };
        let embedding = self.embed(pixels, 1).await?;
        self.result(&embedding)
    }

    async fn handle_batch(&self, requests: Vec<TaskRequest>) -> ServiceResult<Vec<TaskResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let batch = requests.len();
        let mut pixels = Vec::with_capacity(batch * QA_INPUT_NUMEL);
        for request in &requests {
            self.validate_tensor(request)?;
            pixels.extend(bytes_to_f32_le(&request.payload)?);
        }
        let embeddings = self.embed(pixels, batch).await?;
        embeddings
            .chunks(QA_EMBED_DIM)
            .map(|row| {
                self.result(row)
                    .map(|result| result.with_meta("lumen.batch_size", batch.to_string()))
            })
            .collect()
    }
}

/// The full tensor-request metadata contract for the QA task, reusable by
/// unit and e2e tests (gRPC meta maps use the same keys).
pub fn tensor_request_meta() -> Vec<(&'static str, String)> {
    use crate::service::{
        INPUT_KIND_TENSOR, META_INPUT_KIND, META_PREPROCESS_ID, META_PREPROCESS_SKIP,
        META_TENSOR_BYTE_ORDER, META_TENSOR_DTYPE, META_TENSOR_FORMAT, META_TENSOR_LAYOUT,
        META_TENSOR_SHAPE, TENSOR_BYTE_ORDER_LITTLE, TENSOR_FORMAT_CONTIGUOUS,
    };
    vec![
        (META_INPUT_KIND, INPUT_KIND_TENSOR.to_owned()),
        (META_PREPROCESS_SKIP, "true".to_owned()),
        (META_TENSOR_DTYPE, "fp32".to_owned()),
        (META_TENSOR_LAYOUT, "NCHW".to_owned()),
        (META_TENSOR_FORMAT, TENSOR_FORMAT_CONTIGUOUS.to_owned()),
        (META_TENSOR_BYTE_ORDER, TENSOR_BYTE_ORDER_LITTLE.to_owned()),
        (META_PREPROCESS_ID, QA_TENSOR_PREPROCESS_ID.to_owned()),
        (META_TENSOR_SHAPE, "[1,3,32,32]".to_owned()),
    ]
}
