use std::sync::Arc;

use async_trait::async_trait;
use lumen_schema::{Label, LabelsV1};

use super::dataset::BioClipDataset;
use super::model::BioClipVisionModel;
use super::preprocess::ClipImagePreprocessConfig;
use crate::{
    inference_worker,
    service::{
        BatchKey, DEFAULT_TENSOR_MIME, INPUT_KIND_TENSOR, META_INPUT_KIND, META_MODEL_ID,
        META_MODEL_VERSION, PREPROCESS_BIOCLIP2_224_IMAGE, ServiceError, ServiceResult,
        TaskHandler, TaskRequest, TaskResult, TaskSpec, TensorDescriptor, TensorValidationOptions,
        bytes_to_f32_le, validate_tensor_request,
    },
};

const SUPPORTED_IMAGE_INPUT_MIMES: [&str; 4] =
    ["image/jpeg", "image/png", "image/webp", "image/avif"];
const IMAGE_TENSOR_LAYOUT: &str = "NCHW";
const TENSOR_INPUT_DTYPE: &str = "fp32";
const BIOCLIP_DEFAULT_TOP_K: usize = 5;
const BIOCLIP_TOP_K_META_KEYS: [&str; 5] = ["TopK", "topK", "top_k", "top-k", "lumen.top_k"];

/// BioCLIP zero-shot classification: vision encoder → ANN search + rerank over a
/// taxon catalog → top-k taxonomy labels.
pub struct BioClipClassifyTask {
    spec: TaskSpec,
    model: Arc<BioClipVisionModel>,
    model_id: String,
    preprocess: ClipImagePreprocessConfig,
    logit_scale: f32,
    dataset: Arc<BioClipDataset>,
}

impl BioClipClassifyTask {
    pub fn new(
        task_name: impl Into<String>,
        model: Arc<BioClipVisionModel>,
        model_id: impl Into<String>,
        preprocess: ClipImagePreprocessConfig,
        logit_scale: f32,
        dataset_name: &str,
        dataset: Arc<BioClipDataset>,
    ) -> Self {
        Self {
            spec: TaskSpec::new(
                task_name,
                "BioCLIP image classification with dataset text embeddings",
            )
            .with_input_mimes(image_input_mimes_with_tensor())
            .with_output_mime(lumen_schema::mime::LABELS_V1_JSON)
            .with_metadata("output_schema", "labels_v1")
            .with_metadata("dataset", dataset_name)
            .with_tensor_fast_path(PREPROCESS_BIOCLIP2_224_IMAGE, true),
            model,
            model_id: model_id.into(),
            preprocess,
            logit_scale,
            dataset,
        }
    }

    fn tensor_input_descriptor(&self, request: &TaskRequest) -> ServiceResult<TensorDescriptor> {
        let descriptor = validate_tensor_request(
            request,
            TensorValidationOptions {
                dtype: TENSOR_INPUT_DTYPE,
                layout: IMAGE_TENSOR_LAYOUT,
                preprocess_id: PREPROCESS_BIOCLIP2_224_IMAGE,
            },
        )?;
        let expected_shape = self.preprocess.output_shape();
        if descriptor.shape != expected_shape {
            return Err(ServiceError::InvalidArgument(format!(
                "BioCLIP image tensor shape must be {:?}, got {:?}",
                expected_shape, descriptor.shape
            )));
        }
        Ok(descriptor)
    }

    /// Encodes `pixels` ([batch,3,H,W]) and classifies each row, returning one
    /// label list per row (per-row `top_k`).
    async fn classify(
        &self,
        pixels: Vec<f32>,
        batch: usize,
        top_k: Vec<usize>,
    ) -> ServiceResult<Vec<Vec<Label>>> {
        let shape = self.preprocess.output_shape();
        let (height, width) = (shape[2], shape[3]);
        let model = Arc::clone(&self.model);
        let dataset = Arc::clone(&self.dataset);
        let logit_scale = self.logit_scale;

        run_blocking(move || {
            let raw = model.encode(pixels, batch, height, width);
            let row_width = raw.len().checked_div(batch.max(1)).unwrap_or(0);
            if row_width == 0 || row_width * batch != raw.len() {
                return Err(ServiceError::Internal(format!(
                    "BioCLIP embedding length {} is not divisible by batch {batch}",
                    raw.len()
                )));
            }
            raw.chunks(row_width)
                .zip(top_k)
                .map(|(row, k)| dataset.top_k(row, k, logit_scale))
                .collect::<ServiceResult<Vec<_>>>()
        })
        .await?
    }
}

#[async_trait]
impl TaskHandler for BioClipClassifyTask {
    fn spec(&self) -> &TaskSpec {
        &self.spec
    }

    fn batch_key(&self, request: &TaskRequest) -> ServiceResult<Option<BatchKey>> {
        if normalized_meta(request.meta.get(META_INPUT_KIND)).as_deref() != Some(INPUT_KIND_TENSOR)
        {
            return Ok(None);
        }
        let descriptor = self.tensor_input_descriptor(request)?;
        Ok(Some(BatchKey::new(format!(
            "model.id={}\nmodel.version={}\npayload_mime={}\ndtype={}\nshape_tail={:?}\nlayout={}\nformat={}\nbyte_order={}\npreprocess.id={}",
            request
                .meta
                .get(META_MODEL_ID)
                .map(String::as_str)
                .unwrap_or(&self.model_id),
            request
                .meta
                .get(META_MODEL_VERSION)
                .map(String::as_str)
                .unwrap_or(""),
            DEFAULT_TENSOR_MIME,
            descriptor.dtype,
            &descriptor.shape[1..],
            descriptor.layout,
            descriptor.format,
            descriptor.byte_order,
            PREPROCESS_BIOCLIP2_224_IMAGE
        ))))
    }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
        let top_k = top_k_from_request(&request, self.dataset.len())?;
        let pixels = if normalized_meta(request.meta.get(META_INPUT_KIND)).as_deref()
            == Some(INPUT_KIND_TENSOR)
        {
            self.tensor_input_descriptor(&request)?;
            bytes_to_f32_le(&request.payload)?
        } else {
            if !is_supported_image_input_mime(&request.payload_mime) {
                return Err(ServiceError::InvalidArgument(format!(
                    "unsupported BioCLIP image input MIME `{}`; supported MIME types: {}",
                    request.payload_mime,
                    SUPPORTED_IMAGE_INPUT_MIMES.join(", ")
                )));
            }
            self.preprocess.preprocess_image_bytes(&request.payload)?
        };

        let mut labels = self.classify(pixels, 1, vec![top_k]).await?;
        let labels = labels.pop().unwrap_or_default();
        labels_json_result(LabelsV1::new(labels, &self.model_id))
    }

    async fn handle_batch(&self, requests: Vec<TaskRequest>) -> ServiceResult<Vec<TaskResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let mut pixels = Vec::new();
        let mut top_k_values = Vec::with_capacity(requests.len());
        for request in &requests {
            self.tensor_input_descriptor(request)?;
            pixels.extend(bytes_to_f32_le(&request.payload)?);
            top_k_values.push(top_k_from_request(request, self.dataset.len())?);
        }

        let label_lists = self.classify(pixels, requests.len(), top_k_values).await?;
        label_lists
            .into_iter()
            .map(|labels| labels_json_result(LabelsV1::new(labels, &self.model_id)))
            .collect()
    }
}

/// Runs a blocking inference closure on the dedicated inference worker.
async fn run_blocking<F, T>(f: F) -> ServiceResult<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    inference_worker::run(f)
        .await
        .map_err(|e| ServiceError::Internal(format!("inference worker failed: {e}")))
}

fn top_k_from_request(request: &TaskRequest, max_len: usize) -> ServiceResult<usize> {
    for key in BIOCLIP_TOP_K_META_KEYS {
        if let Some(value) = request.meta.get(key) {
            let parsed = value.trim().parse::<usize>().map_err(|err| {
                ServiceError::InvalidArgument(format!("invalid BioCLIP TopK `{value}`: {err}"))
            })?;
            if parsed == 0 {
                return Err(ServiceError::InvalidArgument(
                    "BioCLIP TopK must be greater than zero".to_owned(),
                ));
            }
            return Ok(parsed.min(max_len.max(1)));
        }
    }
    Ok(BIOCLIP_DEFAULT_TOP_K.min(max_len.max(1)))
}

fn image_input_mimes_with_tensor() -> Vec<String> {
    SUPPORTED_IMAGE_INPUT_MIMES
        .iter()
        .copied()
        .chain(std::iter::once(DEFAULT_TENSOR_MIME))
        .map(str::to_owned)
        .collect()
}

fn normalized_meta(value: Option<&String>) -> Option<String> {
    value.map(|value| value.trim().to_ascii_lowercase())
}

fn is_supported_image_input_mime(mime: &str) -> bool {
    let base = mime
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    SUPPORTED_IMAGE_INPUT_MIMES
        .iter()
        .any(|supported| *supported == base)
}

fn labels_json_result(labels: LabelsV1) -> ServiceResult<TaskResult> {
    let json_bytes = labels
        .to_json_bytes()
        .map_err(|e| ServiceError::Internal(e.to_string()))?;
    Ok(
        TaskResult::new(json_bytes, lumen_schema::mime::LABELS_V1_JSON)
            .with_result_schema("labels_v1"),
    )
}

#[cfg(test)]
mod tests {
    use super::{BIOCLIP_DEFAULT_TOP_K, top_k_from_request};
    use crate::service::TaskRequest;

    #[test]
    fn top_k_defaults_and_overrides() {
        let base = TaskRequest::new(Vec::<u8>::new(), "image/jpeg");
        assert_eq!(
            top_k_from_request(&base, 100).unwrap(),
            BIOCLIP_DEFAULT_TOP_K
        );

        let with_k = base.clone().with_meta("top_k", "10");
        assert_eq!(top_k_from_request(&with_k, 100).unwrap(), 10);

        let clamped = base.with_meta("top_k", "999");
        assert_eq!(top_k_from_request(&clamped, 7).unwrap(), 7);
    }
}
