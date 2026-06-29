use std::sync::Arc;

use async_trait::async_trait;
use image::{
    RgbImage,
    imageops::{self, FilterType},
};
use lumen_schema::EmbeddingV1;
use serde::{Deserialize, Deserializer, de};

use super::model::{SiglipTextModel, SiglipVisionModel};
use crate::{
    inference_worker,
    service::{
        BatchKey, DEFAULT_TENSOR_MIME, INPUT_KIND_TENSOR, META_INPUT_KIND, META_MODEL_ID,
        META_MODEL_VERSION, PREPROCESS_SIGLIP2_BASE_PATCH16_224_IMAGE,
        PREPROCESS_SIGLIP2_SO400M_PATCH14_384_IMAGE, ServiceError, ServiceResult, TaskHandler,
        TaskRequest, TaskResult, TaskSpec, TensorDescriptor, TensorValidationOptions,
        bytes_to_f32_le, validate_tensor_request,
    },
};

const SUPPORTED_IMAGE_INPUT_MIMES: [&str; 4] =
    ["image/jpeg", "image/png", "image/webp", "image/avif"];
const IMAGE_TENSOR_LAYOUT: &str = "NCHW";
const TENSOR_INPUT_DTYPE: &str = "fp32";

/// SigLIP image preprocessing settings loaded from `model_info.json`
/// `task_metadata.tasks.<image task>.preprocess`.
#[derive(Debug, Clone)]
pub struct SiglipImagePreprocessConfig {
    resize_shortest_edge: u32,
    crop_width: u32,
    crop_height: u32,
    do_resize: bool,
    do_center_crop: bool,
    do_rescale: bool,
    do_normalize: bool,
    rescale_factor: f32,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    filter: FilterType,
    color_space: SiglipImageColorSpace,
    layout: SiglipTensorLayout,
}

impl SiglipImagePreprocessConfig {
    pub fn from_json_str(input: &str) -> Result<Self, String> {
        serde_json::from_str(input)
            .map_err(|err| format!("failed to parse image preprocess metadata: {err}"))
    }

    pub(crate) fn output_shape(&self) -> Vec<usize> {
        debug_assert!(matches!(self.layout, SiglipTensorLayout::Nchw));
        vec![1, 3, self.crop_height as usize, self.crop_width as usize]
    }

    fn preprocess_image_bytes(&self, bytes: &[u8]) -> ServiceResult<Vec<f32>> {
        debug_assert!(matches!(self.color_space, SiglipImageColorSpace::Rgb));
        let image = image::load_from_memory(bytes).map_err(|err| {
            ServiceError::InvalidArgument(format!("failed to decode image: {err}"))
        })?;
        let mut rgb = image.to_rgb8();

        if self.do_center_crop {
            // CLIP-style: resize shortest edge, then center crop.
            if self.do_resize {
                rgb = resize_shortest_edge(&rgb, self.resize_shortest_edge, self.filter);
            }
            rgb = center_crop(&rgb, self.crop_width, self.crop_height, self.filter);
        } else if rgb.width() != self.crop_width || rgb.height() != self.crop_height {
            // SigLIP-style (HF `do_center_crop=false`): a single direct resize
            // to the target size, ignoring aspect ratio. A shortest-edge
            // pre-pass would add a second resampling pass that training-time
            // preprocessing does not have.
            rgb = imageops::resize(&rgb, self.crop_width, self.crop_height, self.filter);
        }

        Ok(rgb_to_nchw_normalized(self, &rgb))
    }

    fn from_raw(raw: RawImagePreprocessConfig) -> Result<Self, String> {
        if raw.resize_shortest_edge == 0 {
            return Err("`resize_shortest_edge` must be greater than zero".to_owned());
        }
        if raw.crop_size.width == 0 || raw.crop_size.height == 0 {
            return Err(
                "`crop_size.width` and `crop_size.height` must be greater than zero".to_owned(),
            );
        }
        if !raw.rescale_factor.is_finite() {
            return Err("`rescale_factor` must be finite".to_owned());
        }

        Ok(Self {
            resize_shortest_edge: raw.resize_shortest_edge,
            crop_width: raw.crop_size.width,
            crop_height: raw.crop_size.height,
            do_resize: raw.do_resize,
            do_center_crop: raw.do_center_crop,
            do_rescale: raw.do_rescale,
            do_normalize: raw.do_normalize,
            rescale_factor: raw.rescale_factor,
            image_mean: vec3(raw.image_mean, "image_mean")?,
            image_std: nonzero_vec3(raw.image_std, "image_std")?,
            filter: raw.resample.into_filter_type(),
            color_space: raw.color_space,
            layout: raw.layout,
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawImagePreprocessConfig {
    resize_shortest_edge: u32,
    crop_size: CropSize,
    do_resize: bool,
    do_center_crop: bool,
    do_rescale: bool,
    do_normalize: bool,
    rescale_factor: f32,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    resample: ResizeFilter,
    color_space: SiglipImageColorSpace,
    layout: SiglipTensorLayout,
}

#[derive(Debug, Deserialize)]
struct CropSize {
    width: u32,
    height: u32,
}

#[derive(Debug, Clone, Copy)]
enum ResizeFilter {
    Nearest,
    Lanczos3,
    Bilinear,
    Bicubic,
}

impl ResizeFilter {
    fn into_filter_type(self) -> FilterType {
        match self {
            ResizeFilter::Nearest => FilterType::Nearest,
            ResizeFilter::Lanczos3 => FilterType::Lanczos3,
            ResizeFilter::Bilinear => FilterType::Triangle,
            ResizeFilter::Bicubic => FilterType::CatmullRom,
        }
    }
}

impl<'de> Deserialize<'de> for ResizeFilter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let normalized = match value {
            serde_json::Value::String(value) => value.trim().to_ascii_lowercase(),
            serde_json::Value::Number(value) => value.to_string(),
            other => {
                return Err(de::Error::custom(format!(
                    "`resample` must be a string or integer, got {other}"
                )));
            }
        };

        match normalized.as_str() {
            "nearest" | "0" => Ok(Self::Nearest),
            "lanczos" | "lanczos3" | "1" => Ok(Self::Lanczos3),
            "bilinear" | "triangle" | "2" => Ok(Self::Bilinear),
            "bicubic" | "catmull_rom" | "catmullrom" | "3" => Ok(Self::Bicubic),
            other => Err(de::Error::custom(format!(
                "unsupported `resample` value `{other}`"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SiglipImageColorSpace {
    Rgb,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SiglipTensorLayout {
    Nchw,
}

impl<'de> Deserialize<'de> for SiglipImagePreprocessConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawImagePreprocessConfig::deserialize(deserializer)?;
        Self::from_raw(raw).map_err(de::Error::custom)
    }
}

fn vec3(values: Vec<f32>, field_name: &str) -> Result<[f32; 3], String> {
    if values.len() != 3 {
        return Err(format!(
            "`{field_name}` must contain exactly 3 values, got {}",
            values.len()
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("`{field_name}` values must be finite"));
    }
    Ok([values[0], values[1], values[2]])
}

fn nonzero_vec3(values: Vec<f32>, field_name: &str) -> Result<[f32; 3], String> {
    let values = vec3(values, field_name)?;
    if values.contains(&0.0) {
        return Err(format!("`{field_name}` values must be non-zero"));
    }
    Ok(values)
}

fn resize_shortest_edge(image: &RgbImage, shortest_edge: u32, filter: FilterType) -> RgbImage {
    let (width, height) = image.dimensions();
    let shortest = width.min(height);
    if shortest == shortest_edge {
        return image.clone();
    }

    let scale = shortest_edge as f32 / shortest as f32;
    let resized_width = ((width as f32 * scale).round() as u32).max(1);
    let resized_height = ((height as f32 * scale).round() as u32).max(1);
    imageops::resize(image, resized_width, resized_height, filter)
}

fn center_crop(
    image: &RgbImage,
    crop_width: u32,
    crop_height: u32,
    filter: FilterType,
) -> RgbImage {
    let image = if image.width() < crop_width || image.height() < crop_height {
        imageops::resize(image, crop_width, crop_height, filter)
    } else {
        image.clone()
    };

    let x = image.width().saturating_sub(crop_width) / 2;
    let y = image.height().saturating_sub(crop_height) / 2;
    imageops::crop_imm(&image, x, y, crop_width, crop_height).to_image()
}

fn rgb_to_nchw_normalized(config: &SiglipImagePreprocessConfig, image: &RgbImage) -> Vec<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let plane = width * height;
    let mut values = vec![0.0; 3 * plane];

    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32).0;
            for channel in 0..3 {
                let mut value = pixel[channel] as f32;
                if config.do_rescale {
                    value *= config.rescale_factor;
                }
                if config.do_normalize {
                    value = (value - config.image_mean[channel]) / config.image_std[channel];
                }
                values[channel * plane + y * width + x] = value;
            }
        }
    }

    values
}

/// L2-normalizes each row (length `row_width`) of `values` in place.
fn normalize_rows(values: &mut [f32], row_width: usize) {
    const EPSILON: f32 = 1e-12;
    if row_width == 0 {
        return;
    }
    for row in values.chunks_mut(row_width) {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm <= EPSILON {
            continue;
        }
        for value in row {
            *value /= norm;
        }
    }
}

/// Task handler for SigLIP text embedding.
pub struct SiglipTextEmbedTask {
    spec: TaskSpec,
    model: Arc<SiglipTextModel>,
    model_id: String,
    tokenizer: Arc<tokenizers::Tokenizer>,
}

impl SiglipTextEmbedTask {
    pub fn new(
        task_name: impl Into<String>,
        model: Arc<SiglipTextModel>,
        model_id: impl Into<String>,
        tokenizer: Arc<tokenizers::Tokenizer>,
    ) -> Self {
        Self {
            spec: TaskSpec::new(task_name, "SigLIP text encoder -> L2-normalized embedding")
                .with_input_mimes(["text/plain"])
                .with_output_mime("application/json;schema=embedding_v1"),
            model,
            model_id: model_id.into(),
            tokenizer,
        }
    }

    fn token_ids(&self, text: &str) -> ServiceResult<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| ServiceError::Internal(format!("tokenization failed: {e}")))?;
        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        // The tokenizer pads to the encoder's sequence length, but guard against
        // misconfiguration so the positional table is never overrun.
        let seq_len = self.model.seq_len;
        if ids.len() != seq_len {
            ids.resize(seq_len, 0);
        }
        Ok(ids)
    }
}

#[async_trait]
impl TaskHandler for SiglipTextEmbedTask {
    fn spec(&self) -> &TaskSpec {
        &self.spec
    }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
        let text = std::str::from_utf8(&request.payload).map_err(|e| {
            ServiceError::InvalidArgument(format!("payload is not valid UTF-8: {e}"))
        })?;
        let ids = self.token_ids(text)?;
        let model = Arc::clone(&self.model);
        let mut embedding = run_blocking(move || model.encode(&ids)).await?;
        let len = embedding.len();
        normalize_rows(&mut embedding, len);
        embedding_json_result(EmbeddingV1::new(embedding, &self.model_id))
    }
}

/// Task handler for SigLIP image embedding.
pub struct SiglipImageEmbedTask {
    spec: TaskSpec,
    model: Arc<SiglipVisionModel>,
    model_id: String,
    preprocess: SiglipImagePreprocessConfig,
    tensor_preprocess_id: String,
}

impl SiglipImageEmbedTask {
    pub fn new(
        task_name: impl Into<String>,
        model: Arc<SiglipVisionModel>,
        model_id: impl Into<String>,
        preprocess: SiglipImagePreprocessConfig,
    ) -> Self {
        let tensor_preprocess_id = siglip_tensor_preprocess_id(&preprocess.output_shape())
            .unwrap_or(PREPROCESS_SIGLIP2_BASE_PATCH16_224_IMAGE)
            .to_owned();
        Self {
            spec: TaskSpec::new(
                task_name,
                "SigLIP vision encoder -> L2-normalized embedding",
            )
            .with_input_mimes(image_input_mimes_with_tensor())
            .with_output_mime("application/json;schema=embedding_v1")
            .with_tensor_fast_path(&tensor_preprocess_id, true),
            model,
            model_id: model_id.into(),
            preprocess,
            tensor_preprocess_id,
        }
    }

    fn tensor_input_descriptor(&self, request: &TaskRequest) -> ServiceResult<TensorDescriptor> {
        let descriptor = validate_tensor_request(
            request,
            TensorValidationOptions {
                dtype: TENSOR_INPUT_DTYPE,
                layout: IMAGE_TENSOR_LAYOUT,
                preprocess_id: &self.tensor_preprocess_id,
            },
        )?;
        let expected_shape = self.preprocess.output_shape();
        if descriptor.shape != expected_shape {
            return Err(ServiceError::InvalidArgument(format!(
                "SigLIP image tensor shape must be {:?}, got {:?}",
                expected_shape, descriptor.shape
            )));
        }
        Ok(descriptor)
    }

    /// Runs the vision encoder on a flattened `[batch, 3, H, W]` buffer and
    /// returns L2-normalized embedding rows plus, when an aesthetic head is
    /// loaded, one score per row (from the same forward pass).
    async fn embed_pixels(
        &self,
        pixels: Vec<f32>,
        batch: usize,
    ) -> ServiceResult<(Vec<f32>, Option<Vec<f32>>)> {
        let shape = self.preprocess.output_shape();
        let (height, width) = (shape[2], shape[3]);
        let model = Arc::clone(&self.model);
        let (mut raw, scores) =
            run_blocking(move || model.encode(pixels, batch, height, width)).await?;
        let row_width = raw.len().checked_div(batch.max(1)).unwrap_or(0);
        normalize_rows(&mut raw, row_width);
        Ok((raw, scores))
    }
}

/// Builds an `embedding_v1` result, attaching the aesthetic score when present.
fn embedding_result(
    vector: Vec<f32>,
    model_id: &str,
    score: Option<f32>,
) -> ServiceResult<TaskResult> {
    let mut embedding = EmbeddingV1::new(vector, model_id);
    if let Some(score) = score {
        embedding = embedding.with_aesthetic_score(score);
    }
    embedding_json_result(embedding)
}

#[async_trait]
impl TaskHandler for SiglipImageEmbedTask {
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
            self.tensor_preprocess_id
        ))))
    }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
        if normalized_meta(request.meta.get(META_INPUT_KIND)).as_deref() == Some(INPUT_KIND_TENSOR)
        {
            self.tensor_input_descriptor(&request)?;
            let pixels = bytes_to_f32_le(&request.payload)?;
            let (embedding, scores) = self.embed_pixels(pixels, 1).await?;
            return embedding_result(embedding, &self.model_id, first_score(&scores));
        }

        if !is_supported_image_input_mime(&request.payload_mime) {
            return Err(ServiceError::InvalidArgument(format!(
                "unsupported SigLIP image input MIME `{}`; supported MIME types: {}",
                request.payload_mime,
                SUPPORTED_IMAGE_INPUT_MIMES.join(", ")
            )));
        }
        let pixels = self.preprocess.preprocess_image_bytes(&request.payload)?;
        let (embedding, scores) = self.embed_pixels(pixels, 1).await?;
        embedding_result(embedding, &self.model_id, first_score(&scores))
    }

    async fn handle_batch(&self, requests: Vec<TaskRequest>) -> ServiceResult<Vec<TaskResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let batch_len = requests.len();
        let mut pixels = Vec::new();
        for request in &requests {
            self.tensor_input_descriptor(request)?;
            pixels.extend(bytes_to_f32_le(&request.payload)?);
        }

        let (embeddings, scores) = self.embed_pixels(pixels, batch_len).await?;
        let row_width = embeddings.len().checked_div(batch_len).ok_or_else(|| {
            ServiceError::Internal("SigLIP batch output has invalid batch size".to_owned())
        })?;
        if row_width * batch_len != embeddings.len() {
            return Err(ServiceError::Internal(format!(
                "SigLIP batch output element count {} is not divisible by batch size {batch_len}",
                embeddings.len()
            )));
        }

        embeddings
            .chunks(row_width)
            .enumerate()
            .map(|(row_index, row)| {
                let score = scores.as_ref().and_then(|s| s.get(row_index).copied());
                embedding_result(row.to_vec(), &self.model_id, score)
            })
            .collect()
    }
}

/// First (single-image) score from an optional per-row score vector.
fn first_score(scores: &Option<Vec<f32>>) -> Option<f32> {
    scores.as_ref().and_then(|s| s.first().copied())
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

fn siglip_tensor_preprocess_id(shape: &[usize]) -> Option<&'static str> {
    match shape {
        [1, 3, 224, 224] => Some(PREPROCESS_SIGLIP2_BASE_PATCH16_224_IMAGE),
        [1, 3, 384, 384] => Some(PREPROCESS_SIGLIP2_SO400M_PATCH14_384_IMAGE),
        _ => None,
    }
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

fn embedding_json_result(embedding: EmbeddingV1) -> ServiceResult<TaskResult> {
    let json_bytes = embedding
        .to_json_bytes()
        .map_err(|e| ServiceError::Internal(e.to_string()))?;

    Ok(
        TaskResult::new(json_bytes, "application/json;schema=embedding_v1")
            .with_result_schema("embedding_v1"),
    )
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{SiglipImagePreprocessConfig, is_supported_image_input_mime, normalize_rows};

    #[test]
    fn image_preprocess_config_parses_siglip_metadata() {
        let config = SiglipImagePreprocessConfig::from_json_str(
            &json!({
                "resize_shortest_edge": 224,
                "crop_size": { "width": 224, "height": 224 },
                "do_resize": true,
                "do_center_crop": false,
                "do_rescale": true,
                "do_normalize": true,
                "rescale_factor": 0.00392156862745098,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "resample": "bilinear",
                "color_space": "rgb",
                "layout": "nchw"
            })
            .to_string(),
        )
        .unwrap();

        assert_eq!(config.output_shape(), vec![1, 3, 224, 224]);
    }

    #[test]
    fn image_preprocess_config_requires_explicit_metadata() {
        let err = SiglipImagePreprocessConfig::from_json_str(
            &json!({
                "resize_shortest_edge": 224,
                "crop_size": { "width": 224, "height": 224 },
                "do_resize": true,
                "do_center_crop": false,
                "do_rescale": true,
                "do_normalize": true,
                "rescale_factor": 0.00392156862745098,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "resample": "bilinear",
                "layout": "nchw"
            })
            .to_string(),
        )
        .unwrap_err();

        assert!(err.contains("color_space"));
    }

    #[test]
    fn image_input_mime_support_is_explicit() {
        assert!(is_supported_image_input_mime("image/jpeg"));
        assert!(is_supported_image_input_mime("IMAGE/JPEG; charset=binary"));
        assert!(!is_supported_image_input_mime("image/gif"));
    }

    #[test]
    fn normalize_rows_normalizes_each_row() {
        let mut values = vec![3.0, 4.0, 5.0, 12.0];
        normalize_rows(&mut values, 2);
        assert!((values[0] - 0.6).abs() < 1e-6);
        assert!((values[1] - 0.8).abs() < 1e-6);
        assert!((values[2] - 5.0 / 13.0).abs() < 1e-6);
        assert!((values[3] - 12.0 / 13.0).abs() < 1e-6);
    }
}
