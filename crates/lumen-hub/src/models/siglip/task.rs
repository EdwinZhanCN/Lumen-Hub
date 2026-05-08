use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use half::f16;
use image::{
    RgbImage,
    imageops::{self, FilterType},
};
use lumen_schema::EmbeddingV1;
use lumnn::core::{
    context::MLContext,
    packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor},
    pipeline::MLPipeline,
};
use serde::{Deserialize, Deserializer, de};

use crate::service::{ServiceError, ServiceResult, TaskHandler, TaskRequest, TaskResult, TaskSpec};

const SUPPORTED_IMAGE_INPUT_MIMES: [&str; 4] =
    ["image/jpeg", "image/png", "image/webp", "image/avif"];

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

        if self.do_resize {
            rgb = resize_shortest_edge(&rgb, self.resize_shortest_edge, self.filter);
        }

        if self.do_center_crop {
            rgb = center_crop(&rgb, self.crop_width, self.crop_height, self.filter);
        } else if rgb.width() != self.crop_width || rgb.height() != self.crop_height {
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

/// Task handler for SigLIP text embedding.
pub struct SiglipTextEmbedTask {
    spec: TaskSpec,
    pipeline: Arc<MLPipeline>,
    context: Arc<MLContext>,
    model_id: String,
    input_names: Vec<String>,
    output_name: String,
    tokenizer: tokenizers::Tokenizer,
}

impl SiglipTextEmbedTask {
    pub fn new(
        task_name: impl Into<String>,
        pipeline: MLPipeline,
        context: Arc<MLContext>,
        model_id: impl Into<String>,
        input_names: Vec<String>,
        output_name: impl Into<String>,
        tokenizer: tokenizers::Tokenizer,
    ) -> ServiceResult<Self> {
        if input_names.is_empty() {
            return Err(ServiceError::InvalidArgument(
                "SigLIP text task requires at least one ONNX input name".to_owned(),
            ));
        }
        Ok(Self {
            spec: TaskSpec::new(task_name, "SigLIP text encoder -> L2-normalized embedding")
                .with_input_mimes(["text/plain"])
                .with_output_mime("application/json;schema=embedding_v1"),
            pipeline: Arc::new(pipeline),
            context,
            model_id: model_id.into(),
            input_names,
            output_name: output_name.into(),
            tokenizer,
        })
    }

    fn preprocess_text(&self, text: &str) -> ServiceResult<HashMap<String, MLPacket>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| ServiceError::Internal(format!("tokenization failed: {e}")))?;

        let ids = encoding
            .get_ids()
            .iter()
            .map(|&id| id as i64)
            .collect::<Vec<_>>();
        let seq_len = ids.len();
        let mut packets = HashMap::new();

        let input_ids_packet = self
            .context
            .packet_from_host_tensor(
                MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1, seq_len]),
                HostTensor::Int64(ids),
            )
            .map_err(ServiceError::Internal)?;
        packets.insert(self.input_names[0].clone(), input_ids_packet);

        if self.input_names.len() > 1 {
            let attention_mask = encoding
                .get_attention_mask()
                .iter()
                .map(|&mask| mask as i64)
                .collect::<Vec<_>>();
            let attention_mask_packet = self
                .context
                .packet_from_host_tensor(
                    MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1, seq_len]),
                    HostTensor::Int64(attention_mask),
                )
                .map_err(ServiceError::Internal)?;
            packets.insert(self.input_names[1].clone(), attention_mask_packet);
        }

        Ok(packets)
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
        let input_packets = self.preprocess_text(text)?;
        let mut outputs = self
            .pipeline
            .run(input_packets)
            .await
            .map_err(ServiceError::Internal)?;
        let embedding_packet = outputs.remove(&self.output_name).ok_or_else(|| {
            ServiceError::Internal(format!(
                "pipeline output missing key `{}`",
                self.output_name
            ))
        })?;
        let embedding = embedding_from_packet(embedding_packet, &self.model_id).await?;
        let json_bytes = embedding
            .to_json_bytes()
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        Ok(
            TaskResult::new(json_bytes, "application/json;schema=embedding_v1")
                .with_result_schema("embedding_v1"),
        )
    }
}

/// Task handler for SigLIP image embedding.
pub struct SiglipImageEmbedTask {
    spec: TaskSpec,
    pipeline: Arc<MLPipeline>,
    context: Arc<MLContext>,
    model_id: String,
    input_name: String,
    input_dtype: MLPacketDataType,
    output_name: String,
    preprocess: SiglipImagePreprocessConfig,
}

impl SiglipImageEmbedTask {
    pub fn new(
        task_name: impl Into<String>,
        pipeline: MLPipeline,
        context: Arc<MLContext>,
        model_id: impl Into<String>,
        input_names: Vec<String>,
        input_dtype: MLPacketDataType,
        output_name: impl Into<String>,
        preprocess: SiglipImagePreprocessConfig,
    ) -> ServiceResult<Self> {
        let input_name = input_names.into_iter().next().ok_or_else(|| {
            ServiceError::InvalidArgument(
                "SigLIP image task requires one ONNX input name".to_owned(),
            )
        })?;
        Ok(Self {
            spec: TaskSpec::new(
                task_name,
                "SigLIP vision encoder -> L2-normalized embedding",
            )
            .with_input_mimes(SUPPORTED_IMAGE_INPUT_MIMES)
            .with_output_mime("application/json;schema=embedding_v1"),
            pipeline: Arc::new(pipeline),
            context,
            model_id: model_id.into(),
            input_name,
            input_dtype,
            output_name: output_name.into(),
            preprocess,
        })
    }

    fn preprocess_image(&self, bytes: &[u8]) -> ServiceResult<HashMap<String, MLPacket>> {
        let pixel_values = self.preprocess.preprocess_image_bytes(bytes)?;
        let tensor = match self.input_dtype {
            MLPacketDataType::Float32 => HostTensor::Float32(pixel_values),
            MLPacketDataType::Float16 => {
                HostTensor::Float16(pixel_values.into_iter().map(f16::from_f32).collect())
            }
            other => {
                return Err(ServiceError::Internal(format!(
                    "SigLIP image ONNX input `{}` has unsupported dtype {:?}",
                    self.input_name, other
                )));
            }
        };
        let packet = self
            .context
            .packet_from_host_tensor(
                MLPacketDescriptor::new(self.input_dtype, self.preprocess.output_shape()),
                tensor,
            )
            .map_err(ServiceError::Internal)?;

        Ok(HashMap::from([(self.input_name.clone(), packet)]))
    }
}

#[async_trait]
impl TaskHandler for SiglipImageEmbedTask {
    fn spec(&self) -> &TaskSpec {
        &self.spec
    }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
        if !is_supported_image_input_mime(&request.payload_mime) {
            return Err(ServiceError::InvalidArgument(format!(
                "unsupported SigLIP image input MIME `{}`; supported MIME types: {}",
                request.payload_mime,
                SUPPORTED_IMAGE_INPUT_MIMES.join(", ")
            )));
        }
        let input_packets = self.preprocess_image(&request.payload)?;
        let mut outputs = self
            .pipeline
            .run(input_packets)
            .await
            .map_err(ServiceError::Internal)?;
        let embedding_packet = outputs.remove(&self.output_name).ok_or_else(|| {
            ServiceError::Internal(format!(
                "pipeline output missing key `{}`",
                self.output_name
            ))
        })?;
        let embedding = embedding_from_packet(embedding_packet, &self.model_id).await?;
        let json_bytes = embedding
            .to_json_bytes()
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        Ok(
            TaskResult::new(json_bytes, "application/json;schema=embedding_v1")
                .with_result_schema("embedding_v1"),
        )
    }
}

async fn embedding_from_packet(packet: MLPacket, model_id: &str) -> ServiceResult<EmbeddingV1> {
    let tensor = packet
        .to_host_tensor()
        .await
        .map_err(ServiceError::Internal)?;

    match tensor {
        HostTensor::Float32(values) => Ok(EmbeddingV1::new(values, model_id)),
        other => Err(ServiceError::Internal(format!(
            "unexpected tensor type from SigLIP pipeline: {other:?}"
        ))),
    }
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

    use super::{SiglipImagePreprocessConfig, is_supported_image_input_mime};

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
                "resample": "bicubic",
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
                "resample": "bicubic",
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
        assert!(is_supported_image_input_mime("image/png"));
        assert!(is_supported_image_input_mime("image/webp"));
        assert!(is_supported_image_input_mime("image/avif"));
        assert!(is_supported_image_input_mime("IMAGE/JPEG; charset=binary"));

        assert!(!is_supported_image_input_mime("image/gif"));
        assert!(!is_supported_image_input_mime("image/*"));
        assert!(!is_supported_image_input_mime("application/octet-stream"));
    }
}
