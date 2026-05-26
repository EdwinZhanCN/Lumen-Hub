use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use half::f16;
use hnswlib_rs::{Hnsw, InnerProduct, legacy::LegacyVectors};
use image::{
    RgbImage,
    imageops::{self, FilterType},
};
use lumen_schema::{EmbeddingV1, Label, LabelsV1};
use lumnn::core::{
    context::MLContext,
    packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor},
    pipeline::MLPipeline,
};
use serde::{Deserialize, Deserializer, de};

use crate::service::{
    BatchKey, DEFAULT_TENSOR_MIME, INPUT_KIND_TENSOR, META_INPUT_KIND, META_MODEL_ID,
    META_MODEL_VERSION, ServiceError, ServiceResult, TaskHandler, TaskRequest, TaskResult,
    TaskSpec, TensorDescriptor, TensorValidationOptions, bytes_to_f16_le, bytes_to_f32_le,
    validate_tensor_request,
};

const BIOCLIP_DEFAULT_TOP_K: usize = 5;
const BIOCLIP_HNSW_RERANK_K: usize = 200;
const BIOCLIP_TOP_K_META_KEYS: [&str; 5] = ["TopK", "topK", "top_k", "top-k", "lumen.top_k"];
const SUPPORTED_IMAGE_INPUT_MIMES: [&str; 4] =
    ["image/jpeg", "image/png", "image/webp", "image/avif"];
const IMAGE_TENSOR_LAYOUT: &str = "NCHW";
const CLIP_IMAGE_PREPROCESS_ID: &str = "clip_image_preprocess_v1";

/// CLIP image preprocessing settings loaded from `model_info.json`
/// `task_metadata.tasks.<image task>.preprocess`.
#[derive(Debug, Clone)]
pub struct ClipImagePreprocessConfig {
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
    color_space: ClipImageColorSpace,
    layout: ClipTensorLayout,
}

impl ClipImagePreprocessConfig {
    pub fn from_json_str(input: &str) -> Result<Self, String> {
        serde_json::from_str(input)
            .map_err(|err| format!("failed to parse image preprocess metadata: {err}"))
    }

    pub(crate) fn output_shape(&self) -> Vec<usize> {
        debug_assert!(matches!(self.layout, ClipTensorLayout::Nchw));
        vec![1, 3, self.crop_height as usize, self.crop_width as usize]
    }

    fn preprocess_image_bytes(&self, bytes: &[u8]) -> ServiceResult<Vec<f32>> {
        debug_assert!(matches!(self.color_space, ClipImageColorSpace::Rgb));
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
    color_space: ClipImageColorSpace,
    layout: ClipTensorLayout,
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
enum ClipImageColorSpace {
    Rgb,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ClipTensorLayout {
    Nchw,
}

impl<'de> Deserialize<'de> for ClipImagePreprocessConfig {
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

fn rgb_to_nchw_normalized(config: &ClipImagePreprocessConfig, image: &RgbImage) -> Vec<f32> {
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

/// Task handler for CLIP **text** embedding.
///
/// Wraps a pipeline (ONNX forward + L2 normalize) and a HuggingFace tokenizer
/// to convert raw text into an `EmbeddingV1` JSON response.
pub struct ClipTextEmbedTask {
    spec: TaskSpec,
    pipeline: Arc<MLPipeline>,
    context: Arc<MLContext>,
    model_id: String,
    input_names: Vec<String>,
    output_name: String,
    tokenizer: tokenizers::Tokenizer,
}

/// Task handler for CLIP **image** embedding.
///
/// Decodes an image payload, applies the CLIP image preprocessing declared in
/// `model_info.json`, runs the vision encoder, and returns an L2-normalized
/// `EmbeddingV1` JSON response.
pub struct ClipImageEmbedTask {
    spec: TaskSpec,
    pipeline: Arc<MLPipeline>,
    context: Arc<MLContext>,
    model_id: String,
    input_name: String,
    input_dtype: MLPacketDataType,
    output_name: String,
    preprocess: ClipImagePreprocessConfig,
}

/// Task handler for BioCLIP zero-shot classification.
///
/// BioCLIP does not load a text encoder at inference time. It runs the CLIP
/// vision encoder, then compares the normalized image feature with a memory
/// mapped matrix of precomputed text embeddings for the configured dataset.
pub struct BioClipClassifyTask {
    spec: TaskSpec,
    pipeline: Arc<MLPipeline>,
    context: Arc<MLContext>,
    model_id: String,
    input_name: String,
    input_dtype: MLPacketDataType,
    output_name: String,
    preprocess: ClipImagePreprocessConfig,
    logit_scale: f32,
    dataset: Arc<BioClipDataset>,
}

impl BioClipClassifyTask {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        task_name: impl Into<String>,
        pipeline: MLPipeline,
        context: Arc<MLContext>,
        model_id: impl Into<String>,
        input_names: Vec<String>,
        input_dtype: MLPacketDataType,
        output_name: impl Into<String>,
        preprocess: ClipImagePreprocessConfig,
        logit_scale: f32,
        dataset_name: impl Into<String>,
        embeddings_path: impl Into<PathBuf>,
        labels_path: impl Into<PathBuf>,
        index_path: Option<PathBuf>,
    ) -> ServiceResult<Self> {
        let input_name = input_names.into_iter().next().ok_or_else(|| {
            ServiceError::InvalidArgument(
                "BioCLIP classify task requires one vision input name".to_owned(),
            )
        })?;
        let dataset_name = dataset_name.into();
        let dataset = BioClipDataset::open(
            dataset_name.clone(),
            embeddings_path.into(),
            labels_path.into(),
            index_path,
        )?;
        let output_name = output_name.into();
        Ok(Self {
            spec: TaskSpec::new(
                task_name,
                "BioCLIP image classification with dataset text embeddings",
            )
            .with_input_mimes(image_input_mimes_with_tensor())
            .with_output_mime(lumen_schema::mime::LABELS_V1_JSON)
            .with_metadata("dataset", dataset_name),
            pipeline: Arc::new(pipeline),
            context,
            model_id: model_id.into(),
            input_name,
            input_dtype,
            output_name,
            preprocess,
            logit_scale,
            dataset: Arc::new(dataset),
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
                    "BioCLIP image input `{}` has unsupported dtype {:?}",
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

    fn tensor_input_descriptor(&self, request: &TaskRequest) -> ServiceResult<TensorDescriptor> {
        let descriptor = validate_tensor_request(
            request,
            TensorValidationOptions {
                dtype: ml_dtype_to_tensor_dtype(self.input_dtype)?,
                layout: IMAGE_TENSOR_LAYOUT,
                preprocess_id: CLIP_IMAGE_PREPROCESS_ID,
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

    fn tensor_request_to_packets(
        &self,
        request: &TaskRequest,
    ) -> ServiceResult<HashMap<String, MLPacket>> {
        let descriptor = self.tensor_input_descriptor(request)?;
        let tensor = tensor_payload_to_host_tensor(&request.payload, self.input_dtype)?;
        let packet = self
            .context
            .packet_from_host_tensor(
                MLPacketDescriptor::new(self.input_dtype, descriptor.shape),
                tensor,
            )
            .map_err(ServiceError::Internal)?;
        Ok(HashMap::from([(self.input_name.clone(), packet)]))
    }

    async fn run_pipeline(
        &self,
        input_packets: HashMap<String, MLPacket>,
    ) -> ServiceResult<MLPacket> {
        let mut outputs = self
            .pipeline
            .run(input_packets)
            .await
            .map_err(ServiceError::Internal)?;

        outputs.remove(&self.output_name).ok_or_else(|| {
            ServiceError::Internal(format!(
                "pipeline output missing key `{}`",
                self.output_name
            ))
        })
    }

    async fn labels_result_from_packet(
        &self,
        packet: MLPacket,
        top_k: usize,
    ) -> ServiceResult<TaskResult> {
        let (values, _) = float32_values_and_shape(packet).await?;
        let labels = self.dataset.top_k(&values, top_k, self.logit_scale)?;
        labels_json_result(LabelsV1::new(labels, &self.model_id))
    }

    fn batched_tensor_packets(
        &self,
        requests: &[TaskRequest],
    ) -> ServiceResult<HashMap<String, MLPacket>> {
        let expected_shape = self.preprocess.output_shape();
        let mut batched_shape = expected_shape.clone();
        batched_shape[0] = requests.len();

        let tensor = match self.input_dtype {
            MLPacketDataType::Float32 => {
                let mut values = Vec::new();
                for request in requests {
                    self.tensor_input_descriptor(request)?;
                    values.extend(bytes_to_f32_le(&request.payload)?);
                }
                HostTensor::Float32(values)
            }
            MLPacketDataType::Float16 => {
                let mut values = Vec::new();
                for request in requests {
                    self.tensor_input_descriptor(request)?;
                    values.extend(bytes_to_f16_le(&request.payload)?);
                }
                HostTensor::Float16(values)
            }
            other => {
                return Err(ServiceError::Internal(format!(
                    "BioCLIP image ONNX input `{}` has unsupported dtype {:?}",
                    self.input_name, other
                )));
            }
        };

        let packet = self
            .context
            .packet_from_host_tensor(
                MLPacketDescriptor::new(self.input_dtype, batched_shape),
                tensor,
            )
            .map_err(ServiceError::Internal)?;

        Ok(HashMap::from([(self.input_name.clone(), packet)]))
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
            CLIP_IMAGE_PREPROCESS_ID
        ))))
    }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
        let top_k = top_k_from_request(&request, self.dataset.len())?;
        let packet = if normalized_meta(request.meta.get(META_INPUT_KIND)).as_deref()
            == Some(INPUT_KIND_TENSOR)
        {
            self.run_pipeline(self.tensor_request_to_packets(&request)?)
                .await?
        } else {
            if !is_supported_image_input_mime(&request.payload_mime) {
                return Err(ServiceError::InvalidArgument(format!(
                    "unsupported BioCLIP image input MIME `{}`; supported MIME types: {}",
                    request.payload_mime,
                    SUPPORTED_IMAGE_INPUT_MIMES.join(", ")
                )));
            }
            self.run_pipeline(self.preprocess_image(&request.payload)?)
                .await?
        };

        self.labels_result_from_packet(packet, top_k).await
    }

    async fn handle_batch(&self, requests: Vec<TaskRequest>) -> ServiceResult<Vec<TaskResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let batch_len = requests.len();
        let top_k_values: Vec<usize> = requests
            .iter()
            .map(|request| top_k_from_request(request, self.dataset.len()))
            .collect::<ServiceResult<_>>()?;

        let packet = self
            .run_pipeline(self.batched_tensor_packets(&requests)?)
            .await?;
        let (values, shape) = float32_values_and_shape(packet).await?;
        if shape.first().copied() != Some(batch_len) {
            return Err(ServiceError::Internal(format!(
                "BioCLIP batch output shape {:?} does not match batch size {batch_len}",
                shape
            )));
        }
        let row_width = values.len().checked_div(batch_len).ok_or_else(|| {
            ServiceError::Internal("BioCLIP batch output has invalid batch size".to_owned())
        })?;
        if row_width * batch_len != values.len() {
            return Err(ServiceError::Internal(format!(
                "BioCLIP batch output element count {} is not divisible by batch size {batch_len}",
                values.len()
            )));
        }

        values
            .chunks(row_width)
            .zip(top_k_values)
            .map(|(row, top_k)| {
                let labels = self.dataset.top_k(row, top_k, self.logit_scale)?;
                labels_json_result(LabelsV1::new(labels, &self.model_id))
            })
            .collect()
    }
}

struct BioClipDataset {
    name: String,
    embeddings: MmapNpyMatrix,
    labels: Vec<String>,
    layout: BioClipEmbeddingLayout,
    index: Option<(Hnsw<u64, InnerProduct>, LegacyVectors<'static>)>,
    #[allow(dead_code)]
    mmap: Option<memmap2::Mmap>,
}

impl BioClipDataset {
    fn open(
        name: String,
        embeddings_path: PathBuf,
        labels_path: PathBuf,
        index_path: Option<PathBuf>,
    ) -> ServiceResult<Self> {
        let embeddings = MmapNpyMatrix::open(&embeddings_path)?;
        let labels = load_bioclip_labels(&labels_path)?;
        let layout = BioClipEmbeddingLayout::from_shape(labels.len(), &embeddings).ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "BioCLIP dataset `{name}` label count {} from {} does not match embedding matrix shape [{}, {}] from {}",
                labels.len(),
                labels_path.display(),
                embeddings.rows(),
                embeddings.cols(),
                embeddings_path.display()
            ))
        })?;
        if layout.embedding_dim(&embeddings) == 0 {
            return Err(ServiceError::InvalidArgument(format!(
                "BioCLIP dataset `{name}` embedding dimension must be greater than zero"
            )));
        }

        let mut mmap_owner = None;
        let index = if let Some(path) = index_path {
            let file = std::fs::File::open(&path).map_err(|err| {
                ServiceError::InvalidArgument(format!(
                    "failed to open BioCLIP HNSW index at {}: {err}",
                    path.display()
                ))
            })?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|err| {
                ServiceError::InvalidArgument(format!(
                    "failed to mmap BioCLIP HNSW index at {}: {err}",
                    path.display()
                ))
            })?;
            let dim = layout.embedding_dim(&embeddings);

            // Extend lifetime of the mmap reference to 'static.
            // This is safe because mmap is stored alongside the index inside BioClipDataset and stays alive.
            let mmap_static_ref: &'static [u8] = unsafe { std::mem::transmute(&mmap[..]) };
            let (graph, vectors) =
                hnswlib_rs::legacy::load_hnswlib(InnerProduct::new(), dim, mmap_static_ref)
                    .map_err(|err| {
                        ServiceError::Internal(format!(
                            "failed to load legacy HNSW index at {}: {:?}",
                            path.display(),
                            err
                        ))
                    })?;

            graph.set_ef_search(256);

            mmap_owner = Some(mmap);
            Some((graph, vectors))
        } else {
            None
        };

        Ok(Self {
            name,
            embeddings,
            labels,
            layout,
            index,
            mmap: mmap_owner,
        })
    }

    fn len(&self) -> usize {
        self.labels.len()
    }

    fn top_k(&self, query: &[f32], top_k: usize, logit_scale: f32) -> ServiceResult<Vec<Label>> {
        let embedding_dim = self.layout.embedding_dim(&self.embeddings);
        if query.len() != embedding_dim {
            return Err(ServiceError::Internal(format!(
                "BioCLIP image embedding dimension {} does not match dataset `{}` dimension {}",
                query.len(),
                self.name,
                embedding_dim
            )));
        }
        let query_norm = l2_norm(query);
        if query_norm == 0.0 {
            return Err(ServiceError::Internal(
                "BioCLIP image embedding has zero norm".to_owned(),
            ));
        }

        let limit = top_k.min(self.labels.len());
        let best = if let Some((graph, vectors)) = &self.index {
            let search_k = BIOCLIP_HNSW_RERANK_K.min(self.labels.len());
            let search_hits = graph
                .search(vectors, query, search_k, None)
                .map_err(|err| ServiceError::Internal(format!("HNSW search failed: {:?}", err)))?;
            let candidates: Vec<usize> = search_hits
                .into_iter()
                .map(|hit| hit.key as usize)
                .collect();
            self.rerank_candidates(query, query_norm, limit, candidates, logit_scale)?
        } else {
            match self.layout {
                BioClipEmbeddingLayout::LabelRows => {
                    self.top_k_label_rows(query, query_norm, limit, logit_scale)?
                }
                BioClipEmbeddingLayout::LabelColumns => {
                    self.top_k_label_columns(query, query_norm, limit, logit_scale)?
                }
            }
        };

        Ok(best
            .into_iter()
            .map(|(index, score)| Label {
                label: self.labels[index].clone(),
                score,
            })
            .collect())
    }

    fn top_k_label_rows(
        &self,
        query: &[f32],
        query_norm: f32,
        limit: usize,
        logit_scale: f32,
    ) -> ServiceResult<Vec<(usize, f32)>> {
        let rows = self.embeddings.rows();
        let scale = logit_scale.exp();

        let mut similarities = Vec::with_capacity(rows);
        let mut max_similarity = f32::NEG_INFINITY;
        for label_index in 0..rows {
            let sim = self
                .embeddings
                .row_cosine_similarity(label_index, query, query_norm)?;
            similarities.push(sim);
            if sim > max_similarity && sim.is_finite() {
                max_similarity = sim;
            }
        }

        if max_similarity == f32::NEG_INFINITY {
            return Ok(Vec::new());
        }

        let max_scaled = max_similarity * scale;
        let mut sum_exp = 0.0f32;
        for &sim in &similarities {
            if sim.is_finite() {
                sum_exp += ((sim * scale) - max_scaled).exp();
            }
        }

        let mut best: Vec<(usize, f32)> = Vec::with_capacity(limit);
        for (label_index, &sim) in similarities.iter().enumerate() {
            if sim.is_finite() && sum_exp > 0.0 {
                let softmax_score = ((sim * scale) - max_scaled).exp() / sum_exp;
                push_top_k(&mut best, limit, label_index, softmax_score);
            }
        }
        Ok(best)
    }

    fn rerank_candidates(
        &self,
        query: &[f32],
        query_norm: f32,
        limit: usize,
        candidates: Vec<usize>,
        logit_scale: f32,
    ) -> ServiceResult<Vec<(usize, f32)>> {
        let scale = logit_scale.exp();
        let mut candidate_sims = Vec::with_capacity(candidates.len());
        let mut max_similarity = f32::NEG_INFINITY;

        for label_index in candidates {
            if label_index >= self.labels.len() {
                continue;
            }
            let sim = match self.layout {
                BioClipEmbeddingLayout::LabelRows => {
                    self.embeddings
                        .row_cosine_similarity(label_index, query, query_norm)?
                }
                BioClipEmbeddingLayout::LabelColumns => {
                    self.embeddings
                        .column_cosine_similarity(label_index, query, query_norm)?
                }
            };
            candidate_sims.push((label_index, sim));
            if sim > max_similarity && sim.is_finite() {
                max_similarity = sim;
            }
        }

        if max_similarity == f32::NEG_INFINITY {
            return Ok(Vec::new());
        }

        let max_scaled = max_similarity * scale;
        let mut sum_exp = 0.0f32;
        for &(_, sim) in &candidate_sims {
            if sim.is_finite() {
                sum_exp += ((sim * scale) - max_scaled).exp();
            }
        }

        let mut best: Vec<(usize, f32)> = Vec::with_capacity(limit);
        for &(label_index, sim) in &candidate_sims {
            if sim.is_finite() && sum_exp > 0.0 {
                let softmax_score = ((sim * scale) - max_scaled).exp() / sum_exp;
                push_top_k(&mut best, limit, label_index, softmax_score);
            }
        }
        Ok(best)
    }

    fn top_k_label_columns(
        &self,
        query: &[f32],
        query_norm: f32,
        limit: usize,
        logit_scale: f32,
    ) -> ServiceResult<Vec<(usize, f32)>> {
        let label_count = self.labels.len();
        let mut dots = vec![0.0f32; label_count];
        let mut norms_sq = vec![0.0f32; label_count];

        for (dim, query_value) in query.iter().copied().enumerate() {
            match self.embeddings.element_type {
                NpyElementType::Float32 => {
                    for (label_index, chunk) in
                        self.embeddings.row_bytes(dim).chunks_exact(4).enumerate()
                    {
                        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        dots[label_index] += value * query_value;
                        norms_sq[label_index] += value * value;
                    }
                }
                NpyElementType::Float16 => {
                    for (label_index, chunk) in
                        self.embeddings.row_bytes(dim).chunks_exact(2).enumerate()
                    {
                        let value = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                        dots[label_index] += value * query_value;
                        norms_sq[label_index] += value * value;
                    }
                }
            }
        }

        let mut similarities = Vec::with_capacity(label_count);
        let mut max_similarity = f32::NEG_INFINITY;
        for label_index in 0..label_count {
            let norm_sq = norms_sq[label_index];
            if norm_sq == 0.0 {
                similarities.push(f32::NAN);
                continue;
            }
            let sim = dots[label_index] / (norm_sq.sqrt() * query_norm);
            similarities.push(sim);
            if sim > max_similarity && sim.is_finite() {
                max_similarity = sim;
            }
        }

        if max_similarity == f32::NEG_INFINITY {
            return Ok(Vec::new());
        }

        let scale = logit_scale.exp();
        let max_scaled = max_similarity * scale;
        let mut sum_exp = 0.0f32;
        for &sim in &similarities {
            if sim.is_finite() {
                sum_exp += ((sim * scale) - max_scaled).exp();
            }
        }

        let mut best: Vec<(usize, f32)> = Vec::with_capacity(limit);
        for (label_index, &sim) in similarities.iter().enumerate() {
            if sim.is_finite() && sum_exp > 0.0 {
                let softmax_score = ((sim * scale) - max_scaled).exp() / sum_exp;
                push_top_k(&mut best, limit, label_index, softmax_score);
            }
        }
        Ok(best)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BioClipEmbeddingLayout {
    /// Matrix shape is `[label_count, embedding_dim]`.
    LabelRows,
    /// Matrix shape is `[embedding_dim, label_count]`.
    LabelColumns,
}

impl BioClipEmbeddingLayout {
    fn from_shape(label_count: usize, embeddings: &MmapNpyMatrix) -> Option<Self> {
        if embeddings.rows() == label_count {
            Some(Self::LabelRows)
        } else if embeddings.cols() == label_count {
            Some(Self::LabelColumns)
        } else {
            None
        }
    }

    fn embedding_dim(self, embeddings: &MmapNpyMatrix) -> usize {
        match self {
            Self::LabelRows => embeddings.cols(),
            Self::LabelColumns => embeddings.rows(),
        }
    }
}

struct MmapNpyMatrix {
    mmap: memmap2::Mmap,
    data_offset: usize,
    rows: usize,
    cols: usize,
    element_type: NpyElementType,
}

impl MmapNpyMatrix {
    fn open(path: &Path) -> ServiceResult<Self> {
        let file = File::open(path).map_err(|err| {
            ServiceError::InvalidArgument(format!(
                "failed to open BioCLIP embeddings at {}: {err}",
                path.display()
            ))
        })?;
        // The mapping is read-only and the file handle is not mutated while the
        // task is alive; this is the intended memmap2 usage for immutable model
        // assets.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|err| {
            ServiceError::InvalidArgument(format!(
                "failed to mmap BioCLIP embeddings at {}: {err}",
                path.display()
            ))
        })?;
        let header = parse_npy_header(&mmap, path)?;
        let expected_bytes = header
            .rows
            .checked_mul(header.cols)
            .and_then(|count| count.checked_mul(header.element_type.size_bytes()))
            .and_then(|bytes| bytes.checked_add(header.data_offset))
            .ok_or_else(|| {
                ServiceError::InvalidArgument(format!(
                    "BioCLIP embeddings at {} are too large to address",
                    path.display()
                ))
            })?;
        if mmap.len() < expected_bytes {
            return Err(ServiceError::InvalidArgument(format!(
                "BioCLIP embeddings at {} are truncated: expected at least {expected_bytes} bytes, got {}",
                path.display(),
                mmap.len()
            )));
        }

        Ok(Self {
            mmap,
            data_offset: header.data_offset,
            rows: header.rows,
            cols: header.cols,
            element_type: header.element_type,
        })
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn value(&self, row: usize, col: usize) -> f32 {
        let offset = self.data_offset + (row * self.cols + col) * self.element_type.size_bytes();
        match self.element_type {
            NpyElementType::Float32 => f32::from_le_bytes([
                self.mmap[offset],
                self.mmap[offset + 1],
                self.mmap[offset + 2],
                self.mmap[offset + 3],
            ]),
            NpyElementType::Float16 => {
                f16::from_le_bytes([self.mmap[offset], self.mmap[offset + 1]]).to_f32()
            }
        }
    }

    fn row_bytes(&self, row: usize) -> &[u8] {
        let row_size = self.cols * self.element_type.size_bytes();
        let offset = self.data_offset + row * row_size;
        &self.mmap[offset..offset + row_size]
    }

    fn row_cosine_similarity(
        &self,
        row: usize,
        query: &[f32],
        query_norm: f32,
    ) -> ServiceResult<f32> {
        let mut dot = 0.0;
        let mut row_norm_sq = 0.0;
        for (col, query_value) in query.iter().enumerate() {
            let value = self.value(row, col);
            dot += value * query_value;
            row_norm_sq += value * value;
        }
        if row_norm_sq == 0.0 {
            return Ok(f32::NAN);
        }
        Ok(dot / (row_norm_sq.sqrt() * query_norm))
    }

    fn column_cosine_similarity(
        &self,
        col: usize,
        query: &[f32],
        query_norm: f32,
    ) -> ServiceResult<f32> {
        let mut dot = 0.0;
        let mut col_norm_sq = 0.0;
        for (row, query_value) in query.iter().enumerate() {
            let value = self.value(row, col);
            dot += value * query_value;
            col_norm_sq += value * value;
        }
        if col_norm_sq == 0.0 {
            return Ok(f32::NAN);
        }
        Ok(dot / (col_norm_sq.sqrt() * query_norm))
    }
}

struct NpyHeader {
    data_offset: usize,
    rows: usize,
    cols: usize,
    element_type: NpyElementType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NpyElementType {
    Float16,
    Float32,
}

impl NpyElementType {
    fn size_bytes(self) -> usize {
        match self {
            Self::Float16 => std::mem::size_of::<f16>(),
            Self::Float32 => std::mem::size_of::<f32>(),
        }
    }
}

fn parse_npy_header(bytes: &[u8], path: &Path) -> ServiceResult<NpyHeader> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} are not a .npy file",
            path.display()
        )));
    }

    let major = bytes[6];
    let header_len_size = match major {
        1 => 2,
        2 | 3 => 4,
        other => {
            return Err(ServiceError::InvalidArgument(format!(
                "unsupported .npy major version {other} in {}",
                path.display()
            )));
        }
    };
    let header_len_offset = 8;
    let header_offset = header_len_offset + header_len_size;
    if bytes.len() < header_offset {
        return Err(ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} have an incomplete .npy header",
            path.display()
        )));
    }
    let header_len = match header_len_size {
        2 => u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
        4 => u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        _ => unreachable!(),
    };
    let data_offset = header_offset + header_len;
    if bytes.len() < data_offset {
        return Err(ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} have a truncated .npy header",
            path.display()
        )));
    }
    let header = std::str::from_utf8(&bytes[header_offset..data_offset]).map_err(|err| {
        ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} have a non-UTF8 .npy header: {err}",
            path.display()
        ))
    })?;

    let element_type = if header.contains("'descr': '<f4'")
        || header.contains("\"descr\": \"<f4\"")
        || header.contains("'descr': '|f4'")
        || header.contains("\"descr\": \"|f4\"")
    {
        NpyElementType::Float32
    } else if header.contains("'descr': '<f2'")
        || header.contains("\"descr\": \"<f2\"")
        || header.contains("'descr': '|f2'")
        || header.contains("\"descr\": \"|f2\"")
    {
        NpyElementType::Float16
    } else {
        return Err(ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} must be little-endian float16 or float32 .npy",
            path.display()
        )));
    };
    if !(header.contains("'fortran_order': False")
        || header.contains("\"fortran_order\": False")
        || header.contains("\"fortran_order\": false"))
    {
        return Err(ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} must be C-contiguous, not Fortran order",
            path.display()
        )));
    }

    let shape_start = header.find('(').ok_or_else(|| {
        ServiceError::InvalidArgument(format!("missing .npy shape in {}", path.display()))
    })?;
    let shape_end = header[shape_start..]
        .find(')')
        .map(|index| shape_start + index)
        .ok_or_else(|| {
            ServiceError::InvalidArgument(format!("missing .npy shape in {}", path.display()))
        })?;
    let shape: Vec<usize> = header[shape_start + 1..shape_end]
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value.parse::<usize>().map_err(|err| {
                ServiceError::InvalidArgument(format!(
                    "invalid .npy shape dimension `{value}` in {}: {err}",
                    path.display()
                ))
            })
        })
        .collect::<ServiceResult<_>>()?;
    if shape.len() != 2 {
        return Err(ServiceError::InvalidArgument(format!(
            "BioCLIP embeddings at {} must be a 2D matrix, got shape {:?}",
            path.display(),
            shape
        )));
    }

    Ok(NpyHeader {
        data_offset,
        rows: shape[0],
        cols: shape[1],
        element_type,
    })
}

const BIOCLIP_TAXONOMY_PLACEHOLDER: &str = "*";

#[derive(serde::Deserialize)]
struct BioClipLabel {
    #[serde(default)]
    kingdom: String,
    #[serde(default)]
    phylum: String,
    #[serde(default)]
    class: String,
    #[serde(default)]
    order: String,
    #[serde(default)]
    family: String,
    #[serde(default)]
    genus: String,
    #[serde(default)]
    species: String,
    #[serde(default)]
    common_name: String,
}

fn clean_rank(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        BIOCLIP_TAXONOMY_PLACEHOLDER.to_owned()
    } else {
        trimmed.to_owned()
    }
}

fn load_bioclip_labels(path: &Path) -> ServiceResult<Vec<String>> {
    let contents = std::fs::read_to_string(path).map_err(|err| {
        ServiceError::InvalidArgument(format!(
            "failed to read BioCLIP labels at {}: {err}",
            path.display()
        ))
    })?;
    let items: Vec<BioClipLabel> = serde_json::from_str(&contents).map_err(|err| {
        ServiceError::InvalidArgument(format!(
            "failed to parse BioCLIP labels at {}: {err}",
            path.display()
        ))
    })?;

    let labels = items
        .into_iter()
        .map(|item| {
            let clean_k = clean_rank(&item.kingdom);
            let clean_p = clean_rank(&item.phylum);
            let clean_c = clean_rank(&item.class);
            let clean_o = clean_rank(&item.order);
            let clean_f = clean_rank(&item.family);
            let clean_g = clean_rank(&item.genus);
            let clean_s = clean_rank(&item.species);
            let clean_cn = clean_rank(&item.common_name);
            format!(
                "{clean_k}/{clean_p}/{clean_c}/{clean_o}/{clean_f}/{clean_g}/{clean_s}/{clean_cn}"
            )
        })
        .collect();

    Ok(labels)
}

fn push_top_k(best: &mut Vec<(usize, f32)>, limit: usize, label_index: usize, score: f32) {
    if limit == 0 || !score.is_finite() {
        return;
    }
    best.push((label_index, score));
    best.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    best.truncate(limit);
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
            return Ok(parsed.min(max_len));
        }
    }
    Ok(BIOCLIP_DEFAULT_TOP_K.min(max_len))
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

impl ClipImageEmbedTask {
    pub fn new(
        task_name: impl Into<String>,
        pipeline: MLPipeline,
        context: Arc<MLContext>,
        model_id: impl Into<String>,
        input_names: Vec<String>,
        input_dtype: MLPacketDataType,
        output_name: impl Into<String>,
        preprocess: ClipImagePreprocessConfig,
    ) -> ServiceResult<Self> {
        let input_name = input_names.into_iter().next().ok_or_else(|| {
            ServiceError::InvalidArgument("CLIP image task requires one ONNX input name".to_owned())
        })?;
        let output_name = output_name.into();
        Ok(Self {
            spec: TaskSpec::new(task_name, "CLIP vision encoder → L2-normalized embedding")
                .with_input_mimes(image_input_mimes_with_tensor())
                .with_output_mime("application/json;schema=embedding_v1"),
            pipeline: Arc::new(pipeline),
            context,
            model_id: model_id.into(),
            input_name,
            input_dtype,
            output_name,
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
                    "CLIP image ONNX input `{}` has unsupported dtype {:?}",
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

    fn tensor_input_descriptor(&self, request: &TaskRequest) -> ServiceResult<TensorDescriptor> {
        let descriptor = validate_tensor_request(
            request,
            TensorValidationOptions {
                dtype: ml_dtype_to_tensor_dtype(self.input_dtype)?,
                layout: IMAGE_TENSOR_LAYOUT,
                preprocess_id: CLIP_IMAGE_PREPROCESS_ID,
            },
        )?;
        let expected_shape = self.preprocess.output_shape();
        if descriptor.shape != expected_shape {
            return Err(ServiceError::InvalidArgument(format!(
                "CLIP image tensor shape must be {:?}, got {:?}",
                expected_shape, descriptor.shape
            )));
        }
        Ok(descriptor)
    }

    fn tensor_request_to_packet(&self, request: &TaskRequest) -> ServiceResult<MLPacket> {
        let descriptor = self.tensor_input_descriptor(request)?;
        let tensor = tensor_payload_to_host_tensor(&request.payload, self.input_dtype)?;
        self.context
            .packet_from_host_tensor(
                MLPacketDescriptor::new(self.input_dtype, descriptor.shape),
                tensor,
            )
            .map_err(ServiceError::Internal)
    }

    fn tensor_request_to_packets(
        &self,
        request: &TaskRequest,
    ) -> ServiceResult<HashMap<String, MLPacket>> {
        Ok(HashMap::from([(
            self.input_name.clone(),
            self.tensor_request_to_packet(request)?,
        )]))
    }

    fn batched_tensor_packets(
        &self,
        requests: &[TaskRequest],
    ) -> ServiceResult<HashMap<String, MLPacket>> {
        let expected_shape = self.preprocess.output_shape();
        let mut batched_shape = expected_shape.clone();
        batched_shape[0] = requests.len();

        let tensor = match self.input_dtype {
            MLPacketDataType::Float32 => {
                let mut values = Vec::new();
                for request in requests {
                    self.tensor_input_descriptor(request)?;
                    values.extend(bytes_to_f32_le(&request.payload)?);
                }
                HostTensor::Float32(values)
            }
            MLPacketDataType::Float16 => {
                let mut values = Vec::new();
                for request in requests {
                    self.tensor_input_descriptor(request)?;
                    values.extend(bytes_to_f16_le(&request.payload)?);
                }
                HostTensor::Float16(values)
            }
            other => {
                return Err(ServiceError::Internal(format!(
                    "CLIP image ONNX input `{}` has unsupported dtype {:?}",
                    self.input_name, other
                )));
            }
        };

        let packet = self
            .context
            .packet_from_host_tensor(
                MLPacketDescriptor::new(self.input_dtype, batched_shape),
                tensor,
            )
            .map_err(ServiceError::Internal)?;

        Ok(HashMap::from([(self.input_name.clone(), packet)]))
    }

    async fn run_pipeline(
        &self,
        input_packets: HashMap<String, MLPacket>,
    ) -> ServiceResult<MLPacket> {
        let mut outputs = self
            .pipeline
            .run(input_packets)
            .await
            .map_err(ServiceError::Internal)?;

        outputs.remove(&self.output_name).ok_or_else(|| {
            ServiceError::Internal(format!(
                "pipeline output missing key `{}`",
                self.output_name
            ))
        })
    }

    async fn embedding_result_from_packet(&self, packet: MLPacket) -> ServiceResult<TaskResult> {
        let tensor = packet
            .to_host_tensor()
            .await
            .map_err(ServiceError::Internal)?;

        let embedding = match tensor {
            HostTensor::Float32(values) => EmbeddingV1::new(values, &self.model_id),
            other => {
                return Err(ServiceError::Internal(format!(
                    "unexpected tensor type from CLIP pipeline: {other:?}"
                )));
            }
        };

        embedding_json_result(embedding)
    }
}

#[async_trait]
impl TaskHandler for ClipImageEmbedTask {
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
            CLIP_IMAGE_PREPROCESS_ID
        ))))
    }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
        if normalized_meta(request.meta.get(META_INPUT_KIND)).as_deref() == Some(INPUT_KIND_TENSOR)
        {
            let packet = self
                .run_pipeline(self.tensor_request_to_packets(&request)?)
                .await?;
            return self.embedding_result_from_packet(packet).await;
        }

        if !is_supported_image_input_mime(&request.payload_mime) {
            return Err(ServiceError::InvalidArgument(format!(
                "unsupported CLIP image input MIME `{}`; supported MIME types: {}",
                request.payload_mime,
                SUPPORTED_IMAGE_INPUT_MIMES.join(", ")
            )));
        }

        let input_packets = self.preprocess_image(&request.payload)?;
        let packet = self.run_pipeline(input_packets).await?;
        self.embedding_result_from_packet(packet).await
    }

    async fn handle_batch(&self, requests: Vec<TaskRequest>) -> ServiceResult<Vec<TaskResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let batch_len = requests.len();
        let packet = self
            .run_pipeline(self.batched_tensor_packets(&requests)?)
            .await?;
        let (values, shape) = float32_values_and_shape(packet).await?;
        if shape.first().copied() != Some(batch_len) {
            return Err(ServiceError::Internal(format!(
                "CLIP batch output shape {:?} does not match batch size {batch_len}",
                shape
            )));
        }
        let row_width = values.len().checked_div(batch_len).ok_or_else(|| {
            ServiceError::Internal("CLIP batch output has invalid batch size".to_owned())
        })?;
        if row_width * batch_len != values.len() {
            return Err(ServiceError::Internal(format!(
                "CLIP batch output element count {} is not divisible by batch size {batch_len}",
                values.len()
            )));
        }

        values
            .chunks(row_width)
            .map(|row| embedding_json_result(EmbeddingV1::new(row.to_vec(), &self.model_id)))
            .collect()
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

fn ml_dtype_to_tensor_dtype(dtype: MLPacketDataType) -> ServiceResult<&'static str> {
    match dtype {
        MLPacketDataType::Float32 => Ok("fp32"),
        MLPacketDataType::Float16 => Ok("fp16"),
        other => Err(ServiceError::Internal(format!(
            "unsupported image tensor dtype {other:?}"
        ))),
    }
}

fn tensor_payload_to_host_tensor(
    payload: &[u8],
    dtype: MLPacketDataType,
) -> ServiceResult<HostTensor> {
    match dtype {
        MLPacketDataType::Float32 => Ok(HostTensor::Float32(bytes_to_f32_le(payload)?)),
        MLPacketDataType::Float16 => Ok(HostTensor::Float16(bytes_to_f16_le(payload)?)),
        other => Err(ServiceError::Internal(format!(
            "unsupported image tensor dtype {other:?}"
        ))),
    }
}

async fn float32_values_and_shape(packet: MLPacket) -> ServiceResult<(Vec<f32>, Vec<usize>)> {
    let shape = packet.descriptor.shape.clone();
    let tensor = packet
        .to_host_tensor()
        .await
        .map_err(ServiceError::Internal)?;
    match tensor {
        HostTensor::Float32(values) => Ok((values, shape)),
        other => Err(ServiceError::Internal(format!(
            "unexpected tensor type from CLIP pipeline: {other:?}"
        ))),
    }
}

fn embedding_json_result(embedding: EmbeddingV1) -> ServiceResult<TaskResult> {
    let json_bytes = embedding
        .to_json_bytes()
        .map_err(|err| ServiceError::Internal(err.to_string()))?;

    Ok(
        TaskResult::new(json_bytes, "application/json;schema=embedding_v1")
            .with_result_schema("embedding_v1"),
    )
}

fn labels_json_result(labels: LabelsV1) -> ServiceResult<TaskResult> {
    let json_bytes = labels
        .to_json_bytes()
        .map_err(|err| ServiceError::Internal(err.to_string()))?;

    Ok(
        TaskResult::new(json_bytes, lumen_schema::mime::LABELS_V1_JSON)
            .with_result_schema(lumen_schema::mime::LABELS_V1_SCHEMA),
    )
}

impl ClipTextEmbedTask {
    pub fn new(
        task_name: impl Into<String>,
        pipeline: MLPipeline,
        context: Arc<MLContext>,
        model_id: impl Into<String>,
        input_names: Vec<String>,
        output_name: impl Into<String>,
        tokenizer: tokenizers::Tokenizer,
    ) -> Self {
        let output_name = output_name.into();
        Self {
            spec: TaskSpec::new(task_name, "CLIP text encoder → L2-normalized embedding")
                .with_input_mimes(["text/plain"])
                .with_output_mime("application/json;schema=embedding_v1"),
            pipeline: Arc::new(pipeline),
            context,
            model_id: model_id.into(),
            input_names,
            output_name,
            tokenizer,
        }
    }

    fn preprocess_text(&self, text: &str) -> ServiceResult<HashMap<String, MLPacket>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| ServiceError::Internal(format!("tokenization failed: {e}")))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let seq_len = ids.len();

        let mut packets = HashMap::new();

        // Primary input (e.g. "input_ids")
        let primary_name = &self.input_names[0];
        let input_ids_packet = self
            .context
            .packet_from_host_tensor(
                MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1, seq_len]),
                HostTensor::Int64(ids),
            )
            .map_err(ServiceError::Internal)?;
        packets.insert(primary_name.clone(), input_ids_packet);

        // Optional attention mask input
        if self.input_names.len() > 1 {
            let attention_mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&mask| mask as i64)
                .collect();
            let mask_name = &self.input_names[1];
            let attention_mask_packet = self
                .context
                .packet_from_host_tensor(
                    MLPacketDescriptor::new(MLPacketDataType::Int64, vec![1, seq_len]),
                    HostTensor::Int64(attention_mask),
                )
                .map_err(ServiceError::Internal)?;
            packets.insert(mask_name.clone(), attention_mask_packet);
        }

        Ok(packets)
    }
}

#[async_trait]
impl TaskHandler for ClipTextEmbedTask {
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

        let tensor = embedding_packet
            .to_host_tensor()
            .await
            .map_err(ServiceError::Internal)?;

        let embedding = match tensor {
            HostTensor::Float32(values) => EmbeddingV1::new(values, &self.model_id),
            other => {
                return Err(ServiceError::Internal(format!(
                    "unexpected tensor type from CLIP pipeline: {other:?}"
                )));
            }
        };

        let json_bytes = embedding
            .to_json_bytes()
            .map_err(|e| ServiceError::Internal(e.to_string()))?;

        Ok(
            TaskResult::new(json_bytes, "application/json;schema=embedding_v1")
                .with_result_schema("embedding_v1"),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::{fs, io::Write};

    use bytes::Bytes;
    use lumnn::core::{
        context::{MLContext, MLContextOptions},
        pipeline::MLPipeline,
    };
    use serde_json::json;

    use crate::service::{
        DEFAULT_TENSOR_MIME, INPUT_KIND_TENSOR, META_INPUT_KIND, META_PREPROCESS_ID,
        META_PREPROCESS_SKIP, META_TENSOR_BYTE_ORDER, META_TENSOR_DTYPE, META_TENSOR_FORMAT,
        META_TENSOR_LAYOUT, META_TENSOR_SHAPE, TENSOR_BYTE_ORDER_LITTLE, TENSOR_FORMAT_CONTIGUOUS,
        TaskHandler, TaskRequest, f32_to_le_bytes,
    };

    use super::{
        BIOCLIP_DEFAULT_TOP_K, BioClipEmbeddingLayout, CLIP_IMAGE_PREPROCESS_ID,
        ClipImageEmbedTask, ClipImagePreprocessConfig, MmapNpyMatrix,
        image_input_mimes_with_tensor, is_supported_image_input_mime, load_bioclip_labels,
        top_k_from_request,
    };

    #[test]
    fn image_preprocess_config_parses_model_info_metadata() {
        let config = ClipImagePreprocessConfig::from_json_str(
            &json!({
                "resize_shortest_edge": 224,
                "crop_size": { "width": 224, "height": 224 },
                "do_resize": true,
                "do_center_crop": true,
                "do_rescale": true,
                "do_normalize": true,
                "rescale_factor": 0.00392156862745098,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
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
        let err = ClipImagePreprocessConfig::from_json_str(
            &json!({
                "resize_shortest_edge": 224,
                "crop_size": { "width": 224, "height": 224 },
                "do_resize": true,
                "do_center_crop": true,
                "do_rescale": true,
                "do_normalize": true,
                "rescale_factor": 0.00392156862745098,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
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

    #[test]
    fn image_task_advertises_tensor_input_mime() {
        assert!(image_input_mimes_with_tensor().contains(&DEFAULT_TENSOR_MIME.to_owned()));
    }

    #[test]
    fn image_tensor_batch_key_validates_shape_and_preprocess_id() {
        let task = test_image_task();
        let request = tensor_request(vec![1, 3, 224, 224], CLIP_IMAGE_PREPROCESS_ID);

        assert!(task.batch_key(&request).unwrap().is_some());

        let wrong_shape = tensor_request(vec![1, 3, 112, 112], CLIP_IMAGE_PREPROCESS_ID);
        assert!(task.batch_key(&wrong_shape).is_err());

        let wrong_preprocess = tensor_request(vec![1, 3, 224, 224], "wrong_preprocess");
        assert!(task.batch_key(&wrong_preprocess).is_err());
    }

    #[test]
    fn bioclip_top_k_meta_accepts_topk_alias() {
        let request = TaskRequest::new(Bytes::new(), "image/jpeg").with_meta("TopK", "3");

        assert_eq!(top_k_from_request(&request, 10).unwrap(), 3);
        assert_eq!(
            top_k_from_request(&TaskRequest::new(Bytes::new(), "image/jpeg"), 10).unwrap(),
            BIOCLIP_DEFAULT_TOP_K
        );
    }

    #[test]
    fn bioclip_mmap_npy_reads_float32_matrix_header() {
        let path = temp_path("bioclip_embeddings.npy");
        write_npy_f32_matrix(&path, 2, 3, &[1.0, 0.0, 0.0, 0.0, 3.0, 4.0]);

        let matrix = MmapNpyMatrix::open(&path).unwrap();

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.value(1, 2), 4.0);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn bioclip_mmap_npy_accepts_float16_matrix() {
        let path = temp_path("bioclip_embeddings_f16.npy");
        write_npy_f16_matrix(&path, 1, 2, &[1.5, -2.0]);

        let matrix = MmapNpyMatrix::open(&path).unwrap();

        assert_eq!(matrix.rows(), 1);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.value(0, 0), 1.5);
        assert_eq!(matrix.value(0, 1), -2.0);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn bioclip_embedding_layout_accepts_label_columns() {
        let path = temp_path("bioclip_embeddings_columns.npy");
        write_npy_f16_matrix(&path, 2, 3, &[1.0, 0.0, 0.5, 0.0, 1.0, 0.5]);
        let matrix = MmapNpyMatrix::open(&path).unwrap();

        let layout = BioClipEmbeddingLayout::from_shape(3, &matrix).unwrap();

        assert_eq!(layout, BioClipEmbeddingLayout::LabelColumns);
        assert_eq!(layout.embedding_dim(&matrix), 2);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn bioclip_labels_accept_strings_and_objects() {
        let path = temp_path("bioclip_labels.json");
        fs::write(
            &path,
            r#"[
              {
                "kingdom": "Plantae",
                "phylum": "Tracheophyta",
                "class": "Magnoliopsida",
                "order": "Sapindales",
                "family": "Sapindaceae",
                "genus": "Acer",
                "species": "rubrum",
                "common_name": "red maple"
              },
              {
                "kingdom": "Animalia",
                "genus": "Acanthodes",
                "species": "pristis"
              }
            ]"#,
        )
        .unwrap();

        let labels = load_bioclip_labels(&path).unwrap();

        assert_eq!(
            labels,
            vec![
                "Plantae/Tracheophyta/Magnoliopsida/Sapindales/Sapindaceae/Acer/rubrum/red maple"
                    .to_owned(),
                "Animalia/*/*/*/*/Acanthodes/pristis/*".to_owned()
            ]
        );

        let _ = fs::remove_file(path);
    }

    fn test_image_task() -> ClipImageEmbedTask {
        let context = MLContext::new(MLContextOptions::default()).unwrap();
        let pipeline = MLPipeline::new("test", Arc::clone(&context), Vec::new());
        ClipImageEmbedTask::new(
            "semantic_image_embed",
            pipeline,
            context,
            "clip-test",
            vec!["pixel_values".to_owned()],
            lumnn::core::packet::MLPacketDataType::Float32,
            "embedding",
            ClipImagePreprocessConfig::from_json_str(
                &json!({
                    "resize_shortest_edge": 224,
                    "crop_size": { "width": 224, "height": 224 },
                    "do_resize": true,
                    "do_center_crop": true,
                    "do_rescale": true,
                    "do_normalize": true,
                    "rescale_factor": 0.00392156862745098,
                    "image_mean": [0.48145466, 0.4578275, 0.40821073],
                    "image_std": [0.26862954, 0.26130258, 0.27577711],
                    "resample": "bicubic",
                    "color_space": "rgb",
                    "layout": "nchw"
                })
                .to_string(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn tensor_request(shape: Vec<usize>, preprocess_id: &str) -> TaskRequest {
        let element_count = shape.iter().product::<usize>();
        TaskRequest::new(
            f32_to_le_bytes(&vec![0.0; element_count]),
            DEFAULT_TENSOR_MIME,
        )
        .with_meta(META_INPUT_KIND, INPUT_KIND_TENSOR)
        .with_meta(META_TENSOR_DTYPE, "fp32")
        .with_meta(META_TENSOR_SHAPE, serde_json::to_string(&shape).unwrap())
        .with_meta(META_TENSOR_LAYOUT, "NCHW")
        .with_meta(META_TENSOR_FORMAT, TENSOR_FORMAT_CONTIGUOUS)
        .with_meta(META_TENSOR_BYTE_ORDER, TENSOR_BYTE_ORDER_LITTLE)
        .with_meta(META_PREPROCESS_ID, preprocess_id)
        .with_meta(META_PREPROCESS_SKIP, "true")
    }

    fn temp_path(filename: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "lumen_hub_clip_{}_{}",
            std::process::id(),
            filename
        ))
    }

    fn write_npy_f32_matrix(path: &std::path::Path, rows: usize, cols: usize, values: &[f32]) {
        assert_eq!(values.len(), rows * cols);
        let mut header = npy_header(rows, cols, "<f4");
        let preamble_len = 10;
        let padding = (16 - ((preamble_len + header.len() + 1) % 16)) % 16;
        header.push_str(&" ".repeat(padding));
        header.push('\n');

        let mut file = fs::File::create(path).unwrap();
        file.write_all(b"\x93NUMPY").unwrap();
        file.write_all(&[1, 0]).unwrap();
        file.write_all(&(header.len() as u16).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        for value in values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }

    fn write_npy_f16_matrix(path: &std::path::Path, rows: usize, cols: usize, values: &[f32]) {
        assert_eq!(values.len(), rows * cols);
        let mut header = npy_header(rows, cols, "<f2");
        let preamble_len = 10;
        let padding = (16 - ((preamble_len + header.len() + 1) % 16)) % 16;
        header.push_str(&" ".repeat(padding));
        header.push('\n');

        let mut file = fs::File::create(path).unwrap();
        file.write_all(b"\x93NUMPY").unwrap();
        file.write_all(&[1, 0]).unwrap();
        file.write_all(&(header.len() as u16).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        for value in values {
            file.write_all(&half::f16::from_f32(*value).to_le_bytes())
                .unwrap();
        }
    }

    fn npy_header(rows: usize, cols: usize, descr: &str) -> String {
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({rows}, {cols}), }}")
    }
}
