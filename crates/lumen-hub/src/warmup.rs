use std::{
    env, fs, io,
    path::{Path, PathBuf},
};

#[cfg(feature = "fastvlm")]
use image::{ImageBuffer, Rgb, imageops::FilterType};
#[cfg(feature = "fastvlm")]
use std::collections::HashSet;

#[cfg(feature = "fastvlm")]
use lumen_schema::TextGenerationV1;
use lumen_schema::{EmbeddingV1, FaceV1, LabelsV1, OCRV1};
use thiserror::Error;

#[cfg(feature = "fastvlm")]
use crate::models::fastvlm::{
    metadata::METADATA as FASTVLM_METADATA,
    task::{FASTVLM_PREPROCESS_ID, META_MAX_TOKENS, META_PROMPT},
};
#[cfg(feature = "fastvlm")]
use crate::service::{
    DEFAULT_TENSOR_MIME, INPUT_KIND_TENSOR, META_INPUT_KIND, META_OUTPUT_TENSOR_SHAPE,
    META_PREPROCESS_ID, META_PREPROCESS_SKIP, META_TENSOR_BYTE_ORDER, META_TENSOR_DTYPE,
    META_TENSOR_FORMAT, META_TENSOR_LAYOUT, META_TENSOR_SHAPE, TENSOR_BYTE_ORDER_LITTLE,
    TENSOR_FORMAT_CONTIGUOUS, f32_to_le_bytes,
};
use crate::service::{ServiceError, ServiceHub, TaskRequest, TaskResult, TaskSpec};

const SEMANTIC_IMAGE_CATEGORY: &str = "semantic";
const BIO_CATEGORY: &str = "bio";
const FACE_CATEGORY: &str = "face";
const OCR_CATEGORY: &str = "ocr";
#[cfg(feature = "fastvlm")]
const CAPTION_CATEGORY: &str = "caption";

const TEXT_EMBED_SAMPLE: &str = "a photo of a city bus";
#[cfg(feature = "fastvlm")]
const VLM_PROMPT: &str = "Describe the image briefly.";
#[cfg(feature = "fastvlm")]
const VLM_MAX_TOKENS: &str = "16";

pub fn default_warmup_dir() -> PathBuf {
    if let Ok(exe) = env::current_exe()
        && let Some(app_home) = exe.parent().and_then(Path::parent)
    {
        let dist_dir = app_home.join("warmup");
        if dist_dir.exists() {
            return dist_dir;
        }
    }

    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("warmup")
}

pub async fn run_startup_warmup(hub: &ServiceHub, sample_root: &Path) -> Result<(), WarmupError> {
    println!("warmup: starting samples={}", sample_root.display());

    let mut warmed = 0usize;
    for capability in hub.capabilities() {
        let service_name = capability.service_name.as_str();

        #[cfg(feature = "fastvlm")]
        {
            let task_names = capability
                .tasks
                .iter()
                .map(|task| task.name.as_str())
                .collect::<HashSet<_>>();
            if let Some((embeds_task, decode_task)) = vlm_tasks(&task_names) {
                warmup_vlm(hub, sample_root, service_name, embeds_task, decode_task).await?;
                warmed += 1;
            }
        }

        for task in &capability.tasks {
            let task_name = task.name.as_str();
            if task_name == "vlm_embeds"
                || task_name == "vlm_decode"
                || task_name.ends_with("_vlm_embeds")
                || task_name.ends_with("_vlm_decode")
            {
                continue;
            }

            if task_name == "semantic_image_embed" || task_name.ends_with("_semantic_image_embed") {
                warmup_embedding_image(
                    hub,
                    sample_root,
                    service_name,
                    task_name,
                    SEMANTIC_IMAGE_CATEGORY,
                    "semantic image embedding",
                )
                .await?;
                warmed += 1;
            } else if task_name == "semantic_text_embed"
                || task_name.ends_with("_semantic_text_embed")
            {
                warmup_embedding_text(hub, service_name, task_name).await?;
                warmed += 1;
            } else if task_name == "bioclip_classify" || task_name.ends_with("_bioclip_classify") {
                warmup_labels(hub, sample_root, service_name, task_name).await?;
                warmed += 1;
            } else if task_name == "face_recognition" || task_name.ends_with("_face_recognition") {
                warmup_face(hub, sample_root, service_name, task_name).await?;
                warmed += 1;
            } else if is_ocr_task(task) {
                warmup_ocr(hub, sample_root, service_name, task_name).await?;
                warmed += 1;
            }
        }
    }

    if warmed == 0 {
        return Err(WarmupError::NoWarmupTasks);
    }

    println!("warmup: completed status=ok tasks={warmed}");
    Ok(())
}

async fn warmup_embedding_image(
    hub: &ServiceHub,
    sample_root: &Path,
    service_name: &str,
    task_name: &str,
    category: &str,
    label: &str,
) -> Result<(), WarmupError> {
    let request = image_request(sample_root, category)?;
    let result = handle(hub, service_name, task_name, request).await?;
    let embedding = parse_result::<EmbeddingV1>(&result, service_name, task_name, "embedding_v1")?;
    println!(
        "warmup: service={service_name} task={task_name} status=ok {label} embedding_shape=[{}]",
        embedding.dim
    );
    Ok(())
}

async fn warmup_embedding_text(
    hub: &ServiceHub,
    service_name: &str,
    task_name: &str,
) -> Result<(), WarmupError> {
    let result = handle(
        hub,
        service_name,
        task_name,
        TaskRequest::new(TEXT_EMBED_SAMPLE.as_bytes().to_vec(), "text/plain"),
    )
    .await?;
    let embedding = parse_result::<EmbeddingV1>(&result, service_name, task_name, "embedding_v1")?;
    println!(
        "warmup: service={service_name} task={task_name} status=ok text embedding_shape=[{}]",
        embedding.dim
    );
    Ok(())
}

async fn warmup_labels(
    hub: &ServiceHub,
    sample_root: &Path,
    service_name: &str,
    task_name: &str,
) -> Result<(), WarmupError> {
    let request = image_request(sample_root, BIO_CATEGORY)?.with_meta("TopK", "1");
    let result = handle(hub, service_name, task_name, request).await?;
    let labels = parse_result::<LabelsV1>(&result, service_name, task_name, "labels_v1")?;
    let label = labels
        .labels
        .first()
        .map(|label| format!("\"{}\" score={:.4}", label.label, label.score))
        .unwrap_or_else(|| "\"<none>\" score=nan".to_owned());
    println!("warmup: service={service_name} task={task_name} status=ok classify_label={label}");
    Ok(())
}

async fn warmup_face(
    hub: &ServiceHub,
    sample_root: &Path,
    service_name: &str,
    task_name: &str,
) -> Result<(), WarmupError> {
    let request = image_request(sample_root, FACE_CATEGORY)?;
    let result = handle(hub, service_name, task_name, request).await?;
    let face = parse_result::<FaceV1>(&result, service_name, task_name, "face_v1")?;
    println!(
        "warmup: service={service_name} task={task_name} status=ok face_count={}",
        face.count
    );
    Ok(())
}

async fn warmup_ocr(
    hub: &ServiceHub,
    sample_root: &Path,
    service_name: &str,
    task_name: &str,
) -> Result<(), WarmupError> {
    let request = image_request(sample_root, OCR_CATEGORY)?;
    let result = handle(hub, service_name, task_name, request).await?;
    let ocr = parse_result::<OCRV1>(&result, service_name, task_name, "ocr_v1")?;
    let text = ocr
        .items
        .iter()
        .map(|item| item.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "warmup: service={service_name} task={task_name} status=ok ocr_label=\"{}\" count={}",
        text, ocr.count
    );
    Ok(())
}

#[cfg(feature = "fastvlm")]
async fn warmup_vlm(
    hub: &ServiceHub,
    sample_root: &Path,
    service_name: &str,
    embeds_task: &str,
    decode_task: &str,
) -> Result<(), WarmupError> {
    let image_path = sample_file(sample_root, CAPTION_CATEGORY)?;
    let payload = fastvlm_image_tensor(&image_path)?;
    let embeds_request = TaskRequest::new(payload, DEFAULT_TENSOR_MIME)
        .with_meta(META_INPUT_KIND, INPUT_KIND_TENSOR)
        .with_meta(META_TENSOR_DTYPE, "fp32")
        .with_meta(META_TENSOR_SHAPE, "[1,3,448,448]")
        .with_meta(META_TENSOR_LAYOUT, "NCHW")
        .with_meta(META_TENSOR_FORMAT, TENSOR_FORMAT_CONTIGUOUS)
        .with_meta(META_TENSOR_BYTE_ORDER, TENSOR_BYTE_ORDER_LITTLE)
        .with_meta(META_PREPROCESS_ID, FASTVLM_PREPROCESS_ID)
        .with_meta(META_PREPROCESS_SKIP, "true")
        .with_meta(META_PROMPT, VLM_PROMPT);

    let embeds = handle(hub, service_name, embeds_task, embeds_request).await?;
    let shape = embeds
        .meta
        .get(META_OUTPUT_TENSOR_SHAPE)
        .cloned()
        .unwrap_or_else(|| "<unknown>".to_owned());

    let mut decode_request = TaskRequest::new(embeds.payload, embeds.payload_mime);
    decode_request.meta = embeds.meta;
    decode_request = decode_request.with_meta(META_MAX_TOKENS, VLM_MAX_TOKENS);
    let decoded = handle(hub, service_name, decode_task, decode_request).await?;
    let text = parse_result::<TextGenerationV1>(
        &decoded,
        service_name,
        decode_task,
        "text_generation_v1",
    )?;
    println!(
        "warmup: service={service_name} task={decode_task} status=ok caption=\"{}\" generated_tokens={} embedding_shape={shape}",
        text.text, text.generated_tokens
    );
    Ok(())
}

#[cfg(feature = "fastvlm")]
fn vlm_tasks<'a>(task_names: &HashSet<&'a str>) -> Option<(&'a str, &'a str)> {
    let embeds = task_names
        .iter()
        .copied()
        .find(|name| *name == "vlm_embeds" || name.ends_with("_vlm_embeds"))?;
    let decode = task_names
        .iter()
        .copied()
        .find(|name| *name == "vlm_decode" || name.ends_with("_vlm_decode"))?;
    Some((embeds, decode))
}

async fn handle(
    hub: &ServiceHub,
    service_name: &str,
    task_name: &str,
    request: TaskRequest,
) -> Result<TaskResult, WarmupError> {
    hub.handle(service_name, task_name, request)
        .await
        .map_err(|source| WarmupError::TaskFailed {
            service: service_name.to_owned(),
            task: task_name.to_owned(),
            source,
        })
}

fn parse_result<T>(
    result: &TaskResult,
    service_name: &str,
    task_name: &str,
    schema: &'static str,
) -> Result<T, WarmupError>
where
    T: serde::de::DeserializeOwned,
{
    serde_json::from_slice(&result.payload).map_err(|source| WarmupError::InvalidResult {
        service: service_name.to_owned(),
        task: task_name.to_owned(),
        schema,
        source,
    })
}

fn image_request(sample_root: &Path, category: &str) -> Result<TaskRequest, WarmupError> {
    let path = sample_file(sample_root, category)?;
    let payload = fs::read(&path).map_err(|source| WarmupError::ReadSample {
        path: path.clone(),
        source,
    })?;
    Ok(TaskRequest::new(payload, image_mime(&path)))
}

fn sample_file(sample_root: &Path, category: &str) -> Result<PathBuf, WarmupError> {
    let dir = sample_root.join(category);
    let entries = fs::read_dir(&dir).map_err(|source| WarmupError::ReadSampleDir {
        path: dir.clone(),
        source,
    })?;
    let mut files = entries
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.file_type().is_ok_and(|file_type| file_type.is_file())
                && entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| !name.starts_with('.'))
        })
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    files.sort();
    files.into_iter().next().ok_or(WarmupError::MissingSample {
        category: category.to_owned(),
        path: dir,
    })
}

fn image_mime(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        _ => "application/octet-stream",
    }
}

fn is_ocr_task(task: &TaskSpec) -> bool {
    task.metadata
        .get("output_schema")
        .is_some_and(|schema| schema == "ocr_v1")
        || task.name == "ocr"
        || task.name.ends_with("_ocr")
        || task.name.contains("ocr")
}

#[cfg(feature = "fastvlm")]
fn fastvlm_image_tensor(path: &Path) -> Result<bytes::Bytes, WarmupError> {
    let image = image::open(path).map_err(|source| WarmupError::DecodeImage {
        path: path.to_path_buf(),
        source,
    })?;
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let target_edge = FASTVLM_METADATA.vision_preprocess.resize_longest_edge;
    let scale = target_edge as f32 / width.max(height) as f32;
    let resized_width = ((width as f32 * scale).round() as u32).max(1);
    let resized_height = ((height as f32 * scale).round() as u32).max(1);
    let resized =
        image::imageops::resize(&rgb, resized_width, resized_height, fastvlm_resize_filter());

    let pad = FASTVLM_METADATA.vision_preprocess.pad_to;
    let pad_color = FASTVLM_METADATA.vision_preprocess.pad_color_rgb;
    let mut canvas = ImageBuffer::from_pixel(
        pad.width,
        pad.height,
        Rgb([pad_color[0], pad_color[1], pad_color[2]]),
    );
    let x = (pad.width.saturating_sub(resized_width)) / 2;
    let y = (pad.height.saturating_sub(resized_height)) / 2;
    image::imageops::replace(&mut canvas, &resized, i64::from(x), i64::from(y));

    let mut values = vec![0.0_f32; 3 * pad.width as usize * pad.height as usize];
    let plane = pad.width as usize * pad.height as usize;
    for (pixel_index, pixel) in canvas.pixels().enumerate() {
        values[pixel_index] = normalize_fastvlm_channel(pixel[0], 0);
        values[plane + pixel_index] = normalize_fastvlm_channel(pixel[1], 1);
        values[2 * plane + pixel_index] = normalize_fastvlm_channel(pixel[2], 2);
    }

    Ok(f32_to_le_bytes(&values))
}

#[cfg(feature = "fastvlm")]
fn normalize_fastvlm_channel(value: u8, channel: usize) -> f32 {
    let preprocess = &FASTVLM_METADATA.vision_preprocess;
    let mut value = f32::from(value);
    if preprocess.do_rescale {
        value *= preprocess.rescale_factor;
    }
    if preprocess.do_normalize {
        value = (value - preprocess.image_mean[channel]) / preprocess.image_std[channel];
    }
    value
}

#[cfg(feature = "fastvlm")]
fn fastvlm_resize_filter() -> FilterType {
    use crate::models::fastvlm::metadata::FastVlmResizeFilter;

    match FASTVLM_METADATA.vision_preprocess.resample {
        FastVlmResizeFilter::Nearest => FilterType::Nearest,
        FastVlmResizeFilter::Lanczos3 => FilterType::Lanczos3,
        FastVlmResizeFilter::Bilinear => FilterType::Triangle,
        FastVlmResizeFilter::Bicubic => FilterType::CatmullRom,
    }
}

#[derive(Debug, Error)]
pub enum WarmupError {
    #[error("no supported warmup task was registered")]
    NoWarmupTasks,

    #[error("warmup sample category `{category}` is empty or missing at `{}`", path.display())]
    MissingSample { category: String, path: PathBuf },

    #[error("failed to read warmup sample directory `{}`: {source}", path.display())]
    ReadSampleDir { path: PathBuf, source: io::Error },

    #[error("failed to read warmup sample `{}`: {source}", path.display())]
    ReadSample { path: PathBuf, source: io::Error },

    #[error("failed to decode warmup image `{}`: {source}", path.display())]
    DecodeImage {
        path: PathBuf,
        source: image::ImageError,
    },

    #[error("warmup task `{service}.{task}` failed: {source}")]
    TaskFailed {
        service: String,
        task: String,
        source: ServiceError,
    },

    #[error("warmup task `{service}.{task}` returned invalid {schema}: {source}")]
    InvalidResult {
        service: String,
        task: String,
        schema: &'static str,
        source: serde_json::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_mime_detects_common_extensions() {
        assert_eq!(image_mime(Path::new("a.jpg")), "image/jpeg");
        assert_eq!(image_mime(Path::new("a.jpeg")), "image/jpeg");
        assert_eq!(image_mime(Path::new("a.png")), "image/png");
    }

    #[test]
    fn ocr_task_detection_uses_schema_or_name() {
        let by_schema = TaskSpec::new("text", "").with_metadata("output_schema", "ocr_v1");
        let by_name = TaskSpec::new("default_ocr", "");
        let other = TaskSpec::new("semantic_image_embed", "");

        assert!(is_ocr_task(&by_schema));
        assert!(is_ocr_task(&by_name));
        assert!(!is_ocr_task(&other));
    }
}
