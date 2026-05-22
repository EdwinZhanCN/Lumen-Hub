use std::{
    env, fs, io,
    path::{Path, PathBuf},
};

use lumen_schema::{EmbeddingV1, FaceV1, LabelsV1, OCRV1};
use thiserror::Error;

use crate::service::{ServiceError, ServiceHub, TaskRequest, TaskResult, TaskSpec};

const SEMANTIC_IMAGE_CATEGORY: &str = "semantic";
const BIO_CATEGORY: &str = "bio";
const FACE_CATEGORY: &str = "face";
const OCR_CATEGORY: &str = "ocr";

const TEXT_EMBED_SAMPLE: &str = "a photo of a city bus";

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

        for task in &capability.tasks {
            let task_name = task.name.as_str();
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
        let by_name = TaskSpec::new("ocr", "");
        let other = TaskSpec::new("semantic_image_embed", "");

        assert!(is_ocr_task(&by_schema));
        assert!(is_ocr_task(&by_name));
        assert!(!is_ocr_task(&other));
    }
}
