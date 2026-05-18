use std::{
    collections::BTreeSet,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use indicatif::{ProgressBar, ProgressStyle};
use lumen_schema::{LumenConfig, ModelInfo, ModelInfoValidationError, Region, Runtime};
use serde::Deserialize;
use thiserror::Error;
use tracing::info;

const HF_ORG: &str = "Lumilio-Photos";
const MODEL_INFO_FILE: &str = "model_info.json";
const DEFAULT_PRECISION: &str = "fp32";

pub fn ensure_models_for_config(
    config: &LumenConfig,
    cache_dir: impl AsRef<Path>,
) -> Result<(), ModelDownloadError> {
    let client = HfHubClient::new(config.metadata.region);
    ensure_models_with_client(config, cache_dir.as_ref(), &client)
}

fn ensure_models_with_client<C: ModelRepoClient>(
    config: &LumenConfig,
    cache_dir: &Path,
    client: &C,
) -> Result<(), ModelDownloadError> {
    info!(cache = %cache_dir.display(), endpoint = client.endpoint_label(), "model cache");
    let mut seen = BTreeSet::new();
    let mut requirements = Vec::new();

    for service_name in config.deployment_service_names() {
        let service = config.services.get(service_name).ok_or_else(|| {
            ModelDownloadError::InvalidConfig(format!(
                "deployment references unknown service `{service_name}`"
            ))
        })?;

        for model in service.models.values() {
            let precision = model
                .precision
                .clone()
                .unwrap_or_else(|| DEFAULT_PRECISION.to_owned());
            let key = format!(
                "{}\0{}\0{}\0{}",
                model.model,
                runtime_asset_key(model.runtime),
                precision,
                model.dataset.as_deref().unwrap_or_default()
            );

            if seen.insert(key) {
                requirements.push(ModelRequirement {
                    model: model.model.clone(),
                    runtime: model.runtime,
                    precision,
                    dataset: model.dataset.clone(),
                });
            }
        }
    }

    for requirement in requirements {
        let dataset = requirement
            .dataset
            .as_deref()
            .map(|dataset| format!(" dataset={dataset}"))
            .unwrap_or_default();
        info!(
            model = %requirement.model,
            runtime = runtime_asset_key(requirement.runtime),
            precision = %requirement.precision,
            dataset = %dataset.trim(),
            "ensuring model artifacts"
        );
        ensure_model(cache_dir, client, &requirement)?;
    }

    Ok(())
}

fn ensure_model<C: ModelRepoClient>(
    cache_dir: &Path,
    client: &C,
    requirement: &ModelRequirement,
) -> Result<(), ModelDownloadError> {
    let model_dir = cache_dir.join(&requirement.model);
    let model_info_path = model_dir.join(MODEL_INFO_FILE);
    ensure_remote_file(
        client,
        &requirement.model,
        MODEL_INFO_FILE,
        &model_info_path,
    )?;

    let model_info_json =
        fs::read_to_string(&model_info_path).map_err(|source| ModelDownloadError::ReadFile {
            path: model_info_path.clone(),
            source,
        })?;
    let model_info = ModelInfo::from_json_str(&model_info_json)?;
    let runtime_spec = validate_runtime_precision(&model_info, requirement)?;
    info!(model = %requirement.model, "validated model_info.json");

    let runtime_dir = runtime_asset_dir(requirement.runtime);
    let runtime_ext = runtime_asset_ext(requirement.runtime);
    let mut remote_paths = BTreeSet::from([MODEL_INFO_FILE.to_owned()]);

    for component in &runtime_spec.components {
        remote_paths.insert(format!(
            "{runtime_dir}/{component}.{}.{runtime_ext}",
            requirement.precision
        ));
    }

    for repo_file in client.list_repo_files(&requirement.model)? {
        if (is_default_root_file(&repo_file)
            && !is_root_dataset_file(&repo_file, &requirement.dataset))
            || is_dataset_file(&repo_file, &requirement.dataset)
        {
            remote_paths.insert(repo_file);
        }
    }

    for remote_path in remote_paths {
        let target = target_path_for_remote(&model_dir, &remote_path)?;
        ensure_remote_file(client, &requirement.model, &remote_path, &target)?;
    }

    Ok(())
}

fn validate_runtime_precision<'a>(
    model_info: &'a ModelInfo,
    requirement: &ModelRequirement,
) -> Result<&'a lumen_schema::RuntimeSpec, ModelDownloadError> {
    let runtime_key = runtime_asset_key(requirement.runtime);
    let runtime_spec = model_info
        .runtimes
        .as_map()
        .get(runtime_key)
        .ok_or_else(|| ModelDownloadError::MissingRuntime {
            model: requirement.model.clone(),
            runtime: runtime_key.to_owned(),
        })?;

    if !runtime_spec.available {
        return Err(ModelDownloadError::UnavailableRuntime {
            model: requirement.model.clone(),
            runtime: runtime_key.to_owned(),
        });
    }

    if !runtime_spec
        .precisions
        .iter()
        .any(|precision| precision == &requirement.precision)
    {
        return Err(ModelDownloadError::MissingPrecision {
            model: requirement.model.clone(),
            runtime: runtime_key.to_owned(),
            precision: requirement.precision.clone(),
        });
    }

    Ok(runtime_spec)
}

fn target_path_for_remote(
    model_dir: &Path,
    remote_path: &str,
) -> Result<PathBuf, ModelDownloadError> {
    let mut target = model_dir.to_path_buf();
    for segment in remote_path.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." {
            return Err(ModelDownloadError::InvalidPath {
                path: remote_path.to_owned(),
            });
        }
        target.push(segment);
    }
    Ok(target)
}

fn is_default_root_file(path: &str) -> bool {
    if path.contains('/') {
        return false;
    }

    !matches!(
        Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("npy" | "bin")
    )
}

fn is_dataset_file(path: &str, dataset: &Option<String>) -> bool {
    let Some(dataset) = dataset.as_deref() else {
        return false;
    };

    path.strip_prefix("datasets/")
        .is_some_and(|file_name| file_name.starts_with(&format!("{dataset}.")))
        && path.split('/').count() == 2
}

fn is_root_dataset_file(path: &str, dataset: &Option<String>) -> bool {
    let Some(dataset) = dataset.as_deref() else {
        return false;
    };

    !path.contains('/') && path.starts_with(&format!("{dataset}."))
}

fn runtime_asset_key(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Onnx | Runtime::CandleOnnx => "onnx",
        Runtime::Mnn => "mnn",
        Runtime::Rknn => "rknn",
    }
}

fn runtime_asset_dir(runtime: Runtime) -> &'static str {
    runtime_asset_key(runtime)
}

fn runtime_asset_ext(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Onnx | Runtime::CandleOnnx => "onnx",
        Runtime::Mnn => "mnn",
        Runtime::Rknn => "rknn",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelRequirement {
    model: String,
    runtime: Runtime,
    precision: String,
    dataset: Option<String>,
}

trait ModelRepoClient {
    fn endpoint_label(&self) -> &str;

    fn list_repo_files(&self, model: &str) -> Result<Vec<String>, ModelDownloadError>;

    fn download_file(
        &self,
        model: &str,
        remote_path: &str,
        target: &Path,
    ) -> Result<(), ModelDownloadError>;
}

struct HfHubClient {
    agent: ureq::Agent,
    endpoint: &'static str,
}

impl HfHubClient {
    fn new(region: Region) -> Self {
        let endpoint = match region {
            Region::Cn => "https://hf-mirror.com",
            Region::Other => "https://huggingface.co",
        };
        Self {
            agent: ureq::Agent::new_with_defaults(),
            endpoint,
        }
    }
}

impl ModelRepoClient for HfHubClient {
    fn endpoint_label(&self) -> &str {
        self.endpoint
    }

    fn list_repo_files(&self, model: &str) -> Result<Vec<String>, ModelDownloadError> {
        let url = format!(
            "{}/api/models/{HF_ORG}/{}/tree/main?recursive=1",
            self.endpoint,
            encode_path_segment(model)
        );
        let spinner = ProgressBar::new_spinner();
        spinner.enable_steady_tick(Duration::from_millis(100));
        spinner.set_message(format!("listing files for {HF_ORG}/{model}"));
        let mut response = self.agent.get(&url).call()?;
        let body = response.body_mut().read_to_string()?;
        let entries = serde_json::from_str::<Vec<HfTreeEntry>>(&body)?;
        spinner.finish_and_clear();
        info!(repo = %format!("{HF_ORG}/{model}"), "listed repository files");

        Ok(entries
            .into_iter()
            .filter(|entry| {
                entry.kind == "file"
                    && (!entry.path.contains('/') || entry.path.starts_with("datasets/"))
            })
            .map(|entry| entry.path)
            .collect())
    }

    fn download_file(
        &self,
        model: &str,
        remote_path: &str,
        target: &Path,
    ) -> Result<(), ModelDownloadError> {
        let url = format!(
            "{}/{HF_ORG}/{}/resolve/main/{}",
            self.endpoint,
            encode_path_segment(model),
            encode_remote_path(remote_path)
        );
        let mut response = self.agent.get(&url).call()?;
        let content_len = response
            .headers()
            .get("content-length")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok());
        let mut output =
            fs::File::create(target).map_err(|source| ModelDownloadError::WriteFile {
                path: target.to_path_buf(),
                source,
            })?;

        let label = format!("{model}/{remote_path}");
        let mut reader = response.body_mut().as_reader();
        let mut buffer = [0_u8; 128 * 1024];
        let mut written = 0_u64;
        let progress = content_len.map(|len| download_progress_bar(len, &label));
        let fallback_spinner = if progress.is_none() {
            let spinner = ProgressBar::new_spinner();
            spinner.enable_steady_tick(Duration::from_millis(100));
            spinner.set_message(format!("downloading {label}"));
            Some(spinner)
        } else {
            None
        };

        loop {
            let read =
                reader
                    .read(&mut buffer)
                    .map_err(|source| ModelDownloadError::WriteFile {
                        path: target.to_path_buf(),
                        source,
                    })?;
            if read == 0 {
                break;
            }
            output
                .write_all(&buffer[..read])
                .map_err(|source| ModelDownloadError::WriteFile {
                    path: target.to_path_buf(),
                    source,
                })?;
            written += read as u64;
            if let Some(progress) = &progress {
                progress.inc(read as u64);
            }
        }

        output
            .flush()
            .map_err(|source| ModelDownloadError::WriteFile {
                path: target.to_path_buf(),
                source,
            })?;

        if let Some(progress) = progress {
            progress.finish_and_clear();
        }
        if let Some(spinner) = fallback_spinner {
            spinner.finish_and_clear();
        }
        info!(file = %label, size = %format_bytes(written), "downloaded model artifact");
        Ok(())
    }
}

fn ensure_remote_file<C: ModelRepoClient>(
    client: &C,
    model: &str,
    remote_path: &str,
    target: &Path,
) -> Result<(), ModelDownloadError> {
    if target.is_file() {
        let size = fs::metadata(target)
            .ok()
            .map(|metadata| metadata.len())
            .map(format_bytes)
            .unwrap_or_else(|| "unknown size".to_owned());
        info!(file = %format!("{model}/{remote_path}"), size = %size, "model cache hit");
        return Ok(());
    }

    let parent = target
        .parent()
        .ok_or_else(|| ModelDownloadError::InvalidPath {
            path: target.display().to_string(),
        })?;
    fs::create_dir_all(parent).map_err(|source| ModelDownloadError::CreateDir {
        path: parent.to_path_buf(),
        source,
    })?;

    let file_name = target
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ModelDownloadError::InvalidPath {
            path: target.display().to_string(),
        })?;
    let tmp = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
    if tmp.exists() {
        fs::remove_file(&tmp).map_err(|source| ModelDownloadError::WriteFile {
            path: tmp.clone(),
            source,
        })?;
    }

    client.download_file(model, remote_path, &tmp)?;
    fs::rename(&tmp, target).map_err(|source| ModelDownloadError::WriteFile {
        path: target.to_path_buf(),
        source,
    })?;

    Ok(())
}

fn download_progress_bar(len: u64, label: &str) -> ProgressBar {
    let progress = ProgressBar::new(len);
    let style = ProgressStyle::with_template(
        "{spinner:.green} {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-");
    progress.set_style(style);
    progress.set_message(format!("downloading {label}"));
    progress
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes_f = bytes as f64;
    if bytes_f >= GIB {
        format!("{:.2} GiB", bytes_f / GIB)
    } else if bytes_f >= MIB {
        format!("{:.1} MiB", bytes_f / MIB)
    } else if bytes_f >= KIB {
        format!("{:.1} KiB", bytes_f / KIB)
    } else {
        format!("{bytes} B")
    }
}

#[derive(Debug, Deserialize)]
struct HfTreeEntry {
    path: String,
    #[serde(rename = "type")]
    kind: String,
}

fn encode_remote_path(path: &str) -> String {
    path.split('/')
        .map(encode_path_segment)
        .collect::<Vec<_>>()
        .join("/")
}

fn encode_path_segment(segment: &str) -> String {
    let mut encoded = String::new();
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

#[derive(Debug, Error)]
pub enum ModelDownloadError {
    #[error("invalid model download config: {0}")]
    InvalidConfig(String),

    #[error("invalid model path `{path}`")]
    InvalidPath { path: String },

    #[error("failed to create directory `{}`: {source}", path.display())]
    CreateDir {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("failed to read file `{}`: {source}", path.display())]
    ReadFile {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("failed to write file `{}`: {source}", path.display())]
    WriteFile {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("Hugging Face request failed: {0}")]
    Http(#[from] ureq::Error),

    #[error("Hugging Face response json parsing failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid model_info.json: {0}")]
    ModelInfo(#[from] ModelInfoValidationError),

    #[error("model `{model}` does not provide runtime `{runtime}` in model_info.json")]
    MissingRuntime { model: String, runtime: String },

    #[error("model `{model}` marks runtime `{runtime}` as unavailable in model_info.json")]
    UnavailableRuntime { model: String, runtime: String },

    #[error(
        "model `{model}` runtime `{runtime}` does not provide precision `{precision}` in model_info.json"
    )]
    MissingPrecision {
        model: String,
        runtime: String,
        precision: String,
    },
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        collections::BTreeMap,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    use serde_json::json;

    use super::{ModelDownloadError, ModelRepoClient, ensure_models_with_client};
    use lumen_schema::LumenConfig;

    #[test]
    fn downloads_runtime_artifacts_root_files_and_dataset_files() {
        let cache = temp_cache_dir("selection");
        let client = FakeClient::new(
            vec![
                "config.json",
                "tokenizer.json",
                "labels.npy",
                "weights.bin",
                "TreeOfLife200M.npy",
                "TreeOfLife200M.bin",
                "TreeOfLife200M.json",
                "datasets/TreeOfLife200M.npy",
                "datasets/TreeOfLife200M.bin",
                "datasets/TreeOfLife200M.json",
                "datasets/TreeOfLife200MCore.npy",
                "onnx/ignored.onnx",
                "nested/ignored.json",
            ],
            model_info_json(&["fp32"], true),
        );
        let config = test_config(Some("TreeOfLife200M"), Some("fp32"));

        ensure_models_with_client(&config, &cache, &client).unwrap();

        let downloaded = client.downloaded_paths();
        assert_eq!(
            downloaded,
            vec![
                "antelopev2:model_info.json",
                "antelopev2:config.json",
                "antelopev2:datasets/TreeOfLife200M.bin",
                "antelopev2:datasets/TreeOfLife200M.json",
                "antelopev2:datasets/TreeOfLife200M.npy",
                "antelopev2:onnx/text.fp32.onnx",
                "antelopev2:onnx/vision.fp32.onnx",
                "antelopev2:tokenizer.json",
            ]
        );
        assert!(
            cache
                .join("antelopev2/datasets/TreeOfLife200M.bin")
                .is_file()
        );
        assert!(!cache.join("antelopev2/labels.npy").exists());
        assert!(!cache.join("antelopev2/weights.bin").exists());
        assert!(!cache.join("antelopev2/TreeOfLife200M.bin").exists());
        assert!(
            !cache
                .join("antelopev2/datasets/TreeOfLife200MCore.npy")
                .exists()
        );

        cleanup_cache(cache);
    }

    #[test]
    fn rejects_missing_runtime_precision() {
        let cache = temp_cache_dir("precision");
        let client = FakeClient::new(vec![], model_info_json(&["fp32"], true));
        let config = test_config(None, Some("fp16"));

        let error = ensure_models_with_client(&config, &cache, &client).unwrap_err();
        assert!(matches!(
            error,
            ModelDownloadError::MissingPrecision {
                model,
                runtime,
                precision
            } if model == "antelopev2" && runtime == "onnx" && precision == "fp16"
        ));

        cleanup_cache(cache);
    }

    #[test]
    fn skips_files_that_already_exist() {
        let cache = temp_cache_dir("existing");
        let model_dir = cache.join("antelopev2");
        std::fs::create_dir_all(model_dir.join("onnx")).unwrap();
        std::fs::write(
            model_dir.join("model_info.json"),
            model_info_json(&["fp32"], true),
        )
        .unwrap();
        std::fs::write(model_dir.join("onnx/vision.fp32.onnx"), "existing").unwrap();

        let client = FakeClient::new(vec![], model_info_json(&["fp32"], true));
        let config = test_config(None, Some("fp32"));

        ensure_models_with_client(&config, &cache, &client).unwrap();

        assert_eq!(
            std::fs::read_to_string(model_dir.join("onnx/vision.fp32.onnx")).unwrap(),
            "existing"
        );
        assert_eq!(
            client.downloaded_paths(),
            vec!["antelopev2:onnx/text.fp32.onnx"]
        );

        cleanup_cache(cache);
    }

    fn test_config(dataset: Option<&str>, precision: Option<&str>) -> LumenConfig {
        LumenConfig::from_json_str(
            &json!({
                "metadata": {
                    "version": "0.1.0",
                    "region": "other",
                    "cache_dir": "/tmp/lumen-test"
                },
                "deployment": {
                    "mode": "hub",
                    "services": ["clip_service"]
                },
                "server": {
                    "port": 50051
                },
                "services": {
                    "clip_service": {
                        "enabled": true,
                        "package": "clip",
                        "models": {
                            "default": {
                                "model": "antelopev2",
                                "runtime": "onnx",
                                "dataset": dataset,
                                "precision": precision
                            }
                        }
                    }
                }
            })
            .to_string(),
        )
        .unwrap()
    }

    fn model_info_json(precisions: &[&str], available: bool) -> String {
        json!({
            "name": "antelopev2",
            "version": "1.0.0",
            "description": "Test model package for downloader tests.",
            "model_type": "clip",
            "source": {
                "format": "huggingface",
                "repo_id": "Lumilio-Photos/antelopev2"
            },
            "runtimes": {
                "onnx": {
                    "available": available,
                    "components": ["vision", "text"],
                    "precisions": precisions
                }
            }
        })
        .to_string()
    }

    struct FakeClient {
        root_files: Vec<String>,
        model_info: String,
        downloads: RefCell<Vec<String>>,
        contents: BTreeMap<String, String>,
    }

    impl FakeClient {
        fn new(root_files: Vec<&str>, model_info: String) -> Self {
            Self {
                root_files: root_files.into_iter().map(str::to_owned).collect(),
                model_info,
                downloads: RefCell::new(Vec::new()),
                contents: BTreeMap::new(),
            }
        }

        fn downloaded_paths(&self) -> Vec<String> {
            self.downloads.borrow().clone()
        }
    }

    impl ModelRepoClient for FakeClient {
        fn endpoint_label(&self) -> &str {
            "fake"
        }

        fn list_repo_files(&self, _model: &str) -> Result<Vec<String>, ModelDownloadError> {
            Ok(self.root_files.clone())
        }

        fn download_file(
            &self,
            model: &str,
            remote_path: &str,
            target: &Path,
        ) -> Result<(), ModelDownloadError> {
            self.downloads
                .borrow_mut()
                .push(format!("{model}:{remote_path}"));
            let contents = self.contents.get(remote_path).cloned().unwrap_or_else(|| {
                if remote_path == "model_info.json" {
                    self.model_info.clone()
                } else {
                    format!("fake {remote_path}")
                }
            });
            std::fs::write(target, contents).unwrap();
            Ok(())
        }
    }

    fn temp_cache_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "lumen-hub-model-download-{name}-{}-{nanos}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    fn cleanup_cache(path: PathBuf) {
        let _ = std::fs::remove_dir_all(path);
    }
}
