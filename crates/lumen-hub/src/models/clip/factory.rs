use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use lumen_schema::{ModelInfo, Runtime};
#[cfg(feature = "candle")]
use lumnn::candle::node::CandleOnnxNode;
#[cfg(feature = "mnn")]
use lumnn::mnn::MnnNode;
use lumnn::{
    core::{context::MLContext, node::MLNode},
    ort::node::OrtNode,
};

use crate::service::{ServiceError, ServiceResult};

/// Resolves model artifacts using the Lumen model repository convention.
///
/// Repository layout under `{cache_dir}/{model_name}/`:
///
/// ```text
/// model_info.json                  # ModelInfo schema
/// tokenizer.json                   # HuggingFace tokenizer (text models)
/// {runtime}/text.{precision}.{ext} # e.g. onnx/text.fp16.onnx
/// {runtime}/vision.{precision}.{ext}
/// ...
/// ```
///
/// The extension is derived from the runtime: `.onnx` for ONNX, `.rknn` for RKNN.
pub struct ClipModelFactory {
    cache_dir: String,
}

/// Factory for BioCLIP models.
///
/// BioCLIP shares the CLIP vision encoder artifact convention, but resolves
/// dataset assets (`*.npy` text embeddings and `*.json` labels) from the model
/// repository instead of loading a text encoder.
pub struct BioClipModelFactory {
    inner: ClipModelFactory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BioClipDatasetPaths {
    pub embeddings_path: PathBuf,
    pub labels_path: PathBuf,
    pub index_path: Option<PathBuf>,
}

impl BioClipModelFactory {
    pub fn new(cache_dir: impl Into<String>) -> Self {
        Self {
            inner: ClipModelFactory::new(cache_dir),
        }
    }

    pub fn model_dir(&self, model_name: &str) -> PathBuf {
        self.inner.model_dir(model_name)
    }

    pub fn load_model_info(&self, model_name: &str) -> ServiceResult<ModelInfo> {
        self.inner.load_model_info(model_name)
    }

    pub fn create_vision_component(
        &self,
        model_name: &str,
        runtime: Runtime,
        precision: &str,
        context: &Arc<MLContext>,
    ) -> ServiceResult<Box<dyn MLNode>> {
        self.inner
            .create_component(model_name, runtime, "vision", precision, context)
    }

    pub fn resolve_dataset_paths(
        &self,
        model_name: &str,
        dataset: &str,
    ) -> ServiceResult<BioClipDatasetPaths> {
        let search_dirs = self.dataset_search_dirs(model_name);
        let embeddings_path = resolve_existing_dataset_file(
            &search_dirs,
            &[format!("{dataset}.npy")],
            "BioCLIP text embeddings",
        )?;
        let label_names = label_filename_candidates(dataset);
        let labels_path =
            resolve_existing_dataset_file(&search_dirs, &label_names, "BioCLIP labels")?;
        let index_path = resolve_optional_dataset_file(&search_dirs, &[format!("{dataset}.bin")]);

        Ok(BioClipDatasetPaths {
            embeddings_path,
            labels_path,
            index_path,
        })
    }

    fn dataset_search_dirs(&self, model_name: &str) -> Vec<PathBuf> {
        let model_dir = self.model_dir(model_name);
        let cache_dir = PathBuf::from(&self.inner.cache_dir);
        let mut dirs = Vec::new();
        for dir in [
            model_dir.clone(),
            model_dir.join("datasets"),
            cache_dir.clone(),
            cache_dir.join("datasets"),
        ] {
            if !dirs.iter().any(|existing| existing == &dir) {
                dirs.push(dir);
            }
        }
        dirs
    }
}

impl ClipModelFactory {
    pub fn new(cache_dir: impl Into<String>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
        }
    }

    /// Returns the root directory for a given model.
    pub fn model_dir(&self, model_name: &str) -> PathBuf {
        PathBuf::from(&self.cache_dir).join(model_name)
    }

    /// Loads and validates the `model_info.json` for a model.
    pub fn load_model_info(&self, model_name: &str) -> ServiceResult<ModelInfo> {
        let path = self.model_dir(model_name).join("model_info.json");
        let contents = fs::read_to_string(&path).map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "failed to read model_info.json at {}: {e}",
                path.display()
            ))
        })?;
        ModelInfo::from_json_str(&contents).map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "invalid model_info.json at {}: {e}",
                path.display()
            ))
        })
    }

    /// Resolves the path for a specific component artifact.
    ///
    /// Convention: `{cache_dir}/{model_name}/{runtime}/{component}.{precision}.{ext}`
    pub fn resolve_component_path(
        &self,
        model_name: &str,
        runtime: Runtime,
        component: &str,
        precision: &str,
    ) -> PathBuf {
        let runtime_dir = match runtime {
            Runtime::Onnx | Runtime::CandleOnnx => "onnx",
            Runtime::Rknn => "rknn",
            Runtime::Mnn => "mnn",
        };
        let ext = match runtime {
            Runtime::Onnx | Runtime::CandleOnnx => "onnx",
            Runtime::Rknn => "rknn",
            Runtime::Mnn => "mnn",
        };
        let filename = format!("{component}.{precision}.{ext}");
        self.model_dir(model_name).join(runtime_dir).join(filename)
    }

    /// Creates a model-forward node for a specific component.
    pub fn create_component(
        &self,
        model_name: &str,
        runtime: Runtime,
        component: &str,
        precision: &str,
        context: &Arc<MLContext>,
    ) -> ServiceResult<Box<dyn MLNode>> {
        let model_path = self.resolve_component_path(model_name, runtime, component, precision);
        let path_str = model_path.to_str().ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model path is not valid UTF-8: {}",
                model_path.display()
            ))
        })?;
        let name = format!("{model_name}_{component}");
        match runtime {
            Runtime::Onnx => OrtNode::new(context.as_ref(), path_str, name)
                .map(|node| Box::new(node) as Box<dyn MLNode>)
                .map_err(ServiceError::Internal),
            #[cfg(feature = "candle")]
            Runtime::CandleOnnx => CandleOnnxNode::new(context.as_ref(), path_str, name)
                .map(|node| Box::new(node) as Box<dyn MLNode>)
                .map_err(ServiceError::Internal),
            #[cfg(not(feature = "candle"))]
            Runtime::CandleOnnx => Err(ServiceError::InvalidArgument(
                "CLIP Candle ONNX runtime is not enabled in this lumen-hub build; use runtime=onnx"
                    .to_owned(),
            )),
            Runtime::Rknn => Err(ServiceError::InvalidArgument(
                "CLIP RKNN runtime is not implemented yet".to_owned(),
            )),
            #[cfg(feature = "mnn")]
            Runtime::Mnn => MnnNode::new(context.as_ref(), path_str, name)
                .map(|node| Box::new(node) as Box<dyn MLNode>)
                .map_err(ServiceError::Internal),
            #[cfg(not(feature = "mnn"))]
            Runtime::Mnn => Err(ServiceError::InvalidArgument(
                "CLIP MNN runtime is not enabled in this lumen-hub build".to_owned(),
            )),
        }
    }

    /// Loads the HuggingFace tokenizer from the model directory root.
    pub fn load_tokenizer(&self, model_name: &str) -> ServiceResult<tokenizers::Tokenizer> {
        let tokenizer_path = self.model_dir(model_name).join("tokenizer.json");
        tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            ServiceError::InvalidArgument(format!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            ))
        })
    }
}

fn resolve_existing_dataset_file(
    search_dirs: &[PathBuf],
    filenames: &[String],
    description: &str,
) -> ServiceResult<PathBuf> {
    let mut checked = Vec::new();
    for dir in search_dirs {
        for filename in filenames {
            let path = dir.join(filename);
            checked.push(path.display().to_string());
            if path.is_file() {
                return Ok(path);
            }
        }
    }

    Err(ServiceError::InvalidArgument(format!(
        "failed to resolve {description}; checked: {}",
        checked.join(", ")
    )))
}

fn resolve_optional_dataset_file(search_dirs: &[PathBuf], filenames: &[String]) -> Option<PathBuf> {
    for dir in search_dirs {
        for filename in filenames {
            let path = dir.join(filename);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn label_filename_candidates(dataset: &str) -> Vec<String> {
    let mut stems = strip_dataset_size_suffix(dataset)
        .map(|stem| vec![stem, dataset.to_owned()])
        .unwrap_or_else(|| vec![dataset.to_owned()]);
    stems.dedup();
    stems
        .into_iter()
        .filter(|stem| !stem.is_empty())
        .map(|stem| format!("{stem}.json"))
        .collect()
}

fn strip_dataset_size_suffix(dataset: &str) -> Option<String> {
    let stem = Path::new(dataset)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(dataset);
    let trimmed = stem.trim_end_matches(|ch: char| ch.is_ascii_digit());
    let trimmed = trimmed.trim_end_matches(|ch| matches!(ch, 'K' | 'M' | 'B' | 'k' | 'm' | 'b'));
    let trimmed = trimmed.trim_end_matches(|ch: char| ch.is_ascii_digit());
    if trimmed.len() < stem.len() {
        Some(trimmed.to_owned())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{label_filename_candidates, strip_dataset_size_suffix};

    #[test]
    fn bioclip_label_candidates_include_size_stripped_stem() {
        let candidates = label_filename_candidates("TreeOfLife200M");

        assert_eq!(
            candidates,
            vec![
                "TreeOfLife.json".to_owned(),
                "TreeOfLife200M.json".to_owned()
            ]
        );
    }

    #[test]
    fn strips_numeric_dataset_suffix_with_optional_unit() {
        assert_eq!(
            strip_dataset_size_suffix("TreeOfLife200M"),
            Some("TreeOfLife".to_owned())
        );
        assert_eq!(
            strip_dataset_size_suffix("dataset42"),
            Some("dataset".to_owned())
        );
        assert_eq!(strip_dataset_size_suffix("TreeOfLife"), None);
    }
}
