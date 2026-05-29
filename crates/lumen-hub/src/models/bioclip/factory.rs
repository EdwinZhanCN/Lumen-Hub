use std::{
    fs,
    path::{Path, PathBuf},
};

use lumen_schema::{ModelInfo, Runtime};

use super::model::BioClipVisionModel;
use crate::backend::Device;
use crate::service::{ServiceError, ServiceResult};

/// Resolved on-disk paths for a BioCLIP taxon catalog.
pub struct BioClipDatasetPaths {
    pub embeddings_path: PathBuf,
    pub labels_path: PathBuf,
    pub index_path: Option<PathBuf>,
}

/// Resolves BioCLIP artifacts under `{cache_dir}/{model_name}/`:
/// `burn/vision.{precision}.bpk` plus `datasets/<dataset>.{npy,json,bin}`.
pub struct BioClipModelFactory {
    cache_dir: String,
}

impl BioClipModelFactory {
    pub fn new(cache_dir: impl Into<String>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
        }
    }

    pub fn model_dir(&self, model_name: &str) -> PathBuf {
        PathBuf::from(&self.cache_dir).join(model_name)
    }

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

    pub fn create_vision_model(
        &self,
        model_name: &str,
        runtime: Runtime,
        precision: &str,
        device: &Device,
    ) -> ServiceResult<BioClipVisionModel> {
        match runtime {
            Runtime::Burn => {}
        }
        let path = self
            .model_dir(model_name)
            .join("burn")
            .join(format!("vision.{precision}.bpk"));
        if !path.exists() {
            return Err(ServiceError::InvalidArgument(format!(
                "BioCLIP vision weights not found at {}",
                path.display()
            )));
        }
        let path_str = path.to_str().ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "model path is not valid UTF-8: {}",
                path.display()
            ))
        })?;
        BioClipVisionModel::load(model_name, path_str, device.clone())
            .map_err(ServiceError::InvalidArgument)
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
        let labels_path = resolve_existing_dataset_file(
            &search_dirs,
            &label_filename_candidates(dataset),
            "BioCLIP labels",
        )?;
        let index_path = resolve_optional_dataset_file(&search_dirs, &[format!("{dataset}.bin")]);

        Ok(BioClipDatasetPaths {
            embeddings_path,
            labels_path,
            index_path,
        })
    }

    fn dataset_search_dirs(&self, model_name: &str) -> Vec<PathBuf> {
        let model_dir = self.model_dir(model_name);
        let cache_dir = PathBuf::from(&self.cache_dir);
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
