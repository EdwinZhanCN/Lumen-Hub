//! Shared helpers for Burn end-to-end model tests.
//!
//! The tests load real FP32 model weights from a local model repository. The
//! directory defaults to `lumen-models` and can be overridden with
//! `LUMEN_MODELS_DIR`. When the directory or a specific model is missing the
//! tests skip gracefully so they remain green in environments without the
//! (large) model assets.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use lumen_schema::{ModelConfig, Runtime, ServiceConfig};

pub fn models_dir() -> PathBuf {
    std::env::var("LUMEN_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("lumen-models"))
}

/// Returns the cache dir + model name if the model (and its fp32 burn weights
/// for `components`) are present; otherwise prints a skip notice and returns
/// `None`.
pub fn require_model(model_name: &str, components: &[&str]) -> Option<(String, String)> {
    require_model_precision(model_name, components, "fp32")
}

/// Returns the cache dir + model name if the model (and its burn weights for
/// `components` at `precision`) are present; otherwise prints a skip notice and
/// returns `None`.
pub fn require_model_precision(
    model_name: &str,
    components: &[&str],
    precision: &str,
) -> Option<(String, String)> {
    let dir = models_dir();
    let model_dir = dir.join(model_name);
    if !model_dir.join("model_info.json").is_file() {
        eprintln!("SKIP: model `{model_name}` not found; set LUMEN_MODELS_DIR");
        return None;
    }
    for component in components {
        let weight = model_dir
            .join("burn")
            .join(format!("{component}.{precision}.bpk"));
        if !weight.is_file() {
            eprintln!(
                "SKIP: missing {precision} burn weight for `{model_name}` component `{component}`"
            );
            return None;
        }
    }
    Some((dir.to_string_lossy().into_owned(), model_name.to_owned()))
}

pub fn has_burn_weights(
    cache_dir: &str,
    model_name: &str,
    components: &[&str],
    precision: &str,
) -> bool {
    let model_dir = Path::new(cache_dir).join(model_name).join("burn");
    components.iter().all(|component| {
        model_dir
            .join(format!("{component}.{precision}.bpk"))
            .is_file()
    })
}

pub fn service_config(package: &str, model_name: &str) -> ServiceConfig {
    ServiceConfig {
        enabled: true,
        package: package.to_owned(),
        models: BTreeMap::from([(
            "default".to_owned(),
            ModelConfig {
                model: model_name.to_owned(),
                runtime: Runtime::Burn,
                dataset: None,
                precision: Some("fp32".to_owned()),
            },
        )]),
    }
}

pub fn sample_bytes(relative: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read sample `{relative}`: {e}"))
}
