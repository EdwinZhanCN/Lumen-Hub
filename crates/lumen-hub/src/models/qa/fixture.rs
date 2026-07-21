//! QA model fixture writer: produces a real, loadable model repository
//! (burnpack weights in three precisions + model_info.json) in milliseconds.
//! Integration tests call this to seed the mock model repo — no binary
//! fixtures are committed.

use std::{fs, path::Path};

use burn_store::{HalfPrecisionAdapter, ModuleSnapshot};

use super::model::QaNet;
use super::service::QA_COMPONENT;
use crate::backend::default_device;

pub const QA_MODEL_NAME: &str = "qa-tiny";
pub const QA_PRECISIONS: [&str; 3] = ["fp32", "fp16", "fp16q8"];

/// Writes the full QA model repository under `model_dir` (typically
/// `<repo>/qa-tiny/`): `model_info.json` + `burn/net.{fp32,fp16,fp16q8}.bpk`.
///
/// The fp16q8 artifact is a byte-identical copy of fp16 — Q8 quantization is
/// applied at load time by `load_burnpack`, mirroring production packaging.
pub fn write_model_fixture(model_dir: &Path) -> Result<(), String> {
    let burn_dir = model_dir.join("burn");
    fs::create_dir_all(&burn_dir).map_err(|e| format!("create {}: {e}", burn_dir.display()))?;

    let device = default_device();
    let net = QaNet::deterministic(&device);

    let fp32_path = burn_dir.join(format!("{QA_COMPONENT}.fp32.bpk"));
    let mut store = burn_store::BurnpackStore::from_file(&fp32_path);
    net.save_into(&mut store)
        .map_err(|e| format!("save {}: {e}", fp32_path.display()))?;

    let fp16_path = burn_dir.join(format!("{QA_COMPONENT}.fp16.bpk"));
    let mut store = burn_store::BurnpackStore::from_file(&fp16_path)
        .with_to_adapter(HalfPrecisionAdapter::new());
    net.save_into(&mut store)
        .map_err(|e| format!("save {}: {e}", fp16_path.display()))?;

    let fp16q8_path = burn_dir.join(format!("{QA_COMPONENT}.fp16q8.bpk"));
    fs::copy(&fp16_path, &fp16q8_path)
        .map_err(|e| format!("copy {}: {e}", fp16q8_path.display()))?;

    let model_info = model_info_json();
    // Self-check: the fixture must satisfy the real schema validator.
    lumen_schema::ModelInfo::from_json_str(&model_info)
        .map_err(|e| format!("qa fixture model_info.json is invalid: {e}"))?;
    let info_path = model_dir.join("model_info.json");
    fs::write(&info_path, model_info).map_err(|e| format!("write {}: {e}", info_path.display()))?;

    Ok(())
}

pub fn model_info_json() -> String {
    serde_json::json!({
        "name": QA_MODEL_NAME,
        "version": "1.0.0",
        "description": "Tiny deterministic QA model for the e2e harness.",
        "model_type": "qa",
        "source": { "format": "huggingface", "repo_id": format!("Lumilio-Photos/{QA_MODEL_NAME}") },
        "runtimes": {
            "burn": {
                "available": true,
                "components": [QA_COMPONENT],
                "precisions": QA_PRECISIONS,
            }
        }
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lumen_schema::ServiceConfig;

    use super::*;
    use crate::backend::default_device;
    use crate::models::qa::model::QA_INPUT_NUMEL;
    use crate::models::qa::service::QaService;
    use crate::models::qa::task::QA_EMBED_TASK;
    use crate::service::{InferenceService, TaskRequest};

    fn service_config(precision: &str) -> ServiceConfig {
        serde_json::from_value(serde_json::json!({
            "enabled": true,
            "package": "qa",
            "models": {
                "default": { "model": QA_MODEL_NAME, "runtime": "burn", "precision": precision }
            }
        }))
        .expect("qa service config")
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    async fn embed_with(cache: &std::path::Path, precision: &str, pixels: &[f32]) -> Vec<f32> {
        let config = service_config(precision);
        let service = QaService::from_config(
            "qa",
            &config,
            &cache.display().to_string(),
            Arc::new(default_device()),
        )
        .unwrap_or_else(|e| panic!("qa service ({precision}): {e}"));
        let mut bytes = Vec::with_capacity(pixels.len() * 4);
        for value in pixels {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut request = TaskRequest::new(bytes, crate::service::DEFAULT_TENSOR_MIME);
        for (key, value) in crate::models::qa::tensor_request_meta() {
            request = request.with_meta(key, value);
        }
        let result = service
            .tasks()
            .handle(QA_EMBED_TASK, request)
            .await
            .unwrap_or_else(|e| panic!("qa embed ({precision}): {e}"));
        let embedding: lumen_schema::EmbeddingV1 =
            serde_json::from_slice(&result.payload).expect("embedding_v1");
        embedding.vector
    }

    #[tokio::test]
    async fn fixture_roundtrips_through_all_precisions() {
        let dir = tempdir();
        let model_dir = dir.join(QA_MODEL_NAME);
        write_model_fixture(&model_dir).expect("write fixture");

        let pixels: Vec<f32> = (0..QA_INPUT_NUMEL)
            .map(|i| (i % 251) as f32 / 251.0)
            .collect();

        let fp32 = embed_with(&dir, "fp32", &pixels).await;
        let fp16 = embed_with(&dir, "fp16", &pixels).await;
        let fp16q8 = embed_with(&dir, "fp16q8", &pixels).await;

        let norm: f32 = fp32.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "fp32 norm {norm}");
        assert!(
            cosine(&fp32, &fp16) > 0.999,
            "fp16 drift: {}",
            cosine(&fp32, &fp16)
        );
        assert!(
            cosine(&fp32, &fp16q8) > 0.99,
            "fp16q8 drift: {}",
            cosine(&fp32, &fp16q8)
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "lumen-qa-fixture-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
