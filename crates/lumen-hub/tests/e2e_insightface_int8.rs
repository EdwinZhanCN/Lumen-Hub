//! int8 runtime spot-check for antelopev2: compares fp32 vs int8 through the full
//! InsightFace service path. Embedding fidelity (ArcFace recognition — the
//! borderline component) is measured on a single clear face; detection parity on a
//! multi-face image.
//!
//! Run on CPU:   cargo test --test e2e_insightface_int8 -- --nocapture
//! Run on Metal: cargo test --features metal --test e2e_insightface_int8 -- --nocapture

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use lumen_hub::backend::default_device;
use lumen_hub::models::insightface::InsightFaceService;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::{FaceV1, ModelConfig, Runtime, ServiceConfig};

const MODEL: &str = "antelopev2";

fn config(model: &str, precision: &str) -> ServiceConfig {
    ServiceConfig {
        enabled: true,
        package: "insightface".to_owned(),
        models: BTreeMap::from([(
            "default".to_owned(),
            ModelConfig {
                model: model.to_owned(),
                runtime: Runtime::Burn,
                dataset: None,
                precision: Some(precision.to_owned()),
            },
        )]),
    }
}

async fn run(cache_dir: &str, model: &str, precision: &str, rel: &str, mime: &str) -> FaceV1 {
    let device = Arc::new(default_device());
    let service = InsightFaceService::from_config(
        "insightface",
        &config(model, precision),
        cache_dir,
        device,
    )
    .unwrap_or_else(|e| panic!("service builds ({precision}): {e:?}"));
    let task = service
        .tasks()
        .task_names()
        .into_iter()
        .next()
        .expect("task");
    let img = common::sample_bytes(rel);
    let result = service
        .tasks()
        .handle(&task, TaskRequest::new(img, mime))
        .await
        .unwrap_or_else(|e| panic!("face task ({precision}, {rel}): {e:?}"));
    serde_json::from_slice(&result.payload).expect("face_v1 JSON")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

async fn body(cache_dir: String, model: String) {
    // --- ArcFace embedding fidelity on a single clear face ---
    let f32_face = run(
        &cache_dir,
        &model,
        "fp32",
        "warmup/face/face.jpg",
        "image/jpeg",
    )
    .await;
    let i8_face = run(
        &cache_dir,
        &model,
        "int8",
        "warmup/face/face.jpg",
        "image/jpeg",
    )
    .await;
    assert!(
        f32_face.count > 0 && i8_face.count > 0,
        "expected a face in face.jpg"
    );
    let e32 = f32_face.faces[0]
        .embedding
        .as_ref()
        .expect("fp32 embedding");
    let e8 = i8_face.faces[0].embedding.as_ref().expect("int8 embedding");
    let emb_cos = cosine(e32, e8);
    eprintln!("\n=== antelopev2 int8 runtime spot-check ===");
    eprintln!(
        "face.jpg: embedding cosine fp32-vs-int8 = {emb_cos:.5} (dim {})",
        e32.len()
    );

    // --- detection parity on a multi-face image ---
    let f32_multi = run(
        &cache_dir,
        &model,
        "fp32",
        "tests/test_sample/face_test_1.png",
        "image/png",
    )
    .await;
    let i8_multi = run(
        &cache_dir,
        &model,
        "int8",
        "tests/test_sample/face_test_1.png",
        "image/png",
    )
    .await;
    eprintln!(
        "face_test_1.png: detected faces fp32={} int8={}",
        f32_multi.count, i8_multi.count
    );

    assert!(
        emb_cos > 0.95,
        "int8 ArcFace embedding drifted too far: cosine {emb_cos:.5}"
    );
    assert!(i8_multi.count > 0, "int8 detection found no faces");
}

#[test]
fn insightface_int8_matches_fp32() {
    let Some((cache_dir, model_name)) = common::require_model(MODEL, &["detection", "recognition"])
    else {
        return;
    };
    if common::require_model_precision(MODEL, &["detection", "recognition"], "int8").is_none() {
        return;
    }
    const STACK: usize = 256 * 1024 * 1024;
    std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(move || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_stack_size(STACK)
                .build()
                .expect("tokio runtime")
                .block_on(body(cache_dir, model_name));
        })
        .expect("spawn test thread")
        .join()
        .expect("test thread panicked");
}
