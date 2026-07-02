//! End-to-end InsightFace test on the active Burn backend.
//!
//! Run on CPU:   cargo test --test e2e_insightface
//! Run on Metal: cargo test --features metal --test e2e_insightface

mod common;

use std::sync::Arc;

use lumen_hub::backend::default_device;
use lumen_hub::models::insightface::InsightFaceService;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::FaceV1;

const MODEL: &str = "antelopev2";

#[test]
fn insightface_detects_face_and_produces_embedding() {
    const STACK: usize = 256 * 1024 * 1024;
    std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_stack_size(STACK)
                .build()
                .expect("tokio runtime")
                .block_on(body());
        })
        .expect("spawn test thread")
        .join()
        .expect("test thread panicked");
}

async fn body() {
    let Some((cache_dir, model_name)) = common::require_model(MODEL, &["detection", "recognition"])
    else {
        return;
    };
    let config = common::service_config("insightface", &model_name);
    let device = Arc::new(default_device());
    let service = InsightFaceService::from_config("insightface", &config, &cache_dir, device)
        .expect("InsightFace service builds from config");

    let task = service
        .tasks()
        .task_names()
        .into_iter()
        .next()
        .expect("InsightFace exposes a task");

    let image = common::sample_bytes("tests/test_sample/face_test_1.png");
    let result = service
        .tasks()
        .handle(&task, TaskRequest::new(image, "image/png"))
        .await
        .expect("face recognition task succeeds");

    let faces: FaceV1 = serde_json::from_slice(&result.payload).expect("face_v1 JSON");
    eprintln!("insightface detected {} face(s)", faces.count);
    assert!(faces.count > 0, "expected at least one detected face");

    let face = &faces.faces[0];
    let embedding = face.embedding.as_ref().expect("face has an embedding");
    assert_eq!(embedding.len(), 512, "ArcFace embedding dim");
    let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "embedding should be L2-normalized, got {norm}"
    );
    assert!(face.confidence > 0.0 && face.confidence <= 1.0);
}
