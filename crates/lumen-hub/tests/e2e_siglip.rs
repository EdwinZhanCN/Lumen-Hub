//! End-to-end SigLIP test on the active Burn backend (base + so400m variants).
//!
//! Run on CPU:   cargo test --test e2e_siglip
//! Run on Metal: cargo test --features metal --test e2e_siglip

mod common;

use std::sync::Arc;

use lumen_hub::backend::default_device;
use lumen_hub::models::siglip::SiglipService;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::EmbeddingV1;

async fn embed(service: &SiglipService, task: &str, request: TaskRequest) -> EmbeddingV1 {
    let result = service
        .tasks()
        .handle(task, request)
        .await
        .unwrap_or_else(|e| panic!("task `{task}` failed: {e}"));
    serde_json::from_slice(&result.payload).expect("embedding_v1 JSON")
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Shared checks: normalized embeddings + matching caption beats a distractor.
async fn check_siglip_model(model: &str) {
    let Some((cache_dir, model_name)) = common::require_model(model, &["text", "vision"]) else {
        return;
    };
    let config = common::service_config("siglip", &model_name);
    let device = Arc::new(default_device());
    let service = SiglipService::from_config("siglip", &config, &cache_dir, device)
        .expect("SigLIP service builds from config");

    let image = common::sample_bytes("warmup/semantic/bus.jpg");
    let image_emb = embed(
        &service,
        "semantic_image_embed",
        TaskRequest::new(image, "image/jpeg"),
    )
    .await;
    assert!(image_emb.dim > 0, "{model}: embedding dim");
    assert_eq!(
        image_emb.vector.len(),
        image_emb.dim,
        "{model}: dim matches vector"
    );
    assert!(
        (l2_norm(&image_emb.vector) - 1.0).abs() < 1e-3,
        "{model}: image embedding should be L2-normalized, got norm {}",
        l2_norm(&image_emb.vector)
    );

    let matching = embed(
        &service,
        "semantic_text_embed",
        TaskRequest::new("a photo of a bus".as_bytes().to_vec(), "text/plain"),
    )
    .await;
    let distractor = embed(
        &service,
        "semantic_text_embed",
        TaskRequest::new("a photo of a kitten".as_bytes().to_vec(), "text/plain"),
    )
    .await;

    assert_eq!(
        matching.vector.len(),
        image_emb.vector.len(),
        "{model}: text/image dims match"
    );
    assert!(
        (l2_norm(&matching.vector) - 1.0).abs() < 1e-3,
        "{model}: text normalized"
    );

    let sim_match = dot(&image_emb.vector, &matching.vector);
    let sim_distractor = dot(&image_emb.vector, &distractor.vector);
    eprintln!("{model} sim: bus={sim_match:.4} kitten={sim_distractor:.4}");
    assert!(
        sim_match > sim_distractor,
        "{model}: matching caption should score higher: bus={sim_match} kitten={sim_distractor}"
    );
}

#[tokio::test]
async fn siglip_base_patch16_224_embeddings_are_aligned() {
    check_siglip_model("siglip2-base-patch16-224").await;
}

#[tokio::test]
async fn siglip_so400m_patch14_384_embeddings_are_aligned() {
    check_siglip_model("siglip2-so400m-patch14-384").await;
}
