use std::{collections::BTreeMap, env, io::Cursor, path::PathBuf};

use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use lumen_hub::models::clip::ClipService;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::{EmbeddingV1, ModelConfig, Runtime, ServiceConfig};
use lumnn::core::context::{MLContext, MLContextOptions};

#[tokio::test]
#[ignore = "requires an exported CLIP model directory; defaults to ./out"]
async fn exported_clip_model_serves_text_and_image_embeddings() {
    let model_dir = env::var("LUMEN_CLIP_E2E_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out"));
    assert!(
        model_dir.join("model_info.json").is_file(),
        "missing exported CLIP model at {}; set LUMEN_CLIP_E2E_MODEL_DIR",
        model_dir.display()
    );

    let model_dir = model_dir.canonicalize().expect("model dir canonicalizes");
    let cache_dir = model_dir
        .parent()
        .expect("model dir has a parent")
        .to_string_lossy()
        .into_owned();
    let model_name = model_dir
        .file_name()
        .expect("model dir has a name")
        .to_string_lossy()
        .into_owned();
    let precision = env::var("LUMEN_CLIP_E2E_PRECISION").unwrap_or_else(|_| "fp32".to_owned());

    let accelerated =
        env_bool("LUMEN_CLIP_E2E_ACCELERATED") || env_bool("LUMEN_CLIP_LATENCY_ACCELERATED");
    let context = MLContext::new(MLContextOptions { accelerated }).expect("ML context initializes");
    let service_config = ServiceConfig {
        enabled: true,
        package: "clip".to_owned(),
        models: BTreeMap::from([(
            "clip".to_owned(),
            ModelConfig {
                model: model_name.clone(),
                runtime: Runtime::Onnx,
                rknn_device: None,
                dataset: None,
                precision: Some(precision),
            },
        )]),
    };

    let service = ClipService::from_config("clip", &service_config, &cache_dir, context)
        .expect("CLIP service loads exported model");
    assert_eq!(
        service.tasks().task_names(),
        vec!["clip_image_embed", "clip_text_embed"]
    );

    let text = service
        .tasks()
        .handle(
            "clip_text_embed",
            TaskRequest::new("a photo of a cat", "text/plain"),
        )
        .await
        .expect("text embedding succeeds");
    assert_embedding(&text.payload, &model_name);

    let image = service
        .tasks()
        .handle(
            "clip_image_embed",
            TaskRequest::new(sample_png(), "image/png"),
        )
        .await
        .expect("image embedding succeeds");
    assert_embedding(&image.payload, &model_name);
}

fn env_bool(name: &str) -> bool {
    matches!(
        env::var(name).as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn assert_embedding(payload: &[u8], model_name: &str) {
    let embedding: EmbeddingV1 =
        serde_json::from_slice(payload).expect("embedding payload decodes");
    assert_eq!(embedding.dim, 512);
    assert_eq!(embedding.vector.len(), 512);
    assert_eq!(embedding.model_id, model_name);

    let norm = embedding
        .vector
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected L2-normalized embedding, got norm {norm}"
    );
}

fn sample_png() -> Vec<u8> {
    let image = RgbImage::from_fn(256, 192, |x, y| {
        Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image)
        .write_to(&mut bytes, ImageFormat::Png)
        .expect("sample image encodes as PNG");
    bytes.into_inner()
}
