use std::{collections::BTreeMap, env, io::Cursor, path::PathBuf};

use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use lumen_hub::models::siglip::SiglipService;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::{EmbeddingV1, ModelConfig, Runtime, ServiceConfig};
use lumnn::core::context::{MLContext, MLContextOptions};

#[tokio::test]
#[ignore = "requires an exported SigLIP model directory; defaults to ./out"]
async fn exported_siglip_model_serves_text_and_image_embeddings() {
    let model_dir = env::var("LUMEN_SIGLIP_E2E_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out"));
    assert!(
        model_dir.join("model_info.json").is_file(),
        "missing exported SigLIP model at {}; set LUMEN_SIGLIP_E2E_MODEL_DIR",
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
    let precision = env::var("LUMEN_SIGLIP_E2E_PRECISION").unwrap_or_else(|_| "fp32".to_owned());
    let runtime = siglip_runtime();

    let accelerated = siglip_accelerated(runtime);
    let context = MLContext::new(MLContextOptions { accelerated }).expect("ML context initializes");
    let service_config = ServiceConfig {
        enabled: true,
        package: "siglip".to_owned(),
        models: BTreeMap::from([(
            "siglip".to_owned(),
            ModelConfig {
                model: model_name.clone(),
                runtime,
                rknn_device: None,
                dataset: None,
                precision: Some(precision),
            },
        )]),
    };

    let service = SiglipService::from_config("siglip", &service_config, &cache_dir, context)
        .expect("SigLIP service loads exported model");
    assert_eq!(
        service.tasks().task_names(),
        vec!["siglip_semantic_image_embed", "siglip_semantic_text_embed"]
    );

    let text = service
        .tasks()
        .handle(
            "siglip_semantic_text_embed",
            TaskRequest::new("a photo of a cat", "text/plain"),
        )
        .await
        .expect("text embedding succeeds");
    assert_embedding(&text.payload, &model_name);

    let image = service
        .tasks()
        .handle(
            "siglip_semantic_image_embed",
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

fn siglip_runtime() -> Runtime {
    env::var("LUMEN_SIGLIP_E2E_RUNTIME")
        .or_else(|_| env::var("LUMEN_SIGLIP_RUNTIME"))
        .ok()
        .as_deref()
        .map(parse_runtime)
        .unwrap_or(Runtime::Onnx)
}

fn parse_runtime(value: &str) -> Runtime {
    match value {
        "onnx" => Runtime::Onnx,
        "candle_onnx" | "candle-onnx" | "candle" => Runtime::CandleOnnx,
        other => panic!("unsupported SigLIP runtime `{other}`; expected onnx or candle_onnx"),
    }
}

fn siglip_accelerated(runtime: Runtime) -> bool {
    let requested =
        env_bool("LUMEN_SIGLIP_E2E_ACCELERATED") || env_bool("LUMEN_SIGLIP_LATENCY_ACCELERATED");

    #[cfg(target_vendor = "apple")]
    {
        if runtime == Runtime::CandleOnnx {
            return requested;
        }
        if requested {
            eprintln!("SigLIP accelerated ONNX Runtime is disabled on Apple; using CPU-only");
        }
        false
    }

    #[cfg(not(target_vendor = "apple"))]
    {
        requested
    }
}

fn assert_embedding(payload: &[u8], model_name: &str) {
    let embedding: EmbeddingV1 =
        serde_json::from_slice(payload).expect("embedding payload decodes");
    assert_eq!(embedding.vector.len(), embedding.dim);
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
