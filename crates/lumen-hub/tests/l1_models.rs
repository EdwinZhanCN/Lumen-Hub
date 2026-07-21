//! L1 model suite: real production weights, semantic quality checks, and
//! golden-embedding regression. Env-gated — every test skips gracefully when
//! `LUMEN_MODELS_DIR` lacks the required weights, and runs nightly in CI.
//!
//! Run: LUMEN_MODELS_DIR=/path/to/lumen-models cargo test --release --test l1_models
//!
//! Golden files live in `tests/golden/`; regenerate with `cargo xtask golden`
//! (sets LUMEN_GOLDEN_WRITE=1 under the hood) and review the diff. Because
//! the same goldens are compared on every backend job, they double as the
//! cpu ↔ metal parity check.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lumen_hub::backend::default_device;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::{FaceV1, LabelsV1, ModelConfig, OCRV1, Runtime, ServiceConfig};

const GOLDEN_COSINE_FLOOR: f32 = 0.999;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

/// Compares (or, with LUMEN_GOLDEN_WRITE=1, records) a golden embedding.
fn golden_check(slot: &str, vector: &[f32]) {
    let path = golden_dir().join(format!("{slot}.json"));
    if std::env::var("LUMEN_GOLDEN_WRITE").as_deref() == Ok("1") {
        std::fs::create_dir_all(golden_dir()).expect("golden dir");
        let body = serde_json::json!({ "vector": vector });
        std::fs::write(&path, serde_json::to_vec_pretty(&body).unwrap()).expect("write golden");
        eprintln!("golden: wrote {}", path.display());
        return;
    }
    let Ok(raw) = std::fs::read(&path) else {
        eprintln!(
            "SKIP golden `{slot}`: {} missing (run `cargo xtask golden`)",
            path.display()
        );
        return;
    };
    let golden: serde_json::Value = serde_json::from_slice(&raw).expect("golden json");
    let reference: Vec<f32> = golden["vector"]
        .as_array()
        .expect("golden vector")
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let similarity = dot(vector, &reference);
    assert!(
        similarity > GOLDEN_COSINE_FLOOR,
        "golden `{slot}` drifted: cosine {similarity:.6} < {GOLDEN_COSINE_FLOOR}"
    );
}

/// Model tests construct huge generated Burn graphs — run each on a dedicated
/// big-stack thread with its own runtime, mirroring the serving setup.
fn run_on_large_stack<F>(future_factory: F)
where
    F: FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()>>> + Send + 'static,
{
    const STACK: usize = 256 * 1024 * 1024;
    std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(move || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_stack_size(STACK)
                .build()
                .expect("tokio runtime")
                .block_on(future_factory());
        })
        .expect("spawn test thread")
        .join()
        .expect("test thread panicked");
}

// ---- SigLIP: alignment semantics + golden regression ----

async fn check_siglip(model: &'static str) {
    use lumen_hub::models::siglip::SiglipService;
    use lumen_schema::EmbeddingV1;

    let Some((cache_dir, model_name)) = common::require_model(model, &["text", "vision"]) else {
        return;
    };
    let config = common::service_config("siglip", &model_name);
    let device = Arc::new(default_device());
    let service = SiglipService::from_config("siglip", &config, &cache_dir, device)
        .expect("SigLIP service builds from config");

    let embed = |task: &'static str, request: TaskRequest| {
        let tasks = service.tasks();
        async move {
            let result = tasks
                .handle(task, request)
                .await
                .unwrap_or_else(|e| panic!("task `{task}` failed: {e}"));
            let embedding: EmbeddingV1 =
                serde_json::from_slice(&result.payload).expect("embedding_v1 JSON");
            embedding
        }
    };

    let image = common::sample_bytes("warmup/semantic/bus.jpg");
    let image_emb = embed(
        "semantic_image_embed",
        TaskRequest::new(image, "image/jpeg"),
    )
    .await;
    assert!(image_emb.dim > 0, "{model}: embedding dim");
    assert_eq!(image_emb.vector.len(), image_emb.dim);
    assert!(
        (l2_norm(&image_emb.vector) - 1.0).abs() < 1e-3,
        "{model}: image embedding normalized"
    );

    let matching = embed(
        "semantic_text_embed",
        TaskRequest::new("a photo of a bus".as_bytes().to_vec(), "text/plain"),
    )
    .await;
    let distractor = embed(
        "semantic_text_embed",
        TaskRequest::new("a photo of a kitten".as_bytes().to_vec(), "text/plain"),
    )
    .await;
    assert_eq!(matching.vector.len(), image_emb.vector.len());
    assert!((l2_norm(&matching.vector) - 1.0).abs() < 1e-3);
    let sim_match = dot(&image_emb.vector, &matching.vector);
    let sim_distractor = dot(&image_emb.vector, &distractor.vector);
    assert!(
        sim_match > sim_distractor,
        "{model}: matching caption must beat the distractor: bus={sim_match} kitten={sim_distractor}"
    );

    golden_check(&format!("{model}.image.bus"), &image_emb.vector);
    golden_check(&format!("{model}.text.bus"), &matching.vector);
}

#[test]
fn siglip_base_patch16_224() {
    run_on_large_stack(|| Box::pin(check_siglip("siglip2-base-patch16-224")));
}

#[test]
fn siglip_so400m_patch14_384() {
    run_on_large_stack(|| Box::pin(check_siglip("siglip2-so400m-patch14-384")));
}

// ---- BioCLIP: taxonomy classification ----

async fn check_bioclip() {
    use lumen_hub::models::bioclip::BioclipService;

    const MODEL: &str = "bioclip-2";
    const DATASET: &str = "TreeOfLife200MCore";

    let Some((cache_dir, model_name)) = common::require_model(MODEL, &["vision"]) else {
        return;
    };
    let dataset_dir = Path::new(&cache_dir).join(&model_name).join("datasets");
    if !dataset_dir.join(format!("{DATASET}.npy")).is_file()
        || !dataset_dir.join(format!("{DATASET}.json")).is_file()
    {
        eprintln!("SKIP: missing {DATASET} catalog");
        return;
    }
    let config = ServiceConfig {
        enabled: true,
        package: "clip".to_owned(),
        models: BTreeMap::from([(
            "default".to_owned(),
            ModelConfig {
                model: model_name.clone(),
                runtime: Runtime::Burn,
                dataset: Some(DATASET.to_owned()),
                precision: Some("fp32".to_owned()),
            },
        )]),
    };
    let device = Arc::new(default_device());
    let service = BioclipService::from_config("bioclip", &config, &cache_dir, device)
        .expect("BioCLIP service builds from config");

    let image = common::sample_bytes("warmup/bio/abyssinian.jpg");
    let result = service
        .tasks()
        .handle(
            "bioclip_classify",
            TaskRequest::new(image, "image/jpeg").with_meta("top_k", "5"),
        )
        .await
        .expect("bioclip_classify succeeds");
    let labels: LabelsV1 = serde_json::from_slice(&result.payload).expect("labels_v1 JSON");
    assert!(!labels.labels.is_empty(), "expected at least one label");
    assert!(labels.labels.len() <= 5, "top_k=5 respected");
    let top = &labels.labels[0];
    assert!(top.score > 0.0 && top.score <= 1.0001, "softmax in range");
    assert_eq!(
        top.label.split('/').count(),
        8,
        "label `{}` should have 8 taxonomy ranks",
        top.label
    );
    assert!(
        labels.labels.windows(2).all(|w| w[0].score >= w[1].score),
        "labels sorted by descending score"
    );
}

#[test]
fn bioclip_classifies_into_taxonomy() {
    run_on_large_stack(|| Box::pin(check_bioclip()));
}

// ---- InsightFace: detection + ArcFace embedding ----

async fn check_insightface() {
    use lumen_hub::models::insightface::InsightFaceService;

    let Some((cache_dir, model_name)) =
        common::require_model("antelopev2", &["detection", "recognition"])
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
        .expect("face recognition succeeds");
    let faces: FaceV1 = serde_json::from_slice(&result.payload).expect("face_v1 JSON");
    assert!(faces.count > 0, "expected at least one detected face");
    let face = &faces.faces[0];
    let embedding = face.embedding.as_ref().expect("face embedding");
    assert_eq!(embedding.len(), 512, "ArcFace embedding dim");
    assert!(
        (l2_norm(embedding) - 1.0).abs() < 1e-3,
        "face embedding normalized"
    );
    assert!(face.confidence > 0.0 && face.confidence <= 1.0);

    golden_check("antelopev2.face.sample1", embedding);
}

#[test]
fn insightface_detects_and_embeds() {
    run_on_large_stack(|| Box::pin(check_insightface()));
}

// ---- PP-OCR family: one parameterized check, three packagings ----

async fn check_ocr(
    model: &'static str,
    precision: &'static str,
    components: &'static [&'static str],
    sample: &'static str,
    mime: &'static str,
) {
    use lumen_hub::models::ppocr::PpocrService;

    let Some((cache_dir, model_name)) =
        common::require_model_precision(model, components, precision)
    else {
        return;
    };
    let config = ServiceConfig {
        enabled: true,
        package: "ppocr".to_owned(),
        models: BTreeMap::from([(
            "default".to_owned(),
            ModelConfig {
                model: model_name,
                runtime: Runtime::Burn,
                dataset: None,
                precision: Some(precision.to_owned()),
            },
        )]),
    };
    let device = Arc::new(default_device());
    let service = PpocrService::from_config("ppocr", &config, &cache_dir, device)
        .unwrap_or_else(|e| panic!("{model} ({precision}) service: {e}"));
    let task = service
        .tasks()
        .task_names()
        .into_iter()
        .next()
        .expect("PP-OCR exposes a task");

    let image = common::sample_bytes(sample);
    let result = service
        .tasks()
        .handle(&task, TaskRequest::new(image, mime))
        .await
        .unwrap_or_else(|e| panic!("{model} OCR failed: {e}"));
    let ocr: OCRV1 = serde_json::from_slice(&result.payload).expect("ocr_v1 JSON");
    eprintln!(
        "{model} ({precision}) detected {} regions: {:?}",
        ocr.count,
        ocr.items.iter().map(|i| &i.text).collect::<Vec<_>>()
    );
    assert!(ocr.count > 0, "{model}: expected at least one text region");
    assert!(
        ocr.items.iter().any(|item| !item.text.trim().is_empty()),
        "{model}: expected a non-empty recognized string"
    );
}

#[test]
fn ppocr_v5_fp32_reads_sample() {
    run_on_large_stack(|| {
        Box::pin(check_ocr(
            "pp-ocrv5",
            "fp32",
            &["detection", "recognition"],
            "tests/test_sample/ocr_test_1.jpeg",
            "image/jpeg",
        ))
    });
}

#[test]
fn ppocr_v6_small_fp16q8_reads_border() {
    run_on_large_stack(|| {
        Box::pin(check_ocr(
            "pp-ocrv6-small",
            "fp16q8",
            &["detection", "recognition", "classification"],
            "warmup/ocr/border.png",
            "image/png",
        ))
    });
}

#[test]
fn ppocr_v5_server_fp16q8_reads_border() {
    run_on_large_stack(|| {
        Box::pin(check_ocr(
            "pp-ocrv5-server",
            "fp16q8",
            &["detection", "recognition", "classification"],
            "warmup/ocr/border.png",
            "image/png",
        ))
    });
}
