//! End-to-end PP-OCRv6-small test through the full runtime service path,
//! loading fp16q8 weights (detection + text-line-orientation classification +
//! recognition) and reading the "BORDER" warmup crop.
//!
//! Run on CPU:   cargo test --test e2e_ppocr_v6_small
//! Run on Metal: cargo test --features metal --test e2e_ppocr_v6_small

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use lumen_hub::backend::default_device;
use lumen_hub::models::ppocr::PpocrService;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::{ModelConfig, OCRV1, Runtime, ServiceConfig};

const MODEL: &str = "pp-ocrv6-small";

// The generated v6-small modules are also large enough to need an expanded stack
// on GPU backends.
#[test]
fn ppocr_v6_small_fp16q8_reads_border_test() {
    const STACK: usize = 256 * 1024 * 1024;
    std::thread::Builder::new()
        .stack_size(STACK)
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .thread_stack_size(STACK)
                .build()
                .expect("tokio runtime")
                .block_on(ppocr_v6_small_fp16q8_reads_border());
        })
        .expect("spawn test thread")
        .join()
        .expect("test thread panicked");
}

async fn ppocr_v6_small_fp16q8_reads_border() {
    let Some((cache_dir, model_name)) = common::require_model_precision(
        MODEL,
        &["detection", "recognition", "classification"],
        "fp16q8",
    ) else {
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
                precision: Some("fp16q8".to_owned()),
            },
        )]),
    };

    let device = Arc::new(default_device());
    let service = PpocrService::from_config("ppocr", &config, &cache_dir, device)
        .expect("PP-OCRv6-small service builds from fp16q8 config");

    let task = service
        .tasks()
        .task_names()
        .into_iter()
        .next()
        .expect("PP-OCR exposes a task");

    let image = common::sample_bytes("warmup/ocr/border.png");
    let result = service
        .tasks()
        .handle(&task, TaskRequest::new(image, "image/png"))
        .await
        .expect("OCR task succeeds");

    let ocr: OCRV1 = serde_json::from_slice(&result.payload).expect("ocr_v1 JSON");
    let texts: Vec<&String> = ocr.items.iter().map(|i| &i.text).collect();
    eprintln!("v6-small fp16q8 OCR: count={} texts={texts:?}", ocr.count);

    assert!(ocr.count > 0, "expected at least one detected text region");
    let joined: String = ocr.items.iter().map(|i| i.text.as_str()).collect();
    assert!(
        joined.to_uppercase().contains("BORD"),
        "expected BORDER-like text from fp16q8 v6-small pipeline, got {joined:?}"
    );
}
