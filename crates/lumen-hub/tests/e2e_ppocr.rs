//! End-to-end PP-OCR test on the active Burn backend.
//!
//! Run on CPU:   cargo test --test e2e_ppocr
//! Run on Metal: cargo test --features metal --test e2e_ppocr

mod common;

use std::sync::Arc;

use lumen_hub::backend::default_device;
use lumen_hub::models::ppocr::PpocrService;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::OCRV1;

const MODEL: &str = "pp-ocrv5";

#[tokio::test]
async fn ppocr_detects_and_recognizes_text() {
    let Some((cache_dir, model_name)) = common::require_model(MODEL, &["detection", "recognition"])
    else {
        return;
    };
    let config = common::service_config("ppocr", &model_name);
    let device = Arc::new(default_device());
    let service = PpocrService::from_config("ppocr", &config, &cache_dir, device)
        .expect("PP-OCR service builds from config");

    let task = service
        .tasks()
        .task_names()
        .into_iter()
        .next()
        .expect("PP-OCR exposes a task");

    let image = common::sample_bytes("tests/test_sample/ocr_test_1.jpeg");
    let result = service
        .tasks()
        .handle(&task, TaskRequest::new(image, "image/jpeg"))
        .await
        .expect("OCR task succeeds");

    let ocr: OCRV1 = serde_json::from_slice(&result.payload).expect("ocr_v1 JSON");
    eprintln!(
        "ppocr detected {} text regions: {:?}",
        ocr.count,
        ocr.items.iter().map(|i| &i.text).collect::<Vec<_>>()
    );
    assert!(ocr.count > 0, "expected at least one detected text region");
    assert!(
        ocr.items.iter().any(|item| !item.text.trim().is_empty()),
        "expected at least one non-empty recognized string"
    );
}
