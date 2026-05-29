//! End-to-end BioCLIP classification test on the active Burn backend.
//!
//! Run on CPU:   cargo test --test e2e_bioclip
//! Run on Metal: cargo test --features metal --test e2e_bioclip
//!
//! Requires the bioclip-2 vision weights AND a TreeOfLife catalog
//! (`datasets/<DATASET>.{npy,json}`) under LUMEN_MODELS_DIR; skips otherwise.

mod common;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use lumen_hub::backend::default_device;
use lumen_hub::models::bioclip::BioclipService;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::{LabelsV1, ModelConfig, Runtime, ServiceConfig};

const MODEL: &str = "bioclip-2";
const DATASET: &str = "TreeOfLife200MCore";

#[tokio::test]
async fn bioclip_classifies_image_into_taxonomy_labels() {
    let Some((cache_dir, model_name)) = common::require_model(MODEL, &["vision"]) else {
        return;
    };
    let dataset_dir = Path::new(&cache_dir).join(&model_name).join("datasets");
    if !dataset_dir.join(format!("{DATASET}.npy")).is_file()
        || !dataset_dir.join(format!("{DATASET}.json")).is_file()
    {
        eprintln!(
            "SKIP: missing {DATASET} catalog under {}",
            dataset_dir.display()
        );
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
    eprintln!(
        "bioclip top labels: {:?}",
        labels
            .labels
            .iter()
            .map(|l| (&l.label, l.score))
            .collect::<Vec<_>>()
    );

    assert!(!labels.labels.is_empty(), "expected at least one label");
    assert!(labels.labels.len() <= 5, "top_k=5 respected");
    let top = &labels.labels[0];
    assert!(
        top.score > 0.0 && top.score <= 1.0001,
        "softmax score in range"
    );
    // Taxonomy convention: kingdom/phylum/class/order/family/genus/species/common_name
    assert_eq!(
        top.label.split('/').count(),
        8,
        "label `{}` should have 8 taxonomy ranks",
        top.label
    );
    // Scores are sorted descending.
    assert!(
        labels.labels.windows(2).all(|w| w[0].score >= w[1].score),
        "labels must be sorted by descending score"
    );
}
