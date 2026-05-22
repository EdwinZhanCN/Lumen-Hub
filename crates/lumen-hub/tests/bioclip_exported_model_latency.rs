use std::{
    collections::BTreeMap,
    env, fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use lumen_hub::models::clip::ClipService;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::{LabelsV1, ModelConfig, Runtime, ServiceConfig};
use lumnn::core::context::{MLContext, MLContextOptions};

#[tokio::test]
#[ignore = "requires exported BioCLIP assets; defaults to ./out/bioclip-2"]
async fn exported_bioclip_model_latency() {
    let model_dir = env::var("LUMEN_BIOCLIP_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out/bioclip-2"));
    assert!(
        model_dir.join("model_info.json").is_file(),
        "missing exported BioCLIP model at {}; set LUMEN_BIOCLIP_MODEL_DIR",
        model_dir.display()
    );

    let dataset =
        env::var("LUMEN_BIOCLIP_DATASET").unwrap_or_else(|_| "TreeOfLife200MCore".to_owned());
    assert!(
        model_dir.join(format!("{dataset}.npy")).is_file(),
        "missing {dataset}.npy at {}",
        model_dir.display()
    );
    assert!(
        model_dir.join(format!("{dataset}.bin")).is_file(),
        "missing {dataset}.bin at {}",
        model_dir.display()
    );

    let sample_path = env::var("LUMEN_BIOCLIP_SAMPLE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_sample/bioclip_test_1.png")
        });
    let image_payload = fs::read(&sample_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", sample_path.display()));

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
    let precision = env::var("LUMEN_BIOCLIP_PRECISION").unwrap_or_else(|_| "fp16".to_owned());
    let runtime = env::var("LUMEN_BIOCLIP_RUNTIME").unwrap_or_else(|_| "mnn".to_owned());
    let runtime = match runtime.as_str() {
        "mnn" => Runtime::Mnn,
        "onnx" => Runtime::Onnx,
        other => panic!("unsupported LUMEN_BIOCLIP_RUNTIME `{other}`"),
    };
    let top_k = env::var("LUMEN_BIOCLIP_TOP_K").unwrap_or_else(|_| "5".to_owned());
    let warmup = env_usize("LUMEN_BIOCLIP_LATENCY_WARMUP", 1);
    let iters = env_usize("LUMEN_BIOCLIP_LATENCY_ITERS", 5);
    let accelerated = env_bool("LUMEN_BIOCLIP_ACCELERATED") || runtime == Runtime::Mnn;

    let context = MLContext::new(MLContextOptions { accelerated }).expect("ML context initializes");
    let service_config = ServiceConfig {
        enabled: true,
        package: "clip".to_owned(),
        models: BTreeMap::from([(
            "bioclip".to_owned(),
            ModelConfig {
                model: model_name.clone(),
                runtime,
                dataset: Some(dataset.clone()),
                precision: Some(precision.clone()),
            },
        )]),
    };

    let load_start = Instant::now();
    let service = ClipService::from_config("clip", &service_config, &cache_dir, context)
        .expect("BioCLIP service loads exported model");
    let load_elapsed = load_start.elapsed();

    let mut last = None;
    for _ in 0..warmup {
        last = Some(
            service
                .tasks()
                .handle(
                    "bioclip_classify",
                    TaskRequest::new(image_payload.clone(), "image/png").with_meta("TopK", &top_k),
                )
                .await
                .expect("warmup inference succeeds"),
        );
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        last = Some(
            service
                .tasks()
                .handle(
                    "bioclip_classify",
                    TaskRequest::new(image_payload.clone(), "image/png").with_meta("TopK", &top_k),
                )
                .await
                .expect("inference succeeds"),
        );
        samples.push(start.elapsed());
    }

    let labels: LabelsV1 = serde_json::from_slice(
        &last
            .expect("latency test runs at least one inference")
            .payload,
    )
    .expect("labels_v1 response decodes");
    assert_eq!(labels.labels.len(), top_k.parse::<usize>().unwrap());
    assert!(labels.labels.iter().all(|label| !label.label.is_empty()));

    println!("model_dir={}", model_dir.display());
    println!("dataset={dataset}");
    println!("runtime={}", runtime.as_str());
    println!("precision={precision}");
    println!("accelerated={accelerated}");
    println!("sample={}", sample_path.display());
    println!("top_k={top_k}");
    println!("warmup={warmup}");
    println!("iters={iters}");
    println!("service_load_ms={:.3}", ms(load_elapsed));
    println!("bioclip_classify {}", LatencyStats::from_samples(samples));
    for (index, label) in labels.labels.iter().enumerate() {
        println!("{index}: score={:.6} label={}", label.score, label.label);
    }
}

#[derive(Debug)]
struct LatencyStats {
    min: Duration,
    p50: Duration,
    p95: Duration,
    mean: Duration,
    max: Duration,
}

impl LatencyStats {
    fn from_samples(mut samples: Vec<Duration>) -> Self {
        assert!(
            !samples.is_empty(),
            "latency test requires at least one sample"
        );
        samples.sort_unstable();
        let total: Duration = samples.iter().copied().sum();
        let mean = total / samples.len() as u32;
        Self {
            min: samples[0],
            p50: percentile(&samples, 0.50),
            p95: percentile(&samples, 0.95),
            mean,
            max: *samples.last().expect("samples is non-empty"),
        }
    }
}

impl std::fmt::Display for LatencyStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "min_ms={:.3} p50_ms={:.3} p95_ms={:.3} mean_ms={:.3} max_ms={:.3}",
            ms(self.min),
            ms(self.p50),
            ms(self.p95),
            ms(self.mean),
            ms(self.max)
        )
    }
}

fn percentile(samples: &[Duration], quantile: f64) -> Duration {
    let index = ((samples.len() - 1) as f64 * quantile).ceil() as usize;
    samples[index]
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn env_bool(name: &str) -> bool {
    matches!(
        env::var(name).as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}
