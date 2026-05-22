#![cfg(feature = "insightface")]

use std::{
    collections::BTreeMap,
    env, fs,
    io::Cursor,
    path::PathBuf,
    time::{Duration, Instant},
};

use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use lumen_hub::models::insightface::InsightFaceService;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::{FaceV1, ModelConfig, Runtime, ServiceConfig};
use lumnn::core::context::{MLContext, MLContextOptions};

const INSIGHTFACE_TASK_NAME: &str = "face_recognition";

#[tokio::test]
#[ignore = "requires an exported InsightFace model directory; defaults to ./out/antelopev2"]
async fn exported_insightface_model_latency() {
    let model_dir = env::var("LUMEN_INSIGHTFACE_LATENCY_MODEL_DIR")
        .or_else(|_| env::var("LUMEN_INSIGHTFACE_E2E_MODEL_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out/antelopev2"));
    assert!(
        model_dir.join("model_info.json").is_file(),
        "missing exported InsightFace model at {}; set LUMEN_INSIGHTFACE_LATENCY_MODEL_DIR",
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
    let precision = env::var("LUMEN_INSIGHTFACE_LATENCY_PRECISION")
        .or_else(|_| env::var("LUMEN_INSIGHTFACE_E2E_PRECISION"))
        .unwrap_or_else(|_| "fp32".to_owned());
    let warmup = env_usize("LUMEN_INSIGHTFACE_LATENCY_WARMUP", 3);
    let iters = env_usize("LUMEN_INSIGHTFACE_LATENCY_ITERS", 10);
    let runtime = insightface_runtime();
    let accelerated = env_bool("LUMEN_INSIGHTFACE_LATENCY_ACCELERATED")
        || env_bool("LUMEN_INSIGHTFACE_E2E_ACCELERATED");

    let context = MLContext::new(MLContextOptions { accelerated }).expect("ML context initializes");
    let service_config = ServiceConfig {
        enabled: true,
        package: "insightface".to_owned(),
        models: BTreeMap::from([(
            "insightface".to_owned(),
            ModelConfig {
                model: model_name.clone(),
                runtime,
                dataset: None,
                precision: Some(precision.clone()),
            },
        )]),
    };

    let load_start = Instant::now();
    let service =
        InsightFaceService::from_config("insightface", &service_config, &cache_dir, context)
            .expect("InsightFace service loads exported model");
    let load_elapsed = load_start.elapsed();

    assert_eq!(service.tasks().task_names(), vec![INSIGHTFACE_TASK_NAME]);

    let sample = sample_image();
    let baseline = run_task(&service, &sample.mime, &sample.payload)
        .await
        .expect("baseline face_recognition inference succeeds");
    let baseline_result: FaceV1 =
        serde_json::from_slice(&baseline.payload).expect("payload decodes as FaceV1");

    let stats = measure_task(&service, &sample.mime, &sample.payload, warmup, iters).await;

    println!("model_dir={}", model_dir.display());
    println!("runtime={}", runtime.as_str());
    println!("precision={precision}");
    println!("accelerated={accelerated}");
    println!("warmup={warmup}");
    println!("iters={iters}");
    println!("service_load_ms={:.3}", ms(load_elapsed));
    println!("sample_mime={}", sample.mime);
    println!("faces={}", baseline_result.count);
    println!("face_recognition {}", stats);
}

async fn run_task(
    service: &InsightFaceService,
    mime: &str,
    payload: &[u8],
) -> Result<lumen_hub::service::TaskResult, lumen_hub::service::ServiceError> {
    service
        .tasks()
        .handle(
            INSIGHTFACE_TASK_NAME,
            TaskRequest::new(payload.to_vec(), mime),
        )
        .await
}

async fn measure_task(
    service: &InsightFaceService,
    mime: &str,
    payload: &[u8],
    warmup: usize,
    iters: usize,
) -> LatencyStats {
    for _ in 0..warmup {
        run_task(service, mime, payload)
            .await
            .expect("warmup inference succeeds");
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        run_task(service, mime, payload)
            .await
            .expect("inference succeeds");
        samples.push(start.elapsed());
    }

    LatencyStats::from_samples(samples)
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

fn insightface_runtime() -> Runtime {
    env::var("LUMEN_INSIGHTFACE_LATENCY_RUNTIME")
        .or_else(|_| env::var("LUMEN_INSIGHTFACE_E2E_RUNTIME"))
        .or_else(|_| env::var("LUMEN_INSIGHTFACE_RUNTIME"))
        .ok()
        .as_deref()
        .map(parse_runtime)
        .unwrap_or(Runtime::Onnx)
}

fn parse_runtime(value: &str) -> Runtime {
    match value {
        "onnx" => Runtime::Onnx,
        "mnn" => Runtime::Mnn,
        other => panic!("unsupported InsightFace runtime `{other}`; expected onnx or mnn"),
    }
}

struct SampleImage {
    mime: String,
    payload: Vec<u8>,
}

fn sample_image() -> SampleImage {
    if let Some(path) = env::var("LUMEN_INSIGHTFACE_LATENCY_IMAGE")
        .ok()
        .or_else(|| env::var("LUMEN_INSIGHTFACE_E2E_IMAGE").ok())
        .map(PathBuf::from)
    {
        let payload = fs::read(&path)
            .unwrap_or_else(|err| panic!("failed to read sample image {}: {err}", path.display()));
        return SampleImage {
            mime: mime_from_path(&path).to_owned(),
            payload,
        };
    }

    let image = RgbImage::from_fn(320, 320, |x, y| {
        let dx = x as i32 - 160;
        let dy = y as i32 - 150;
        let skin = dx * dx / 90 + dy * dy / 120 < 900;
        if skin {
            Rgb([214, 172, 142])
        } else {
            Rgb([32, 48, 72])
        }
    });
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image)
        .write_to(&mut bytes, ImageFormat::Png)
        .expect("sample image encodes as PNG");
    SampleImage {
        mime: "image/png".to_owned(),
        payload: bytes.into_inner(),
    }
}

fn mime_from_path(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("avif") => "image/avif",
        other => panic!(
            "unsupported sample image extension {:?}; expected jpg/jpeg/png/webp/avif",
            other
        ),
    }
}
