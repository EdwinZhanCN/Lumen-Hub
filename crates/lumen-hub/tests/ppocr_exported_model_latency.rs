#![cfg(feature = "ppocr")]

use std::{
    collections::BTreeMap,
    env, fs,
    io::Cursor,
    path::PathBuf,
    time::{Duration, Instant},
};

use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use lumen_hub::models::ppocr::PpocrService;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::{ModelConfig, OCRV1, Runtime, ServiceConfig};
use lumnn::core::context::{MLContext, MLContextOptions};

const PPOCR_TASK_ALIAS: &str = "ppocr";
const PPOCR_TASK_NAME: &str = "ppocr_ocr";

#[tokio::test]
#[ignore = "requires an exported PP-OCR model directory; defaults to ./out/pp-ocrv5"]
async fn exported_ppocr_model_latency() {
    let model_dir = env::var("LUMEN_PPOCR_E2E_MODEL_DIR")
        .or_else(|_| env::var("LUMEN_PPOCR_LATENCY_MODEL_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out/pp-ocrv5"));
    assert!(
        model_dir.join("model_info.json").is_file(),
        "missing exported PP-OCR model at {}; set LUMEN_PPOCR_LATENCY_MODEL_DIR",
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
    let precision = env::var("LUMEN_PPOCR_E2E_PRECISION")
        .or_else(|_| env::var("LUMEN_PPOCR_LATENCY_PRECISION"))
        .unwrap_or_else(|_| "fp32".to_owned());
    let warmup = env_usize("LUMEN_PPOCR_LATENCY_WARMUP", 3);
    let iters = env_usize("LUMEN_PPOCR_LATENCY_ITERS", 10);
    let accelerated =
        env_bool("LUMEN_PPOCR_LATENCY_ACCELERATED") || env_bool("LUMEN_PPOCR_E2E_ACCELERATED");

    let runtime = env::var("LUMEN_PPOCR_LATENCY_RUNTIME")
        .or_else(|_| env::var("LUMEN_PPOCR_RUNTIME"))
        .ok()
        .map(|v| match v.as_str() {
            "onnx" => Runtime::Onnx,
            "mnn" => Runtime::Mnn,
            other => panic!("unsupported runtime `{other}`; expected onnx or mnn"),
        })
        .unwrap_or(Runtime::Onnx);

    let context = MLContext::new(MLContextOptions { accelerated }).expect("ML context initializes");
    let service_config = ServiceConfig {
        enabled: true,
        package: "ppocr".to_owned(),
        models: BTreeMap::from([(
            PPOCR_TASK_ALIAS.to_owned(),
            ModelConfig {
                model: model_name.clone(),
                runtime,
                rknn_device: None,
                dataset: None,
                precision: Some(precision.clone()),
            },
        )]),
    };

    let load_start = Instant::now();
    let service = PpocrService::from_config("ppocr", &service_config, &cache_dir, context)
        .expect("PP-OCR service loads exported model");
    let load_elapsed = load_start.elapsed();

    assert_eq!(service.tasks().task_names(), vec![PPOCR_TASK_NAME]);

    let sample = sample_image();
    let baseline = run_task(&service, &sample.mime, &sample.payload)
        .await
        .expect("baseline OCR inference succeeds");
    let baseline_result: OCRV1 =
        serde_json::from_slice(&baseline.payload).expect("OCR payload decodes as OCRV1");

    let stats = measure_task(&service, &sample.mime, &sample.payload, warmup, iters).await;

    println!("model_dir={}", model_dir.display());
    println!("precision={precision}");
    println!("accelerated={accelerated}");
    println!("warmup={warmup}");
    println!("iters={iters}");
    println!("service_load_ms={:.3}", ms(load_elapsed));
    println!("sample_mime={}", sample.mime);
    println!("ocr_items={}", baseline_result.count);
    println!("ocr {}", stats);
}

async fn run_task(
    service: &PpocrService,
    mime: &str,
    payload: &[u8],
) -> Result<lumen_hub::service::TaskResult, lumen_hub::service::ServiceError> {
    service
        .tasks()
        .handle(PPOCR_TASK_NAME, TaskRequest::new(payload.to_vec(), mime))
        .await
}

async fn measure_task(
    service: &PpocrService,
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

struct SampleImage {
    mime: String,
    payload: Vec<u8>,
}

fn sample_image() -> SampleImage {
    if let Some(path) = env::var("LUMEN_PPOCR_LATENCY_IMAGE")
        .ok()
        .or_else(|| env::var("LUMEN_PPOCR_E2E_IMAGE").ok())
        .map(PathBuf::from)
    {
        let payload = fs::read(&path)
            .unwrap_or_else(|err| panic!("failed to read sample image {}: {err}", path.display()));
        return SampleImage {
            mime: mime_from_path(&path).to_owned(),
            payload,
        };
    }

    let image = RgbImage::from_fn(512, 192, |x, y| {
        let band = if (x / 32) % 2 == 0 { 224 } else { 32 };
        Rgb([band, (y % 256) as u8, ((x + y) % 256) as u8])
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
