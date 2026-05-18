use std::{
    collections::BTreeMap,
    env,
    io::Cursor,
    path::PathBuf,
    time::{Duration, Instant},
};

use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use lumen_hub::models::siglip::SiglipService;
use lumen_hub::service::{InferenceService, TaskRequest};
use lumen_schema::{ModelConfig, Runtime, ServiceConfig};
use lumnn::core::context::{MLContext, MLContextOptions};

#[tokio::test]
#[ignore = "requires an exported SigLIP model directory; defaults to ./out"]
async fn exported_siglip_model_latency() {
    let model_dir = env::var("LUMEN_SIGLIP_E2E_MODEL_DIR")
        .or_else(|_| env::var("LUMEN_SIGLIP_LATENCY_MODEL_DIR"))
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
    let precision = env::var("LUMEN_SIGLIP_E2E_PRECISION")
        .or_else(|_| env::var("LUMEN_SIGLIP_LATENCY_PRECISION"))
        .unwrap_or_else(|_| "fp32".to_owned());
    let warmup = env_usize("LUMEN_SIGLIP_LATENCY_WARMUP", 5);
    let iters = env_usize("LUMEN_SIGLIP_LATENCY_ITERS", 30);
    let runtime = siglip_runtime();

    let accelerated = siglip_accelerated(runtime);
    let context = MLContext::new(MLContextOptions { accelerated }).expect("ML context initializes");
    let service_config = ServiceConfig {
        enabled: true,
        package: "siglip".to_owned(),
        models: BTreeMap::from([(
            "siglip".to_owned(),
            ModelConfig {
                model: model_name,
                runtime,
                rknn_device: None,
                dataset: None,
                precision: Some(precision.clone()),
            },
        )]),
    };

    let load_start = Instant::now();
    let service = SiglipService::from_config("siglip", &service_config, &cache_dir, context)
        .expect("SigLIP service loads exported model");
    let load_elapsed = load_start.elapsed();

    let text_payload = b"a photo of a cat".to_vec();
    let image_payload = sample_png();

    let text = measure_task(
        &service,
        "semantic_text_embed",
        "text/plain",
        &text_payload,
        warmup,
        iters,
    )
    .await;
    let image = measure_task(
        &service,
        "semantic_image_embed",
        "image/png",
        &image_payload,
        warmup,
        iters,
    )
    .await;

    println!("model_dir={}", model_dir.display());
    println!("runtime={}", runtime.as_str());
    println!("precision={precision}");
    println!("accelerated={accelerated}");
    println!("warmup={warmup}");
    println!("iters={iters}");
    println!("service_load_ms={:.3}", ms(load_elapsed));
    println!("semantic_text_embed {}", text);
    println!("semantic_image_embed {}", image);
}

async fn measure_task(
    service: &SiglipService,
    task_name: &str,
    mime: &str,
    payload: &[u8],
    warmup: usize,
    iters: usize,
) -> LatencyStats {
    for _ in 0..warmup {
        service
            .tasks()
            .handle(task_name, TaskRequest::new(payload.to_vec(), mime))
            .await
            .expect("warmup inference succeeds");
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        service
            .tasks()
            .handle(task_name, TaskRequest::new(payload.to_vec(), mime))
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

fn siglip_runtime() -> Runtime {
    env::var("LUMEN_SIGLIP_LATENCY_RUNTIME")
        .or_else(|_| env::var("LUMEN_SIGLIP_E2E_RUNTIME"))
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
        "mnn" => Runtime::Mnn,
        other => panic!("unsupported SigLIP runtime `{other}`; expected onnx, candle_onnx, or mnn"),
    }
}

fn siglip_accelerated(runtime: Runtime) -> bool {
    let requested =
        env_bool("LUMEN_SIGLIP_LATENCY_ACCELERATED") || env_bool("LUMEN_SIGLIP_E2E_ACCELERATED");

    #[cfg(target_vendor = "apple")]
    {
        if runtime == Runtime::CandleOnnx || runtime == Runtime::Mnn {
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
