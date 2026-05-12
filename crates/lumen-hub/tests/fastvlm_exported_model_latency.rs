#![cfg(feature = "fastvlm")]

use std::{
    collections::BTreeMap,
    env,
    path::PathBuf,
    time::{Duration, Instant},
};

use lumen_hub::{
    models::fastvlm::{FastVlmService, task::META_PROMPT},
    service::{
        DEFAULT_TENSOR_MIME, INPUT_KIND_TENSOR, InferenceService, META_INPUT_KIND,
        META_PREPROCESS_ID, META_PREPROCESS_SKIP, META_TENSOR_BYTE_ORDER, META_TENSOR_DTYPE,
        META_TENSOR_FORMAT, META_TENSOR_LAYOUT, META_TENSOR_SHAPE, TENSOR_BYTE_ORDER_LITTLE,
        TENSOR_FORMAT_CONTIGUOUS, TaskRequest,
    },
};
use lumen_schema::{ModelConfig, Runtime, ServiceConfig};
use lumnn::core::context::{MLContext, MLContextOptions};

const FASTVLM_TASK_ALIAS: &str = "fastvlm";
const FASTVLM_TASK_NAME: &str = "fastvlm_vlm_embeds";
const FASTVLM_PREPROCESS_ID: &str = "fastvlm_image_preprocess_v1";
const INPUT_LAYOUT: &str = "NCHW";
const INPUT_SHAPE: [usize; 4] = [1, 3, 448, 448];

#[tokio::test]
#[ignore = "requires an exported FastVLM model directory; defaults to ./out/fast-vlm-0.5b"]
async fn exported_fastvlm_model_latency() {
    let model_dir = env::var("LUMEN_FASTVLM_E2E_MODEL_DIR")
        .or_else(|_| env::var("LUMEN_FASTVLM_LATENCY_MODEL_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out/fast-vlm-0.5b")
        });
    assert!(
        model_dir.join("model_info.json").is_file(),
        "missing exported FastVLM model at {}; set LUMEN_FASTVLM_LATENCY_MODEL_DIR",
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
    let precision = env::var("LUMEN_FASTVLM_E2E_PRECISION")
        .or_else(|_| env::var("LUMEN_FASTVLM_LATENCY_PRECISION"))
        .unwrap_or_else(|_| "fp16".to_owned());
    let runtime = fastvlm_runtime();
    let accelerated = fastvlm_accelerated(runtime);
    let warmup = env_usize("LUMEN_FASTVLM_LATENCY_WARMUP", 3);
    let iters = env_usize("LUMEN_FASTVLM_LATENCY_ITERS", 10);

    let context = MLContext::new(MLContextOptions { accelerated }).expect("ML context initializes");
    let service_config = ServiceConfig {
        enabled: true,
        package: "fastvlm".to_owned(),
        models: BTreeMap::from([(
            FASTVLM_TASK_ALIAS.to_owned(),
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
    let service = FastVlmService::from_config("inference", &service_config, &cache_dir, context)
        .expect("FastVLM service loads exported model");
    let load_elapsed = load_start.elapsed();

    let prompt = env::var("LUMEN_FASTVLM_LATENCY_PROMPT")
        .unwrap_or_else(|_| "Describe this image".to_owned());
    let request_dtype = discover_request_dtype(&service, &prompt);
    let payload = fastvlm_tensor_payload(request_dtype);

    let stats = measure_task(&service, &prompt, &payload, request_dtype, warmup, iters).await;

    println!("model_dir={}", model_dir.display());
    println!("runtime={}", runtime.as_str());
    println!("precision={precision}");
    println!("request_dtype={request_dtype}");
    println!("accelerated={accelerated}");
    println!("prompt={prompt}");
    println!("warmup={warmup}");
    println!("iters={iters}");
    println!("service_load_ms={:.3}", ms(load_elapsed));
    println!("vlm_embeds {}", stats);
}

async fn measure_task(
    service: &FastVlmService,
    prompt: &str,
    payload: &[u8],
    request_dtype: &str,
    warmup: usize,
    iters: usize,
) -> LatencyStats {
    for _ in 0..warmup {
        service
            .tasks()
            .handle(
                FASTVLM_TASK_NAME,
                tensor_request(prompt, payload.to_vec(), request_dtype),
            )
            .await
            .expect("warmup inference succeeds");
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        service
            .tasks()
            .handle(
                FASTVLM_TASK_NAME,
                tensor_request(prompt, payload.to_vec(), request_dtype),
            )
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

fn tensor_request(prompt: &str, payload: Vec<u8>, dtype: &str) -> TaskRequest {
    TaskRequest::new(payload, DEFAULT_TENSOR_MIME)
        .with_meta(META_INPUT_KIND, INPUT_KIND_TENSOR)
        .with_meta(META_TENSOR_DTYPE, dtype)
        .with_meta(
            META_TENSOR_SHAPE,
            serde_json::to_string(&INPUT_SHAPE).unwrap(),
        )
        .with_meta(META_TENSOR_LAYOUT, INPUT_LAYOUT)
        .with_meta(META_TENSOR_FORMAT, TENSOR_FORMAT_CONTIGUOUS)
        .with_meta(META_TENSOR_BYTE_ORDER, TENSOR_BYTE_ORDER_LITTLE)
        .with_meta(META_PREPROCESS_ID, FASTVLM_PREPROCESS_ID)
        .with_meta(META_PREPROCESS_SKIP, "true")
        .with_meta(META_PROMPT, prompt)
}

fn fastvlm_tensor_payload(dtype: &str) -> Vec<u8> {
    let element_count = INPUT_SHAPE.iter().product::<usize>();
    match dtype {
        "fp16" => vec![0_u8; element_count * 2],
        "fp32" => vec![0_u8; element_count * 4],
        other => panic!("unsupported FastVLM input dtype `{other}`; expected fp16 or fp32"),
    }
}

fn discover_request_dtype(service: &FastVlmService, prompt: &str) -> &'static str {
    for dtype in ["fp16", "fp32"] {
        let request = tensor_request(prompt, fastvlm_tensor_payload(dtype), dtype);
        if service
            .tasks()
            .batch_key(FASTVLM_TASK_NAME, &request)
            .is_ok()
        {
            return dtype;
        }
    }
    panic!("FastVLM task did not accept fp16 or fp32 tensor input");
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

fn fastvlm_runtime() -> Runtime {
    env::var("LUMEN_FASTVLM_LATENCY_RUNTIME")
        .or_else(|_| env::var("LUMEN_FASTVLM_E2E_RUNTIME"))
        .or_else(|_| env::var("LUMEN_FASTVLM_RUNTIME"))
        .ok()
        .as_deref()
        .map(parse_runtime)
        .unwrap_or(Runtime::Onnx)
}

fn parse_runtime(value: &str) -> Runtime {
    match value {
        "onnx" => Runtime::Onnx,
        "candle_onnx" | "candle-onnx" | "candle" => Runtime::CandleOnnx,
        other => panic!("unsupported FastVLM runtime `{other}`; expected onnx or candle_onnx"),
    }
}

fn fastvlm_accelerated(_runtime: Runtime) -> bool {
    #[cfg(target_vendor = "apple")]
    {
        // Default to CoreML acceleration on Apple platforms
        if env_bool("LUMEN_FASTVLM_LATENCY_NO_ACCELERATED")
            || env_bool("LUMEN_FASTVLM_E2E_NO_ACCELERATED")
        {
            return false;
        }
        true
    }

    #[cfg(not(target_vendor = "apple"))]
    {
        env_bool("LUMEN_FASTVLM_LATENCY_ACCELERATED") || env_bool("LUMEN_FASTVLM_E2E_ACCELERATED")
    }
}
