#![cfg(feature = "fastvlm")]

use std::{
    collections::BTreeMap,
    env,
    path::PathBuf,
    time::{Duration, Instant},
};

use lumen_hub::{
    models::fastvlm::{
        FastVlmService,
        task::{META_MAX_TOKENS, META_PROMPT},
    },
    service::{
        DEFAULT_TENSOR_MIME, INPUT_KIND_TENSOR, InferenceService, META_INPUT_KIND,
        META_OUTPUT_TENSOR_BYTE_ORDER, META_OUTPUT_TENSOR_DTYPE, META_OUTPUT_TENSOR_FORMAT,
        META_OUTPUT_TENSOR_LAYOUT, META_OUTPUT_TENSOR_SHAPE, META_PREPROCESS_ID,
        META_PREPROCESS_SKIP, META_TENSOR_BYTE_ORDER, META_TENSOR_DTYPE, META_TENSOR_FORMAT,
        META_TENSOR_LAYOUT, META_TENSOR_SHAPE, TENSOR_BYTE_ORDER_LITTLE, TENSOR_FORMAT_CONTIGUOUS,
        TaskRequest, TaskResult,
    },
};
use lumen_schema::{
    ModelConfig, Runtime, ServiceConfig, TextGenerationV1,
    mime::{TEXT_GENERATION_V1_JSON, TEXT_GENERATION_V1_SCHEMA},
};
use lumnn::core::context::{MLContext, MLContextOptions};

const FASTVLM_TASK_ALIAS: &str = "fastvlm";
const FASTVLM_EMBEDS_TASK_NAME: &str = "fastvlm_vlm_embeds";
const FASTVLM_DECODE_TASK_NAME: &str = "fastvlm_vlm_decode";
const FASTVLM_PREPROCESS_ID: &str = "fastvlm_image_preprocess_v1";
const INPUT_LAYOUT: &str = "NCHW";
const INPUT_SHAPE: [usize; 4] = [1, 3, 448, 448];

#[tokio::test]
#[ignore = "requires an exported FastVLM model directory; defaults to ./out/fast-vlm-0.5b"]
async fn exported_fastvlm_decode_latency() {
    let model_dir = env::var("LUMEN_FASTVLM_DECODE_LATENCY_MODEL_DIR")
        .or_else(|_| env::var("LUMEN_FASTVLM_E2E_MODEL_DIR"))
        .or_else(|_| env::var("LUMEN_FASTVLM_LATENCY_MODEL_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../out/fast-vlm-0.5b")
        });
    assert!(
        model_dir.join("model_info.json").is_file(),
        "missing exported FastVLM model at {}; set LUMEN_FASTVLM_DECODE_LATENCY_MODEL_DIR",
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
    let precision = env::var("LUMEN_FASTVLM_DECODE_LATENCY_PRECISION")
        .or_else(|_| env::var("LUMEN_FASTVLM_E2E_PRECISION"))
        .or_else(|_| env::var("LUMEN_FASTVLM_LATENCY_PRECISION"))
        .unwrap_or_else(|_| "fp16".to_owned());
    let runtime = fastvlm_runtime();
    let accelerated = fastvlm_accelerated(runtime);
    let warmup = env_usize("LUMEN_FASTVLM_DECODE_LATENCY_WARMUP", 3);
    let iters = env_usize("LUMEN_FASTVLM_DECODE_LATENCY_ITERS", 10);
    let max_tokens = env_usize("LUMEN_FASTVLM_DECODE_LATENCY_MAX_TOKENS", 16);

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

    let prompt = env::var("LUMEN_FASTVLM_DECODE_LATENCY_PROMPT")
        .or_else(|_| env::var("LUMEN_FASTVLM_LATENCY_PROMPT"))
        .unwrap_or_else(|_| "Describe this image in one short phrase.".to_owned());
    let request_dtype = discover_request_dtype(&service, &prompt);
    let embeds = service
        .tasks()
        .handle(
            FASTVLM_EMBEDS_TASK_NAME,
            tensor_request(
                &prompt,
                fastvlm_tensor_payload(request_dtype),
                request_dtype,
            ),
        )
        .await
        .expect("vlm_embeds inference succeeds");
    let decode_request = decode_request_from_embeds(embeds, max_tokens);

    let stats = measure_decode_task(&service, decode_request.clone(), warmup, iters).await;
    let final_result = service
        .tasks()
        .handle(FASTVLM_DECODE_TASK_NAME, decode_request)
        .await
        .expect("decode inference succeeds");
    let response = parse_text_generation(&final_result, max_tokens);
    assert_eq!(response.model_id, model_name);
    assert!(response.generated_tokens <= max_tokens);

    println!("model_dir={}", model_dir.display());
    println!("runtime={}", runtime.as_str());
    println!("precision={precision}");
    println!("request_dtype={request_dtype}");
    println!("accelerated={accelerated}");
    println!("prompt={prompt}");
    println!("max_tokens={max_tokens}");
    println!("warmup={warmup}");
    println!("iters={iters}");
    println!("service_load_ms={:.3}", ms(load_elapsed));
    println!("vlm_decode {}", stats);
    println!("decode_finish_reason={:?}", response.finish_reason);
    println!("decode_generated_tokens={}", response.generated_tokens);
    println!(
        "decode_input_tokens={}",
        response.input_tokens.unwrap_or_default()
    );
    println!("decode_text={}", response.text.replace('\n', "\\n"));
}

async fn measure_decode_task(
    service: &FastVlmService,
    request: TaskRequest,
    warmup: usize,
    iters: usize,
) -> LatencyStats {
    for _ in 0..warmup {
        service
            .tasks()
            .handle(FASTVLM_DECODE_TASK_NAME, request.clone())
            .await
            .expect("warmup decode succeeds");
    }

    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        service
            .tasks()
            .handle(FASTVLM_DECODE_TASK_NAME, request.clone())
            .await
            .expect("decode succeeds");
        samples.push(start.elapsed());
    }

    LatencyStats::from_samples(samples)
}

fn decode_request_from_embeds(result: TaskResult, max_tokens: usize) -> TaskRequest {
    assert_eq!(result.payload_mime, DEFAULT_TENSOR_MIME);
    let dtype = required_meta(&result, META_OUTPUT_TENSOR_DTYPE);
    let shape = required_meta(&result, META_OUTPUT_TENSOR_SHAPE);
    let layout = required_meta(&result, META_OUTPUT_TENSOR_LAYOUT);
    let format = required_meta(&result, META_OUTPUT_TENSOR_FORMAT);
    let byte_order = required_meta(&result, META_OUTPUT_TENSOR_BYTE_ORDER);
    let preprocess_id = required_meta(&result, META_PREPROCESS_ID);

    TaskRequest::new(result.payload, DEFAULT_TENSOR_MIME)
        .with_meta(META_INPUT_KIND, INPUT_KIND_TENSOR)
        .with_meta(META_TENSOR_DTYPE, dtype)
        .with_meta(META_TENSOR_SHAPE, shape)
        .with_meta(META_TENSOR_LAYOUT, layout)
        .with_meta(META_TENSOR_FORMAT, format)
        .with_meta(META_TENSOR_BYTE_ORDER, byte_order)
        .with_meta(META_PREPROCESS_ID, preprocess_id)
        .with_meta(META_PREPROCESS_SKIP, "true")
        .with_meta(META_MAX_TOKENS, max_tokens.to_string())
}

fn required_meta(result: &TaskResult, key: &str) -> String {
    result
        .meta
        .get(key)
        .cloned()
        .unwrap_or_else(|| panic!("missing result metadata `{key}`"))
}

fn parse_text_generation(result: &TaskResult, max_tokens: usize) -> TextGenerationV1 {
    assert_eq!(result.payload_mime, TEXT_GENERATION_V1_JSON);
    assert_eq!(
        result.result_schema.as_deref(),
        Some(TEXT_GENERATION_V1_SCHEMA)
    );

    let response: TextGenerationV1 =
        serde_json::from_slice(&result.payload).expect("text generation payload decodes");
    assert_eq!(
        response
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.max_tokens),
        Some(max_tokens)
    );
    response
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
            .batch_key(FASTVLM_EMBEDS_TASK_NAME, &request)
            .is_ok()
        {
            return dtype;
        }
    }
    panic!("FastVLM embeds task did not accept fp16 or fp32 tensor input");
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
    env::var("LUMEN_FASTVLM_DECODE_LATENCY_RUNTIME")
        .or_else(|_| env::var("LUMEN_FASTVLM_LATENCY_RUNTIME"))
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
        if env_bool("LUMEN_FASTVLM_DECODE_LATENCY_NO_ACCELERATED")
            || env_bool("LUMEN_FASTVLM_LATENCY_NO_ACCELERATED")
            || env_bool("LUMEN_FASTVLM_E2E_NO_ACCELERATED")
        {
            return false;
        }
        true
    }

    #[cfg(not(target_vendor = "apple"))]
    {
        env_bool("LUMEN_FASTVLM_DECODE_LATENCY_ACCELERATED")
            || env_bool("LUMEN_FASTVLM_LATENCY_ACCELERATED")
            || env_bool("LUMEN_FASTVLM_E2E_ACCELERATED")
    }
}
