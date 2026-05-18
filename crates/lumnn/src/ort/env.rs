use ort::{environment::EnvironmentBuilder, ep::ExecutionProviderDispatch};
#[cfg(feature = "ort-load-dynamic")]
use std::path::PathBuf;
use std::sync::Mutex;
use thiserror::Error;

static ORT_ENV_MODE: Mutex<Option<OrtEnvMode>> = Mutex::new(None);

#[cfg(feature = "ort-openvino")]
const OPENVINO_DEVICE_ENV: &str = "LUMNN_ORT_OPENVINO_DEVICE";
#[cfg(feature = "ort-load-dynamic")]
const LUMNN_ORT_DYLIB_PATH_ENV: &str = "LUMNN_ORT_DYLIB_PATH";

type ExecutionProviderPlan = (Vec<ExecutionProviderDispatch>, Vec<String>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrtEnvMode {
    CpuOnly,
    Accelerated,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum OrtEnvInitError {
    #[error(
        "accelerated ONNX Runtime initialization requested, but this binary was not built with any ORT execution-provider feature; enable one of ort-tensorrt, ort-cuda, ort-directml, ort-coreml, ort-openvino, or ort-xnnpack"
    )]
    NoAcceleratedProviders,
    #[error(
        "ONNX Runtime environment is already initialized for CPU-only execution; accelerated execution must be configured before any session or default environment is created"
    )]
    AlreadyInitializedCpuOnly,
    #[error(
        "ONNX Runtime environment was already configured before lumnn could commit its execution-provider configuration"
    )]
    AlreadyConfigured,
    #[error("ONNX Runtime environment state lock was poisoned")]
    StateLock,
    #[cfg(feature = "ort-load-dynamic")]
    #[error("failed to initialize ONNX Runtime from dynamic library `{path}`: {message}")]
    DynamicLoad { path: String, message: String },
    #[cfg(feature = "ort-load-dynamic")]
    #[error("failed to resolve ONNX Runtime dynamic library path: {0}")]
    DynamicPath(String),
}

pub fn init_ort_env(accelerated: bool) -> Result<(), OrtEnvInitError> {
    let requested_mode = if accelerated {
        OrtEnvMode::Accelerated
    } else {
        OrtEnvMode::CpuOnly
    };

    let mut initialized_mode = ORT_ENV_MODE
        .lock()
        .map_err(|_| OrtEnvInitError::StateLock)?;

    if let Some(existing_mode) = *initialized_mode {
        return match (existing_mode, requested_mode) {
            (OrtEnvMode::CpuOnly, OrtEnvMode::Accelerated) => {
                Err(OrtEnvInitError::AlreadyInitializedCpuOnly)
            }
            _ => Ok(()),
        };
    }

    initialize_ort_env(requested_mode)?;
    *initialized_mode = Some(requested_mode);
    Ok(())
}

fn initialize_ort_env(requested_mode: OrtEnvMode) -> Result<(), OrtEnvInitError> {
    tracing::info!("Initializing ONNX Runtime global environment...");

    let mut builder = ort_environment_builder()?;

    if requested_mode == OrtEnvMode::Accelerated {
        let (execution_providers, provider_names) = accelerated_execution_providers()?;
        tracing::info!(
            providers = ?provider_names,
            "Preparing native accelerated execution..."
        );
        builder = builder.with_execution_providers(execution_providers);
    }

    if !builder.commit() {
        return Err(OrtEnvInitError::AlreadyConfigured);
    }

    Ok(())
}

fn accelerated_execution_providers() -> Result<ExecutionProviderPlan, OrtEnvInitError> {
    #[allow(unused_mut)]
    let mut providers = Vec::new();
    #[allow(unused_mut)]
    let mut provider_names = Vec::new();

    #[cfg(feature = "ort-tensorrt")]
    {
        let trt_cache = std::env::var("ORT_TRT_CACHE_DIR")
            .unwrap_or_else(|_| ".cache/onnxruntime/tensorrt".to_owned());

        providers.push(
            ort::ep::TensorRT::default()
                .with_device_id(0)
                .with_fp16(true)
                .with_int8(false)
                .with_max_workspace_size(4usize * 1024 * 1024 * 1024)
                .with_min_subgraph_size(3)
                .with_max_partition_iterations(1000)
                .with_engine_cache(true)
                .with_engine_cache_path(&trt_cache)
                .with_timing_cache(true)
                .with_timing_cache_path(&trt_cache)
                .with_builder_optimization_level(5)
                .with_context_memory_sharing(true)
                .with_layer_norm_fp32_fallback(true)
                .with_build_heuristics(false)
                .with_cuda_graph(false)
                .with_sparsity(false)
                .with_engine_hw_compatible(false)
                .build(),
        );
        provider_names.push(format!("TensorRT(fp16,cache={trt_cache})"));
    }

    #[cfg(feature = "ort-cuda")]
    {
        providers.push(
            ort::ep::CUDA::default()
                .with_device_id(0)
                .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Exhaustive)
                .with_conv_max_workspace(true)
                .with_tf32(true)
                .with_cuda_graph(false)
                .with_skip_layer_norm_strict_mode(false)
                .with_fuse_conv_bias(true)
                .build(),
        );
        provider_names.push("CUDA(tf32,exhaustive-cudnn)".to_owned());
    }

    #[cfg(feature = "ort-directml")]
    {
        providers.push(
            ort::ep::DirectML::default()
                .with_performance_preference(
                    ort::ep::directml::PerformancePreference::HighPerformance,
                )
                .with_device_filter(ort::ep::directml::DeviceFilter::Gpu)
                .build(),
        );
        provider_names.push("DirectML(high-performance-gpu)".to_owned());
    }

    #[cfg(feature = "ort-coreml")]
    {
        let coreml_cache = std::env::var("ORT_COREML_CACHE_DIR")
            .unwrap_or_else(|_| ".cache/onnxruntime/coreml".to_owned());

        providers.push(
            ort::ep::CoreML::default()
                .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                .with_static_input_shapes(false)
                .with_subgraphs(false)
                .with_specialization_strategy(
                    ort::ep::coreml::SpecializationStrategy::FastPrediction,
                )
                .with_low_precision_accumulation_on_gpu(false)
                .with_model_cache_dir(coreml_cache.clone())
                .build(),
        );
        provider_names.push(format!("CoreML(MLProgram,ALL,cache={coreml_cache})"));
    }

    #[cfg(feature = "ort-openvino")]
    {
        let device = openvino_device();

        let ov_cache = std::env::var("ORT_OPENVINO_CACHE_DIR")
            .unwrap_or_else(|_| ".cache/onnxruntime/openvino".to_owned());

        let mut ep = ort::ep::OpenVINO::default()
            .with_device_type(&device)
            .with_cache_dir(&ov_cache)
            .with_dynamic_shapes(true);

        if device.contains("GPU") || device.contains("NPU") {
            ep = ep.with_precision("FP16").with_num_streams(2);
        } else if device.contains("CPU") {
            ep = ep.with_precision("FP32");
        }

        providers.push(ep.build());
        provider_names.push(format!("OpenVINO({device},cache={ov_cache})"));
    }

    #[cfg(feature = "ort-xnnpack")]
    {
        use std::num::NonZeroUsize;
        let threads = std::thread::available_parallelism()
            .ok()
            .and_then(|n| NonZeroUsize::new(n.get()))
            .unwrap_or_else(|| NonZeroUsize::new(1).unwrap());

        providers.push(
            ort::ep::XNNPACK::default()
                .with_intra_op_num_threads(threads)
                .build(),
        );
        provider_names.push(format!("XNNPACK({threads} threads)"));
    }

    if providers.is_empty() {
        return Err(OrtEnvInitError::NoAcceleratedProviders);
    }

    Ok((providers, provider_names))
}

fn ort_environment_builder() -> Result<EnvironmentBuilder, OrtEnvInitError> {
    #[cfg(feature = "ort-load-dynamic")]
    {
        let path = resolve_ort_dylib_path()?;
        return ort::init_from(&path)
            .map(|builder| builder.with_name("lumnn_ort_backend"))
            .map_err(|err| OrtEnvInitError::DynamicLoad {
                path: path.display().to_string(),
                message: err.to_string(),
            });
    }

    #[cfg(not(feature = "ort-load-dynamic"))]
    {
        Ok(ort::init().with_name("lumnn_ort_backend"))
    }
}

#[cfg(feature = "ort-openvino")]
fn openvino_device() -> String {
    std::env::var(OPENVINO_DEVICE_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "GPU.0".to_owned())
}

#[cfg(feature = "ort-load-dynamic")]
fn resolve_ort_dylib_path() -> Result<PathBuf, OrtEnvInitError> {
    if let Some(path) = nonempty_env_path(LUMNN_ORT_DYLIB_PATH_ENV) {
        return Ok(path);
    }
    if let Some(path) = nonempty_env_path("ORT_DYLIB_PATH") {
        return Ok(path);
    }

    let exe_path =
        std::env::current_exe().map_err(|err| OrtEnvInitError::DynamicPath(err.to_string()))?;
    let exe_dir = exe_path.parent().ok_or_else(|| {
        OrtEnvInitError::DynamicPath("current executable has no parent".to_owned())
    })?;

    Ok(exe_dir.join(default_ort_dylib_name()))
}

#[cfg(feature = "ort-load-dynamic")]
fn nonempty_env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).and_then(|value| {
        if value.is_empty() {
            None
        } else {
            Some(PathBuf::from(value))
        }
    })
}

#[cfg(all(feature = "ort-load-dynamic", target_os = "windows"))]
fn default_ort_dylib_name() -> &'static str {
    "onnxruntime.dll"
}

#[cfg(all(
    feature = "ort-load-dynamic",
    any(target_os = "linux", target_os = "android")
))]
fn default_ort_dylib_name() -> &'static str {
    "libonnxruntime.so"
}

#[cfg(all(
    feature = "ort-load-dynamic",
    any(target_os = "macos", target_os = "ios")
))]
fn default_ort_dylib_name() -> &'static str {
    "libonnxruntime.dylib"
}

#[cfg(all(
    feature = "ort-load-dynamic",
    not(any(
        target_os = "windows",
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios"
    ))
))]
fn default_ort_dylib_name() -> &'static str {
    "libonnxruntime.so"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(any(
        feature = "ort-tensorrt",
        feature = "ort-cuda",
        feature = "ort-directml",
        feature = "ort-coreml",
        feature = "ort-openvino",
        feature = "ort-xnnpack"
    )))]
    fn accelerated_provider_plan_requires_explicit_ep_feature() {
        let err = accelerated_execution_providers()
            .expect_err("accelerated provider plan should require an EP feature");
        assert_eq!(err, OrtEnvInitError::NoAcceleratedProviders);
    }

    #[test]
    #[cfg(feature = "ort-openvino")]
    fn openvino_default_device_prefers_gpu_zero() {
        if std::env::var(OPENVINO_DEVICE_ENV).is_err() {
            assert_eq!(openvino_device(), "GPU.0");
        }
    }
}
