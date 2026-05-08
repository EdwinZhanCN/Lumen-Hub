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
        providers.push(ort::ep::TensorRT::default().build());
        provider_names.push("TensorRT".to_owned());
    }

    #[cfg(feature = "ort-cuda")]
    {
        providers.push(ort::ep::CUDA::default().build());
        provider_names.push("CUDA".to_owned());
    }

    #[cfg(feature = "ort-directml")]
    {
        providers.push(ort::ep::DirectML::default().build());
        provider_names.push("DirectML".to_owned());
    }

    #[cfg(feature = "ort-coreml")]
    {
        providers.push(ort::ep::CoreML::default().build());
        provider_names.push("CoreML".to_owned());
    }

    #[cfg(feature = "ort-openvino")]
    {
        let device = openvino_device();
        providers.push(
            ort::ep::OpenVINO::default()
                .with_device_type(&device)
                .build(),
        );
        provider_names.push(format!("OpenVINO({device})"));
    }

    #[cfg(feature = "ort-xnnpack")]
    {
        providers.push(ort::ep::XNNPACK::default().build());
        provider_names.push("XNNPACK".to_owned());
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
