use crate::core::context::MLContext;
use crate::core::node::MLNode;
use crate::core::packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor};
use async_trait::async_trait;
use lumnn_mnn_sys::{Backend, DataFormat, InferenceConfig, InferenceEngine, PrecisionMode};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::convert::mnn_shape_to_descriptor;

/// A computational graph node backed by an MNN (Mobile Neural Network) model.
///
/// `MnnNode` loads an `.mnn` model file and executes it via the MNN inference
/// framework. It supports both static and dynamic input shapes.
///
/// # I/O convention
///
/// The node exposes all named MNN inputs and outputs using the names recorded
/// in the model file.
///
/// # Dynamic shapes
///
/// When the model declares a dynamic dimension (e.g. variable batch size), or
/// when the actual input shape differs from the shape seen at load time, the
/// node automatically selects the dynamic inference path
/// ([`InferenceEngine::run_dynamic_raw`]).
///
/// # Thread safety
///
/// MNN inference is synchronous and serialised through a global C++ mutex.
/// The node wraps the engine in a [`std::sync::Mutex`] to satisfy
/// [`MLNode::execute`]'s `&self` contract.
pub struct MnnNode {
    name: String,
    engine: Mutex<InferenceEngine>,
    input_descriptors: HashMap<String, MLPacketDescriptor>,
    output_descriptors: HashMap<String, MLPacketDescriptor>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    has_dynamic_shape: bool,
}

impl MnnNode {
    /// Loads an MNN model and caches its input and output descriptors.
    ///
    /// # Parameters
    /// - `context`: execution context; when [`MLContext::accelerated`] is
    ///   `false`, the CPU backend is forced regardless of Cargo features.
    /// - `model_path`: path to the `.mnn` model file.
    /// - `name`: human-readable label for this node.
    ///
    /// # Errors
    /// Returns `Err` if the model file cannot be read or is not a valid MNN
    /// model.
    pub fn new(context: &MLContext, model_path: &str, name: String) -> Result<Self, String> {
        let config = InferenceConfig {
            thread_count: 0, // auto
            precision_mode: PrecisionMode::High,
            use_cache: true,
            data_format: DataFormat::Auto,
            backend: resolve_backend(context.accelerated()),
        };

        let engine = InferenceEngine::from_file(model_path, Some(config))
            .map_err(|e| format!("failed to load MNN model `{model_path}`: {e}"))?;

        let has_dynamic = engine.has_dynamic_shape();
        let input_names = engine.input_names().to_vec();
        let output_names = engine.output_names().to_vec();

        // Build descriptors for all inputs
        let mut input_descriptors = HashMap::new();
        for (i, name) in input_names.iter().enumerate() {
            let shape = engine.input_shape_at(i).unwrap_or(&[]);
            let dtype_code = engine.input_dtype_at(i);
            input_descriptors.insert(name.clone(), mnn_shape_to_descriptor(shape, dtype_code));
        }

        // Build descriptors for all outputs
        let mut output_descriptors = HashMap::new();
        for (i, name) in output_names.iter().enumerate() {
            let shape = engine.output_shape_at(i).unwrap_or(&[]);
            let dtype_code = engine.output_dtype_at(i);
            output_descriptors.insert(name.clone(), mnn_shape_to_descriptor(shape, dtype_code));
        }

        Ok(Self {
            name,
            engine: Mutex::new(engine),
            input_descriptors,
            output_descriptors,
            input_names,
            output_names,
            has_dynamic_shape: has_dynamic,
        })
    }

    /// Returns whether this model uses dynamic shapes.
    pub fn has_dynamic_shape(&self) -> bool {
        self.has_dynamic_shape
    }

    /// Query the actual runtime backend of the loaded model.
    ///
    /// Returns the MNNForwardType integer:
    /// 0=CPU, 1=Metal, 2=CUDA, 3=OpenCL, 6=OpenGL, 7=Vulkan, 5=CoreML.
    pub fn backend_type(&self) -> Result<i32, String> {
        let engine = self
            .engine
            .lock()
            .map_err(|_| "failed to lock engine".to_string())?;
        Ok(engine.backend_type())
    }
}

#[async_trait]
impl MLNode for MnnNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> {
        &self.input_descriptors
    }

    fn output_descriptors(&self) -> &HashMap<String, MLPacketDescriptor> {
        &self.output_descriptors
    }

    async fn execute(
        &self,
        inputs: HashMap<String, MLPacket>,
        context: &MLContext,
    ) -> Result<HashMap<String, MLPacket>, String> {
        // ── ① Materialize all inputs to host (without holding the lock) ──
        let mut materialized: Vec<(String, MLPacketDescriptor, HostTensor)> =
            Vec::with_capacity(self.input_names.len());
        for name in &self.input_names {
            let desc = self
                .input_descriptors
                .get(name)
                .ok_or_else(|| format!("missing cached input descriptor for `{name}`"))?;
            let packet = inputs
                .get(name)
                .ok_or_else(|| format!("missing required input `{name}`"))?;
            desc.validate_compatibility(&packet.descriptor, name)
                .map_err(|e| format!("input validation failed: {e}"))?;
            let host = packet
                .to_host_tensor()
                .await
                .map_err(|e| format!("failed to materialize input `{name}`: {e}"))?;
            materialized.push((name.clone(), packet.descriptor.clone(), host));
        }
        for name in inputs.keys() {
            if !self.input_descriptors.contains_key(name) {
                return Err(format!("unexpected input `{name}`"));
            }
        }

        // ── ② Lock engine, copy all inputs, run, copy all outputs ──
        let engine = self
            .engine
            .lock()
            .map_err(|_| "failed to lock MNN engine".to_string())?;

        for (name, actual_desc, host) in &materialized {
            let expected_desc = self
                .input_descriptors
                .get(name)
                .ok_or_else(|| format!("missing cached input descriptor for `{name}`"))?;
            if expected_desc
                .dynamic_axes
                .iter()
                .any(|is_dynamic| *is_dynamic)
                || expected_desc.shape != actual_desc.shape
            {
                engine
                    .resize_input_by_name(name, &actual_desc.shape)
                    .map_err(|e| {
                        format!("resize input `{name}` to {:?}: {e}", actual_desc.shape)
                    })?;
            }
            match host {
                HostTensor::Float32(v) => engine
                    .copy_input_f32(name, v)
                    .map_err(|e| format!("copy f32 input `{name}`: {e}"))?,
                HostTensor::Int32(v) => engine
                    .copy_input_i32(name, v)
                    .map_err(|e| format!("copy i32 input `{name}`: {e}"))?,
                HostTensor::Int64(v) => engine
                    .copy_input_i64(name, v)
                    .map_err(|e| format!("copy i64 input `{name}`: {e}"))?,
                other => {
                    return Err(format!(
                        "MnnNode unsupported dtype {:?} on `{name}`",
                        other.dtype()
                    ));
                }
            }
        }

        engine
            .run_only()
            .map_err(|e| format!("MNN inference failed: {e}"))?;

        let ctx = Arc::new(context.clone());
        let mut result = HashMap::new();
        for name in &self.output_names {
            let desc = self
                .output_descriptors
                .get(name)
                .ok_or_else(|| format!("missing cached output descriptor for `{name}`"))?;
            let shape = engine
                .output_shape_by_name(name)
                .map_err(|e| format!("query output shape `{name}`: {e}"))?;
            let elem_count: usize = shape.iter().product();
            match desc.dtype {
                MLPacketDataType::Float32 => {
                    let mut buf = vec![0.0f32; elem_count];
                    engine
                        .copy_output_f32(name, &mut buf)
                        .map_err(|e| format!("copy f32 output `{name}`: {e}"))?;
                    let out_desc = MLPacketDescriptor::new(MLPacketDataType::Float32, shape);
                    let packet = ctx
                        .packet_from_f32(out_desc.shape.clone(), buf)
                        .map_err(|e| format!("failed to create output packet `{name}`: {e}"))?;
                    result.insert(name.clone(), packet);
                }
                MLPacketDataType::Int32 => {
                    let mut buf = vec![0i32; elem_count];
                    engine
                        .copy_output_i32(name, &mut buf)
                        .map_err(|e| format!("copy i32 output `{name}`: {e}"))?;
                    let out_desc = MLPacketDescriptor::new(MLPacketDataType::Int32, shape);
                    let packet = ctx
                        .packet_from_host_tensor(out_desc, HostTensor::Int32(buf))
                        .map_err(|e| format!("failed to create output packet `{name}`: {e}"))?;
                    result.insert(name.clone(), packet);
                }
                MLPacketDataType::Int64 => {
                    let mut buf = vec![0i64; elem_count];
                    engine
                        .copy_output_i64(name, &mut buf)
                        .map_err(|e| format!("copy i64 output `{name}`: {e}"))?;
                    let out_desc = MLPacketDescriptor::new(MLPacketDataType::Int64, shape);
                    let packet = ctx
                        .packet_from_host_tensor(out_desc, HostTensor::Int64(buf))
                        .map_err(|e| format!("failed to create output packet `{name}`: {e}"))?;
                    result.insert(name.clone(), packet);
                }
                other => return Err(format!("unsupported output dtype {:?} on `{name}`", other)),
            }
        }

        Ok(result)
    }
}

// ── Backend resolution ──

/// Resolves the MNN backend based on the compile-time feature flags and the
/// runtime [`MLContext::accelerated`] flag.
///
/// When `accelerated` is `false`, returns [`Backend::CPU`] regardless of
/// features. When `true`, the first enabled feature in priority order is used:
///
/// `CUDA > Metal > Vulkan > OpenCL > OpenGL > CoreML > CPU`
#[allow(unreachable_code)]
fn resolve_backend(accelerated: bool) -> Backend {
    if !accelerated {
        return Backend::CPU;
    }

    #[cfg(feature = "mnn-cuda")]
    {
        return Backend::CUDA;
    }
    #[cfg(feature = "mnn-metal")]
    {
        return Backend::Metal;
    }
    #[cfg(feature = "mnn-vulkan")]
    {
        return Backend::Vulkan;
    }
    #[cfg(feature = "mnn-opencl")]
    {
        return Backend::OpenCL;
    }
    #[cfg(feature = "mnn-opengl")]
    {
        return Backend::OpenGL;
    }
    #[cfg(feature = "mnn-coreml")]
    {
        return Backend::CoreML;
    }

    Backend::CPU
}
