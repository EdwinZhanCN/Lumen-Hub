use crate::core::context::MLContext;
use crate::core::node::MLNode;
use crate::core::packet::{MLPacket, MLPacketDescriptor};
use crate::ort::convert::{
    collect_session_outputs, descriptors_from_outlets, prepare_ort_values, prepare_session_inputs,
};
use async_trait::async_trait;
use ort::session::builder::GraphOptimizationLevel;
use ort::{
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

/// Represents a node that executes ONNX models using ONNX Runtime.
pub struct OrtNode {
    name: String,
    /// The actual ONNX Runtime session.
    ///
    /// `ort::Session::run` requires a mutable borrow, but `MLNode::execute` only provides `&self`,
    /// so we wrap it in a `Mutex` to provide safe internal mutability.
    session: Mutex<Session>,
    /// Cached model input descriptors to avoid re-parsing model metadata on each execution.
    input_descriptors: HashMap<String, MLPacketDescriptor>,
    /// Cached model output descriptors for external queries of node capabilities.
    output_descriptors: HashMap<String, MLPacketDescriptor>,
    /// When set, execute through ORT I/O Binding and bind every model output to this device.
    io_binding_output_memory: Option<OrtIoBindingOutputMemory>,
}

impl OrtNode {
    /// Loads an ONNX model and caches input and output descriptors.
    pub fn new(context: &MLContext, model_path: &str, name: String) -> Result<Self, String> {
        crate::ort::env::init_ort_env(context.accelerated()).map_err(|err| {
            let mode = if context.accelerated() {
                "accelerated"
            } else {
                "CPU-only"
            };
            format!("failed to initialize {mode} ONNX Runtime: {err}")
        })?;

        let session = Session::builder()
            .map_err(|err| format!("failed to create ORT session builder: {err}"))?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|err| format!("failed to set optimization level: {err}"))?
            .commit_from_file(model_path)
            .map_err(|err| format!("failed to load ONNX model `{model_path}`: {err}"))?;

        let input_descriptors = descriptors_from_outlets(session.inputs(), "input")?;
        let output_descriptors = descriptors_from_outlets(session.outputs(), "output")?;

        Ok(Self {
            name,
            session: Mutex::new(session),
            input_descriptors,
            output_descriptors,
            io_binding_output_memory: None,
        })
    }

    /// Enables ONNX Runtime I/O Binding for this node's outputs.
    ///
    /// All model outputs are bound to the requested device, letting ONNX Runtime
    /// allocate them there before `RunWithBinding` executes. This is most useful
    /// when a downstream node can consume `OrtPayload` directly without calling
    /// `MLPacket::to_host_tensor()`.
    pub fn with_io_binding_outputs_to_device(
        mut self,
        allocation_device: AllocationDevice,
        device_id: i32,
        allocator_type: AllocatorType,
        memory_type: MemoryType,
    ) -> Self {
        self.io_binding_output_memory = Some(OrtIoBindingOutputMemory {
            allocation_device,
            device_id,
            allocator_type,
            memory_type,
        });
        self
    }

    /// Enables ONNX Runtime I/O Binding for this node's outputs using an existing memory description.
    pub fn with_io_binding_outputs_to_memory_info(
        mut self,
        memory_info: &MemoryInfo,
    ) -> Result<Self, String> {
        self.io_binding_output_memory =
            Some(OrtIoBindingOutputMemory::from_memory_info(memory_info)?);
        Ok(self)
    }
}

#[async_trait]
impl MLNode for OrtNode {
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
        if let Some(output_memory) = self.io_binding_output_memory {
            let output_memory = output_memory.to_memory_info()?;
            return self.execute_with_io_binding(inputs, context, output_memory);
        }

        let ort_inputs = prepare_session_inputs(inputs, &self.input_descriptors)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| "failed to lock ORT session".to_string())?;
        let outputs = session
            .run(ort_inputs)
            .map_err(|err| format!("failed to execute ORT session: {err}"))?;

        collect_session_outputs(outputs, Arc::new(context.clone()))
    }
}

impl OrtNode {
    fn execute_with_io_binding(
        &self,
        inputs: HashMap<String, MLPacket>,
        context: &MLContext,
        output_memory: MemoryInfo,
    ) -> Result<HashMap<String, MLPacket>, String> {
        let ort_inputs = prepare_ort_values(inputs, &self.input_descriptors)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| "failed to lock ORT session".to_string())?;
        let mut binding = session
            .create_binding()
            .map_err(|err| format!("failed to create ORT I/O binding: {err}"))?;

        for (name, value) in &ort_inputs {
            binding
                .bind_input(name, value)
                .map_err(|err| format!("failed to bind ORT input `{name}`: {err}"))?;
        }

        for name in self.output_descriptors.keys() {
            binding
                .bind_output_to_device(name, &output_memory)
                .map_err(|err| format!("failed to bind ORT output `{name}`: {err}"))?;
        }

        let outputs = session
            .run_binding(&binding)
            .map_err(|err| format!("failed to execute ORT session with I/O binding: {err}"))?;

        collect_session_outputs(outputs, Arc::new(context.clone()))
    }
}

#[derive(Debug, Clone, Copy)]
struct OrtIoBindingOutputMemory {
    allocation_device: AllocationDevice,
    device_id: i32,
    allocator_type: AllocatorType,
    memory_type: MemoryType,
}

impl OrtIoBindingOutputMemory {
    fn from_memory_info(memory_info: &MemoryInfo) -> Result<Self, String> {
        Ok(Self {
            allocation_device: canonical_allocation_device(memory_info.allocation_device())?,
            device_id: memory_info.device_id(),
            allocator_type: memory_info.allocator_type(),
            memory_type: memory_info.memory_type(),
        })
    }

    fn to_memory_info(self) -> Result<MemoryInfo, String> {
        MemoryInfo::new(
            self.allocation_device,
            self.device_id,
            self.allocator_type,
            self.memory_type,
        )
        .map_err(|err| format!("failed to create ORT I/O binding output memory info: {err}"))
    }
}

fn canonical_allocation_device(device: AllocationDevice) -> Result<AllocationDevice, String> {
    let known_devices = [
        AllocationDevice::CPU,
        AllocationDevice::CUDA,
        AllocationDevice::CUDA_PINNED,
        AllocationDevice::CANN,
        AllocationDevice::CANN_PINNED,
        AllocationDevice::DIRECTML,
        AllocationDevice::HIP,
        AllocationDevice::HIP_PINNED,
        AllocationDevice::OPENVINO_CPU,
        AllocationDevice::OPENVINO_GPU,
        AllocationDevice::QNN_HTP_SHARED,
        AllocationDevice::WEBGPU_BUFFER,
    ];

    known_devices
        .into_iter()
        .find(|known_device| *known_device == device)
        .ok_or_else(|| {
            format!(
                "unsupported ORT I/O binding allocation device `{}`",
                device.as_str()
            )
        })
}
