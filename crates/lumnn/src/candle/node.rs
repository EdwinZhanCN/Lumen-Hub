use crate::{
    candle::convert::{
        collect_candle_outputs, device_runtime, input_descriptors_from_model,
        output_descriptors_from_model, prepare_candle_inputs,
    },
    core::{
        context::MLContext,
        node::MLNode,
        packet::{MLPacket, MLPacketDescriptor},
    },
};
use async_trait::async_trait;
use candle_core::Device;
use candle_onnx::onnx::ModelProto;
use std::{collections::HashMap, sync::Arc};

/// Represents a node that executes ONNX graphs using Candle's ONNX evaluator.
pub struct CandleOnnxNode {
    name: String,
    model: ModelProto,
    device: Device,
    input_descriptors: HashMap<String, MLPacketDescriptor>,
    output_descriptors: HashMap<String, MLPacketDescriptor>,
}

impl CandleOnnxNode {
    /// Loads an ONNX model and caches input and output descriptors.
    pub fn new(context: &MLContext, model_path: &str, name: String) -> Result<Self, String> {
        let device = select_candle_device(context.accelerated())?;
        let model = candle_onnx::read_file(model_path)
            .map_err(|err| format!("failed to load Candle ONNX model `{model_path}`: {err}"))?;
        let input_descriptors = input_descriptors_from_model(&model)?;
        let output_descriptors = output_descriptors_from_model(&model)?;

        Ok(Self {
            name,
            model,
            device,
            input_descriptors,
            output_descriptors,
        })
    }

    /// Returns the concrete Candle device selected for this node.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the runtime label exposed for tensors produced by this node.
    pub fn device_runtime(&self) -> &'static str {
        device_runtime(&self.device)
    }
}

#[async_trait]
impl MLNode for CandleOnnxNode {
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
        let candle_inputs =
            prepare_candle_inputs(inputs, &self.input_descriptors, &self.device).await?;
        let outputs = candle_onnx::simple_eval(&self.model, candle_inputs)
            .map_err(|err| format!("failed to execute Candle ONNX model: {err}"))?;

        collect_candle_outputs(outputs, Arc::new(context.clone()))
    }
}

fn select_candle_device(accelerated: bool) -> Result<Device, String> {
    if !accelerated {
        return Ok(Device::Cpu);
    }

    #[cfg(target_vendor = "apple")]
    {
        let device = Device::metal_if_available(0)
            .map_err(|err| format!("failed to initialize Candle Metal device: {err}"))?;
        if !device.is_cpu() {
            return Ok(device);
        }
    }

    #[cfg(not(target_vendor = "apple"))]
    {
        let device = Device::cuda_if_available(0)
            .map_err(|err| format!("failed to initialize Candle CUDA device: {err}"))?;
        if !device.is_cpu() {
            return Ok(device);
        }
    }

    Ok(Device::Cpu)
}
