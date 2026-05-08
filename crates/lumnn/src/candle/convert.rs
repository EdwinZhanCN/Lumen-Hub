use crate::core::{
    context::MLContext,
    packet::{
        HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor, PacketPayload, RuntimeType,
    },
};
use candle_core::{DType, Device, Tensor, WithDType};
use candle_onnx::onnx::{self, tensor_proto::DataType};
use half::f16;
use std::{
    any::Any,
    collections::{HashMap, HashSet},
    sync::Arc,
};

/// A payload wrapper for Candle tensors.
pub(crate) struct CandlePayload {
    tensor: Tensor,
}

impl CandlePayload {
    pub(crate) fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }

    pub(crate) fn into_inner(self) -> Tensor {
        self.tensor
    }
}

impl PacketPayload for CandlePayload {
    fn runtime(&self) -> RuntimeType {
        RuntimeType::backend("candle", device_runtime(self.tensor.device()))
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }

    fn to_host_tensor(&self) -> Result<HostTensor, String> {
        candle_tensor_to_host_tensor(&self.tensor)
    }
}

/// Extracts static contract descriptions used by the framework from ONNX graph inputs.
pub(crate) fn input_descriptors_from_model(
    model: &onnx::ModelProto,
) -> Result<HashMap<String, MLPacketDescriptor>, String> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| "Candle ONNX model has no graph".to_owned())?;
    let initializer_names = graph
        .initializer
        .iter()
        .map(|initializer| initializer.name.as_str())
        .collect::<HashSet<_>>();
    let mut descriptors = HashMap::new();

    for input in &graph.input {
        if initializer_names.contains(input.name.as_str()) {
            continue;
        }
        let descriptor = descriptor_from_value_info(input, "input")?;
        descriptors.insert(input.name.clone(), descriptor);
    }

    Ok(descriptors)
}

/// Extracts static contract descriptions used by the framework from ONNX graph outputs.
pub(crate) fn output_descriptors_from_model(
    model: &onnx::ModelProto,
) -> Result<HashMap<String, MLPacketDescriptor>, String> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| "Candle ONNX model has no graph".to_owned())?;
    let mut descriptors = HashMap::with_capacity(graph.output.len());

    for output in &graph.output {
        let descriptor = descriptor_from_value_info(output, "output")?;
        descriptors.insert(output.name.clone(), descriptor);
    }

    Ok(descriptors)
}

/// Validates and converts incoming packets into Candle tensors.
pub(crate) async fn prepare_candle_inputs(
    inputs: HashMap<String, MLPacket>,
    expected_descriptors: &HashMap<String, MLPacketDescriptor>,
    device: &Device,
) -> Result<HashMap<String, Tensor>, String> {
    let mut remaining_inputs = inputs;
    let mut candle_inputs = HashMap::with_capacity(expected_descriptors.len());

    for (name, expected_descriptor) in expected_descriptors {
        let packet = remaining_inputs
            .remove(name)
            .ok_or_else(|| format!("missing required input `{name}`"))?;

        expected_descriptor.validate_compatibility(&packet.descriptor, name)?;
        let tensor = packet_to_candle_tensor(name, packet, device).await?;
        candle_inputs.insert(name.clone(), tensor);
    }

    if !remaining_inputs.is_empty() {
        let mut unknown_inputs = remaining_inputs.into_keys().collect::<Vec<_>>();
        unknown_inputs.sort();
        return Err(format!(
            "unexpected inputs provided: {}",
            unknown_inputs.join(", ")
        ));
    }

    Ok(candle_inputs)
}

/// Converts Candle inference results back into the framework's packet collection.
pub(crate) fn collect_candle_outputs(
    outputs: HashMap<String, Tensor>,
    context: Arc<MLContext>,
) -> Result<HashMap<String, MLPacket>, String> {
    let mut result = HashMap::with_capacity(outputs.len());

    for (name, tensor) in outputs {
        let packet = candle_tensor_to_packet(&name, tensor, Arc::clone(&context))
            .map_err(|err| format!("failed to convert output `{name}`: {err}"))?;
        result.insert(name, packet);
    }

    Ok(result)
}

pub(crate) fn device_runtime(device: &Device) -> &'static str {
    if device.is_cuda() {
        "cuda"
    } else if device.is_metal() {
        "metal"
    } else if device.is_cpu() {
        "cpu"
    } else {
        "unknown"
    }
}

fn descriptor_from_value_info(
    value_info: &onnx::ValueInfoProto,
    kind: &str,
) -> Result<MLPacketDescriptor, String> {
    let tensor_type = value_info
        .r#type
        .as_ref()
        .and_then(|type_proto| type_proto.value.as_ref())
        .and_then(|value| match value {
            onnx::type_proto::Value::TensorType(tensor_type) => Some(tensor_type),
            _ => None,
        })
        .ok_or_else(|| format!("{kind} `{}` is not a tensor", value_info.name))?;

    let data_type = DataType::try_from(tensor_type.elem_type).map_err(|_| {
        format!(
            "{kind} `{}` has invalid tensor elem_type {}",
            value_info.name, tensor_type.elem_type
        )
    })?;
    let candle_dtype = candle_onnx::dtype(data_type).ok_or_else(|| {
        format!(
            "{kind} `{}` has unsupported tensor dtype {}",
            value_info.name,
            data_type.as_str_name()
        )
    })?;
    let dtype = candle_dtype_to_ml_dtype(candle_dtype)?;

    let shape_proto = tensor_type.shape.as_ref().ok_or_else(|| {
        format!(
            "{kind} `{}` tensor shape metadata is missing",
            value_info.name
        )
    })?;
    let (shape, dynamic_batch, dynamic_axes) =
        onnx_shape_to_packet_shape(shape_proto.dim.iter().map(|dim| dim.value.as_ref()))?;

    Ok(MLPacketDescriptor {
        dtype,
        shape,
        dynamic_batch,
        dynamic_axes,
    })
}

fn onnx_shape_to_packet_shape<'a>(
    dims: impl IntoIterator<Item = Option<&'a onnx::tensor_shape_proto::dimension::Value>>,
) -> Result<(Vec<usize>, bool, Vec<bool>), String> {
    let mut dynamic_batch = false;
    let mut dynamic_axes = Vec::new();
    let mut shape = Vec::new();

    for (index, dim) in dims.into_iter().enumerate() {
        match dim {
            Some(onnx::tensor_shape_proto::dimension::Value::DimValue(value)) if *value >= 0 => {
                let converted = usize::try_from(*value)
                    .map_err(|_| format!("dimension at axis {index} is too large for usize"))?;
                shape.push(converted);
                dynamic_axes.push(false);
            }
            Some(onnx::tensor_shape_proto::dimension::Value::DimValue(_))
            | Some(onnx::tensor_shape_proto::dimension::Value::DimParam(_))
            | None => {
                if index == 0 {
                    dynamic_batch = true;
                }
                shape.push(1);
                dynamic_axes.push(true);
            }
        }
    }

    Ok((shape, dynamic_batch, dynamic_axes))
}

async fn packet_to_candle_tensor(
    input_name: &str,
    packet: MLPacket,
    device: &Device,
) -> Result<Tensor, String> {
    let descriptor = packet.descriptor.clone();

    match packet.runtime().await? {
        RuntimeType::Backend {
            backend: "candle", ..
        }
        | RuntimeType::Cpu => {
            let (_, descriptor, payload) = packet.into_parts()?;
            let payload = payload.into_any();

            match payload.downcast::<CandlePayload>() {
                Ok(payload) => {
                    let tensor = payload.into_inner();
                    let actual_descriptor = descriptor_from_tensor(&tensor)?;
                    descriptor.validate_compatibility(&actual_descriptor, input_name)?;
                    move_tensor_to_device(tensor, device)
                }
                Err(payload) => match payload.downcast::<HostTensor>() {
                    Ok(host_tensor) => {
                        host_tensor.validate_against(&descriptor, input_name)?;
                        host_tensor_to_candle_tensor(
                            input_name,
                            &descriptor.shape,
                            *host_tensor,
                            device,
                        )
                    }
                    Err(_) => Err(format!(
                        "input `{input_name}` payload type is unsupported by CandleOnnxNode"
                    )),
                },
            }
        }
        _ => {
            let host_tensor = packet.to_host_tensor().await?;
            host_tensor.validate_against(&descriptor, input_name)?;
            host_tensor_to_candle_tensor(input_name, &descriptor.shape, host_tensor, device)
        }
    }
}

fn candle_tensor_to_packet(
    _output_name: &str,
    tensor: Tensor,
    context: Arc<MLContext>,
) -> Result<MLPacket, String> {
    let descriptor = descriptor_from_tensor(&tensor)?;
    let payload = Box::new(CandlePayload::new(tensor));
    Ok(MLPacket::from_payload(context, descriptor, payload))
}

fn descriptor_from_tensor(tensor: &Tensor) -> Result<MLPacketDescriptor, String> {
    Ok(MLPacketDescriptor::new(
        candle_dtype_to_ml_dtype(tensor.dtype())?,
        tensor.dims().to_vec(),
    ))
}

fn candle_dtype_to_ml_dtype(dtype: DType) -> Result<MLPacketDataType, String> {
    match dtype {
        DType::F32 => Ok(MLPacketDataType::Float32),
        DType::F16 => Ok(MLPacketDataType::Float16),
        DType::I64 => Ok(MLPacketDataType::Int64),
        DType::U32 => Ok(MLPacketDataType::Uint32),
        DType::U8 => Ok(MLPacketDataType::Uint8),
        other => Err(format!("unsupported Candle dtype `{other:?}`")),
    }
}

fn host_tensor_to_candle_tensor(
    input_name: &str,
    shape: &[usize],
    tensor: HostTensor,
    device: &Device,
) -> Result<Tensor, String> {
    match tensor {
        HostTensor::Float32(values) => build_tensor(input_name, shape, values, device),
        HostTensor::Float16(values) => build_tensor(input_name, shape, values, device),
        HostTensor::Uint32(values) => build_tensor(input_name, shape, values, device),
        HostTensor::Int64(values) => build_tensor(input_name, shape, values, device),
        HostTensor::Uint8(values) => build_tensor(input_name, shape, values, device),
        HostTensor::Int32(_) => Err(format!(
            "input `{input_name}` uses Int32, which candle-onnx does not currently support"
        )),
        HostTensor::Uint64(_) => Err(format!(
            "input `{input_name}` uses Uint64, which candle-onnx does not currently support"
        )),
        HostTensor::Int8(_) => Err(format!(
            "input `{input_name}` uses Int8, which candle-onnx does not currently support"
        )),
    }
}

fn build_tensor<T>(
    input_name: &str,
    shape: &[usize],
    values: Vec<T>,
    device: &Device,
) -> Result<Tensor, String>
where
    T: WithDType,
{
    let expected_len = shape.iter().product::<usize>();
    if values.len() != expected_len {
        return Err(format!(
            "input `{input_name}` element count mismatch: expected {expected_len}, got {}",
            values.len()
        ));
    }

    Tensor::from_slice(&values, shape, device)
        .map_err(|err| format!("failed to build Candle tensor for `{input_name}`: {err}"))
}

fn move_tensor_to_device(tensor: Tensor, device: &Device) -> Result<Tensor, String> {
    if tensor.device().same_device(device) {
        Ok(tensor)
    } else {
        tensor.to_device(device).map_err(|err| {
            format!(
                "failed to move Candle tensor to {}: {err}",
                device_runtime(device)
            )
        })
    }
}

fn candle_tensor_to_host_tensor(tensor: &Tensor) -> Result<HostTensor, String> {
    let cpu_tensor = tensor
        .to_device(&Device::Cpu)
        .map_err(|err| format!("failed to move Candle tensor to CPU: {err}"))?;
    let flat = cpu_tensor
        .flatten_all()
        .map_err(|err| format!("failed to flatten Candle tensor: {err}"))?;

    match cpu_tensor.dtype() {
        DType::F32 => flat
            .to_vec1::<f32>()
            .map(HostTensor::Float32)
            .map_err(|err| format!("failed to extract Float32 Candle tensor: {err}")),
        DType::F16 => flat
            .to_vec1::<f16>()
            .map(HostTensor::Float16)
            .map_err(|err| format!("failed to extract Float16 Candle tensor: {err}")),
        DType::I64 => flat
            .to_vec1::<i64>()
            .map(HostTensor::Int64)
            .map_err(|err| format!("failed to extract Int64 Candle tensor: {err}")),
        DType::U32 => flat
            .to_vec1::<u32>()
            .map(HostTensor::Uint32)
            .map_err(|err| format!("failed to extract Uint32 Candle tensor: {err}")),
        DType::U8 => flat
            .to_vec1::<u8>()
            .map(HostTensor::Uint8)
            .map_err(|err| format!("failed to extract Uint8 Candle tensor: {err}")),
        other => Err(format!("unsupported Candle dtype `{other:?}`")),
    }
}
