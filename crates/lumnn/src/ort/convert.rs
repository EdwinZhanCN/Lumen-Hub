use crate::core::{
    context::MLContext,
    packet::{
        HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor, PacketPayload, RuntimeType,
    },
};
use half::f16;
use ort::{
    AsPointer,
    memory::AllocationDevice,
    memory::MemoryInfo,
    session::{SessionInputValue, SessionOutputs},
    value::{DynTensorValueType, DynValue, Outlet, Tensor, TensorElementType, ValueType},
};
use std::{any::Any, collections::HashMap, ptr::NonNull, sync::Arc};

/// A payload wrapper for ONNX Runtime values.
pub(crate) struct OrtPayload {
    value: DynValue,
}

impl OrtPayload {
    pub(crate) fn new(value: DynValue) -> Self {
        Self { value }
    }

    pub(crate) fn into_inner(self) -> DynValue {
        self.value
    }
}

impl PacketPayload for OrtPayload {
    fn runtime(&self) -> RuntimeType {
        dyn_value_runtime(&self.value)
    }

    fn as_any(&self) -> &(dyn Any + Send + Sync) {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }

    fn to_host_tensor(&self) -> Result<HostTensor, String> {
        dyn_value_to_host_tensor(&self.value)
    }
}

/// Extracts static contract descriptions used internally by the framework from ORT's input/output definitions.
pub(crate) fn descriptors_from_outlets(
    outlets: &[Outlet],
    kind: &str,
) -> Result<HashMap<String, MLPacketDescriptor>, String> {
    let mut descriptors = HashMap::with_capacity(outlets.len());

    for outlet in outlets {
        let descriptor = descriptor_from_ort_value_type(outlet.dtype())
            .map_err(|err| format!("invalid {kind} `{}`: {err}", outlet.name()))?;
        descriptors.insert(outlet.name().to_string(), descriptor);
    }

    Ok(descriptors)
}

/// Validates and converts the incoming `MLPacket` collection into ORT-executable inputs.
pub(crate) fn prepare_session_inputs(
    inputs: HashMap<String, MLPacket>,
    expected_descriptors: &HashMap<String, MLPacketDescriptor>,
) -> Result<Vec<(String, SessionInputValue<'static>)>, String> {
    prepare_ort_values(inputs, expected_descriptors).map(|values| {
        values
            .into_iter()
            .map(|(name, value)| (name, SessionInputValue::from(value)))
            .collect()
    })
}

/// Validates and converts incoming packets into owned ORT values.
pub(crate) fn prepare_ort_values(
    inputs: HashMap<String, MLPacket>,
    expected_descriptors: &HashMap<String, MLPacketDescriptor>,
) -> Result<Vec<(String, DynValue)>, String> {
    let mut remaining_inputs = inputs;
    let mut ort_values = Vec::with_capacity(expected_descriptors.len());

    for (name, expected_descriptor) in expected_descriptors {
        let packet = remaining_inputs
            .remove(name)
            .ok_or_else(|| format!("missing required input `{name}`"))?;

        expected_descriptor.validate_compatibility(&packet.descriptor, name)?;

        let ort_value = packet_to_ort_value(name, packet)?;
        ort_values.push((name.clone(), ort_value));
    }

    if !remaining_inputs.is_empty() {
        let mut unknown_inputs = remaining_inputs.into_keys().collect::<Vec<_>>();
        unknown_inputs.sort();
        return Err(format!(
            "unexpected inputs provided: {}",
            unknown_inputs.join(", ")
        ));
    }

    Ok(ort_values)
}

/// Converts ORT inference results back into the framework's `MLPacket` collection.
pub(crate) fn collect_session_outputs(
    outputs: SessionOutputs<'_>,
    context: Arc<MLContext>,
) -> Result<HashMap<String, MLPacket>, String> {
    let mut result = HashMap::with_capacity(outputs.len());

    for (name, value) in outputs {
        let packet = ort_value_to_packet(name, value, Arc::clone(&context))
            .map_err(|err| format!("failed to convert output `{name}`: {err}"))?;
        result.insert(name.to_string(), packet);
    }

    Ok(result)
}

/// Converts ORT's value type into a tensor description expressible by the current framework.
pub(crate) fn descriptor_from_ort_value_type(
    value_type: &ValueType,
) -> Result<MLPacketDescriptor, String> {
    let tensor_type = value_type
        .tensor_type()
        .ok_or_else(|| "only tensor inputs/outputs are supported".to_string())?;
    let shape = value_type
        .tensor_shape()
        .ok_or_else(|| "tensor shape metadata is missing".to_string())?;

    let (shape, dynamic_batch, dynamic_axes) = ort_shape_to_packet_shape(shape.iter().copied())?;

    Ok(MLPacketDescriptor {
        dtype: ort_dtype_to_ml_dtype(tensor_type)?,
        shape,
        dynamic_batch,
        dynamic_axes,
    })
}

/// Maps ORT's tensor element type to the framework's internal data type.
pub(crate) fn ort_dtype_to_ml_dtype(dtype: TensorElementType) -> Result<MLPacketDataType, String> {
    match dtype {
        TensorElementType::Float32 => Ok(MLPacketDataType::Float32),
        TensorElementType::Float16 => Ok(MLPacketDataType::Float16),
        TensorElementType::Int32 => Ok(MLPacketDataType::Int32),
        TensorElementType::Uint32 => Ok(MLPacketDataType::Uint32),
        TensorElementType::Int64 => Ok(MLPacketDataType::Int64),
        TensorElementType::Uint64 => Ok(MLPacketDataType::Uint64),
        TensorElementType::Int8 => Ok(MLPacketDataType::Int8),
        TensorElementType::Uint8 => Ok(MLPacketDataType::Uint8),
        other => Err(format!("unsupported tensor element type `{other}`")),
    }
}

/// Converts ORT's `i64` dimensions to internal `usize` shape.
///
/// Negative dimensions are accepted as dynamic axes. Axis 0 is also exposed
/// through the legacy `dynamic_batch` flag for existing batch-only callers.
pub(crate) fn ort_shape_to_packet_shape(
    dims: impl IntoIterator<Item = i64>,
) -> Result<(Vec<usize>, bool, Vec<bool>), String> {
    let mut dynamic_batch = false;
    let mut dynamic_axes = Vec::new();
    let mut shape = Vec::new();

    for (index, dim) in dims.into_iter().enumerate() {
        let converted = match dim {
            dim if dim < 0 && index == 0 => {
                dynamic_batch = true;
                dynamic_axes.push(true);
                Ok(1)
            }
            dim if dim < 0 => {
                dynamic_axes.push(true);
                Ok(1)
            }
            dim => usize::try_from(dim)
                .map_err(|_| format!("dimension at axis {index} is too large for usize")),
        }?;
        if dim >= 0 {
            dynamic_axes.push(false);
        }
        shape.push(converted);
    }

    Ok((shape, dynamic_batch, dynamic_axes))
}

/// Converts internal packet to ORT-consumable tensor value.
pub(crate) fn packet_to_ort_value(input_name: &str, packet: MLPacket) -> Result<DynValue, String> {
    let (_, descriptor, payload) = packet.into_parts()?;
    let payload = payload.into_any();

    match payload.downcast::<OrtPayload>() {
        Ok(payload) => {
            let actual_descriptor =
                descriptor_from_ort_value_type(payload.value.dtype()).map_err(|err| {
                    format!("input `{input_name}` contains invalid ORT payload: {err}")
                })?;
            descriptor.validate_compatibility(&actual_descriptor, input_name)?;
            Ok(payload.into_inner())
        }
        Err(payload) => match payload.downcast::<HostTensor>() {
            Ok(host_tensor) => {
                host_tensor.validate_against(&descriptor, input_name)?;
                host_tensor_to_ort_value(input_name, descriptor.shape, *host_tensor)
            }
            Err(_) => Err(format!(
                "input `{input_name}` payload type is unsupported by OrtNode"
            )),
        },
    }
}

/// Repackages ORT output value into the framework's own `MLPacket`.
pub(crate) fn ort_value_to_packet(
    _output_name: &str,
    value: DynValue,
    context: Arc<MLContext>,
) -> Result<MLPacket, String> {
    let descriptor = descriptor_from_ort_value_type(value.dtype())?;
    let payload = Box::new(OrtPayload::new(value));
    Ok(MLPacket::from_payload(context, descriptor, payload))
}

/// Converts standard host tensor to ORT-consumable tensor value.
fn host_tensor_to_ort_value(
    input_name: &str,
    shape: Vec<usize>,
    tensor: HostTensor,
) -> Result<DynValue, String> {
    match tensor {
        HostTensor::Float32(values) => build_owned_tensor_value(input_name, shape, values),
        HostTensor::Float16(values) => build_owned_tensor_value(input_name, shape, values),
        HostTensor::Int32(values) => build_owned_tensor_value(input_name, shape, values),
        HostTensor::Uint32(values) => build_owned_tensor_value(input_name, shape, values),
        HostTensor::Int64(values) => build_owned_tensor_value(input_name, shape, values),
        HostTensor::Uint64(values) => build_owned_tensor_value(input_name, shape, values),
        HostTensor::Int8(values) => build_owned_tensor_value(input_name, shape, values),
        HostTensor::Uint8(values) => build_owned_tensor_value(input_name, shape, values),
    }
}

/// Builds an owned tensor value from the given shape and values.
fn build_owned_tensor_value<T>(
    input_name: &str,
    shape: Vec<usize>,
    values: Vec<T>,
) -> Result<DynValue, String>
where
    T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + Clone + Send + Sync + 'static,
{
    let expected_len = shape.iter().product::<usize>();
    if values.len() != expected_len {
        return Err(format!(
            "input `{input_name}` element count mismatch: expected {expected_len}, got {}",
            values.len()
        ));
    }

    if expected_len == 0 {
        return build_zero_length_tensor_value::<T>(input_name, shape);
    }

    Tensor::from_array((shape, values))
        .map(DynValue::from)
        .map_err(|err| format!("failed to build ORT tensor for `{input_name}`: {err}"))
}

fn build_zero_length_tensor_value<T>(
    input_name: &str,
    shape: Vec<usize>,
) -> Result<DynValue, String>
where
    T: ort::value::PrimitiveTensorElementType + std::fmt::Debug + Clone + Send + Sync + 'static,
{
    let shape_i64 = shape
        .iter()
        .copied()
        .map(|dim| {
            i64::try_from(dim)
                .map_err(|_| format!("input `{input_name}` shape dimension is too large: {dim}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let memory_info = MemoryInfo::default();
    let mut value_ptr: *mut ort::sys::OrtValue = std::ptr::null_mut();

    let status = unsafe {
        (ort::api().CreateTensorWithDataAsOrtValue)(
            memory_info.ptr(),
            NonNull::<T>::dangling().as_ptr().cast(),
            0,
            shape_i64.as_ptr(),
            shape_i64.len(),
            T::into_tensor_element_type().into(),
            &mut value_ptr,
        )
    };
    unsafe { ort::Error::result_from_status(status) }.map_err(|err| {
        format!("failed to build zero-length ORT tensor for `{input_name}`: {err}")
    })?;
    let value_ptr = NonNull::new(value_ptr)
        .ok_or_else(|| format!("ORT returned null tensor for zero-length input `{input_name}`"))?;

    Ok(unsafe { DynValue::from_ptr(value_ptr, None) })
}

/// Converts a dynamic value to a host tensor.
fn dyn_value_to_host_tensor(value: &DynValue) -> Result<HostTensor, String> {
    let materialized = maybe_materialize_cpu_tensor(value)?;
    let value = materialized.as_ref().unwrap_or(value);
    let dtype = ort_dtype_to_ml_dtype(
        value
            .dtype()
            .tensor_type()
            .ok_or_else(|| "only tensor values can be materialized to host".to_string())?,
    )?;

    match dtype {
        MLPacketDataType::Float32 => extract_host_tensor::<f32, _>(value, HostTensor::Float32),
        MLPacketDataType::Float16 => extract_host_tensor::<f16, _>(value, HostTensor::Float16),
        MLPacketDataType::Int32 => extract_host_tensor::<i32, _>(value, HostTensor::Int32),
        MLPacketDataType::Uint32 => extract_host_tensor::<u32, _>(value, HostTensor::Uint32),
        MLPacketDataType::Int64 => extract_host_tensor::<i64, _>(value, HostTensor::Int64),
        MLPacketDataType::Uint64 => extract_host_tensor::<u64, _>(value, HostTensor::Uint64),
        MLPacketDataType::Int8 => extract_host_tensor::<i8, _>(value, HostTensor::Int8),
        MLPacketDataType::Uint8 => extract_host_tensor::<u8, _>(value, HostTensor::Uint8),
    }
}

/// Attempts to materialize a tensor to CPU if not already accessible.
fn maybe_materialize_cpu_tensor(value: &DynValue) -> Result<Option<DynValue>, String> {
    let tensor = value
        .view()
        .into_dyn()
        .try_upgrade()
        .map_err(|_| "failed to upgrade ORT value view".to_string())?
        .downcast::<DynTensorValueType>()
        .map_err(|err| format!("only tensor values are supported: {err}"))?;

    if tensor.memory_info().is_cpu_accessible() {
        return Ok(None);
    }

    tensor
        .to(AllocationDevice::CPU, 0)
        .map(|tensor| Some(tensor.into_dyn()))
        .map_err(|err| format!("failed to copy ORT tensor to CPU: {err}"))
}

/// Extracts a host tensor from a dynamic value.
fn extract_host_tensor<T, F>(value: &DynValue, constructor: F) -> Result<HostTensor, String>
where
    T: ort::value::PrimitiveTensorElementType + Clone,
    F: FnOnce(Vec<T>) -> HostTensor,
{
    let (_, data) = value
        .try_extract_tensor::<T>()
        .map_err(|err| format!("failed to extract ORT tensor to host: {err}"))?;
    Ok(constructor(data.to_vec()))
}

/// Determines the runtime type of a dynamic value.
fn dyn_value_runtime(value: &DynValue) -> RuntimeType {
    match value.view().into_dyn().try_upgrade() {
        Ok(value) => match value.downcast::<DynTensorValueType>() {
            Ok(tensor) => {
                RuntimeType::backend("ort", tensor.memory_info().allocation_device().as_str())
            }
            Err(_) => RuntimeType::backend("ort", "Unknown"),
        },
        Err(_) => RuntimeType::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::ort_shape_to_packet_shape;

    #[test]
    fn accepts_dynamic_batch_axis() {
        let (shape, dynamic_batch, dynamic_axes) =
            ort_shape_to_packet_shape([-1, 3, 224, 224]).expect("dynamic batch should convert");

        assert_eq!(shape, vec![1, 3, 224, 224]);
        assert!(dynamic_batch);
        assert_eq!(dynamic_axes, vec![true, false, false, false]);
    }

    #[test]
    fn accepts_dynamic_non_batch_axis() {
        let (shape, dynamic_batch, dynamic_axes) =
            ort_shape_to_packet_shape([1, -1]).expect("dynamic sequence should convert");

        assert_eq!(shape, vec![1, 1]);
        assert!(!dynamic_batch);
        assert_eq!(dynamic_axes, vec![false, true]);
    }
}
