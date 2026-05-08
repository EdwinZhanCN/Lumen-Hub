use crate::core::{
    context::MLContext,
    node::MLNode,
    packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor, RuntimeType},
};
use async_trait::async_trait;
use ndarray::{ArrayD, ArrayViewD, IxDyn};
use std::collections::HashMap;

/// CPU node execution function based on `ndarray`.
///
/// This function receives input as a read-only view to avoid additional buffer copying inside the node;
/// the output returns an array with ownership for subsequent repackaging into `MLPacket`.
type ComputeFn = Box<dyn Fn(ArrayViewD<'_, f32>) -> Result<ArrayD<f32>, String> + Send + Sync>;

/// A node explicitly declared to run on CPU / ndarray runtime.
///
/// Input prioritizes the fast path of consuming `HostTensor::Float32`; if the upstream is another runtime,
/// it falls back to the slow path of `to_host_tensor()`, materializing data into host memory before entering `ndarray`.
pub struct NdArrayNode {
    name: String,
    input_name: String,
    output_name: String,
    input_descriptors: HashMap<String, MLPacketDescriptor>,
    output_descriptors: HashMap<String, MLPacketDescriptor>,
    compute: ComputeFn,
}

impl NdArrayNode {
    /// Creates a single-input, single-output `ndarray` CPU node.
    ///
    /// The current implementation only supports `Float32`, because the compute closure takes `ArrayViewD<f32>` as input.
    pub fn new(
        name: impl Into<String>,
        input_name: impl Into<String>,
        output_name: impl Into<String>,
        input_desc: MLPacketDescriptor,
        output_desc: MLPacketDescriptor,
        compute: ComputeFn,
    ) -> Result<Self, String> {
        if input_desc.dtype != MLPacketDataType::Float32 {
            return Err(format!(
                "NdArrayNode only supports Float32 input, got {:?}",
                input_desc.dtype
            ));
        }
        if output_desc.dtype != MLPacketDataType::Float32 {
            return Err(format!(
                "NdArrayNode only supports Float32 output, got {:?}",
                output_desc.dtype
            ));
        }

        let input_name = input_name.into();
        let output_name = output_name.into();

        Ok(Self {
            name: name.into(),
            input_name: input_name.clone(),
            output_name: output_name.clone(),
            input_descriptors: HashMap::from([(input_name, input_desc)]),
            output_descriptors: HashMap::from([(output_name, output_desc)]),
            compute,
        })
    }
}

#[async_trait]
impl MLNode for NdArrayNode {
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
        mut inputs: HashMap<String, MLPacket>,
        _context: &MLContext,
    ) -> Result<HashMap<String, MLPacket>, String> {
        let packet = inputs
            .remove(&self.input_name)
            .ok_or_else(|| format!("missing required input `{}`", self.input_name))?;

        if !inputs.is_empty() {
            let mut unknown_inputs = inputs.into_keys().collect::<Vec<_>>();
            unknown_inputs.sort();
            return Err(format!(
                "unexpected inputs provided: {}",
                unknown_inputs.join(", ")
            ));
        }

        let (context, actual_desc, payload) = packet.into_parts()?;
        let expected_desc = self
            .input_descriptors
            .get(&self.input_name)
            .expect("input descriptor should exist");
        expected_desc.validate_compatibility(&actual_desc, &self.input_name)?;

        let input_values = if payload.runtime() == RuntimeType::Cpu {
            match payload.into_any().downcast::<HostTensor>() {
                Ok(host_tensor) => match *host_tensor {
                    HostTensor::Float32(values) => values,
                    other => {
                        return Err(format!(
                            "NdArrayNode expected Float32 host tensor, got {other:?}"
                        ));
                    }
                },
                Err(_) => {
                    return Err(
                        "NdArrayNode received Cpu runtime payload that was not HostTensor"
                            .to_string(),
                    );
                }
            }
        } else {
            match payload.to_host_tensor()? {
                HostTensor::Float32(values) => values,
                other => {
                    return Err(format!(
                        "NdArrayNode expected Float32 host tensor, got {other:?}"
                    ));
                }
            }
        };

        let input_shape = IxDyn(&expected_desc.shape);
        let input_view = ArrayViewD::from_shape(input_shape, &input_values).map_err(|err| {
            format!(
                "failed to create ndarray view for input `{}`: {err}",
                self.input_name
            )
        })?;

        let output_array = (self.compute)(input_view)?;
        // Flatten into a contiguous host buffer to avoid depending on `ndarray`'s internal layout details.
        let output_values = output_array.iter().copied().collect::<Vec<_>>();
        let output_desc = self
            .output_descriptors
            .get(&self.output_name)
            .expect("output descriptor should exist")
            .clone();
        let output_packet =
            MLPacket::from_host_tensor(context, output_desc, HostTensor::Float32(output_values))?;

        Ok(HashMap::from([(self.output_name.clone(), output_packet)]))
    }
}
