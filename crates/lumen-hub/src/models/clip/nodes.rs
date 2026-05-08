use std::collections::HashMap;

use async_trait::async_trait;
use half::f16;
use lumnn::core::{
    context::MLContext,
    node::MLNode,
    packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor},
};

/// Normalizes each embedding vector along the last tensor dimension.
///
/// Inputs may be `Float32` or `Float16`. Computation is performed in `f32` for
/// numerical stability. By default the node emits `Float32`, while callers that
/// need a lower-precision pipeline contract can opt into `Float16` output.
pub struct L2NormalizeNode {
    name: String,
    input_name: String,
    output_name: String,
    input_descriptors: HashMap<String, MLPacketDescriptor>,
    output_descriptors: HashMap<String, MLPacketDescriptor>,
    epsilon: f32,
}

impl L2NormalizeNode {
    pub fn new(
        name: impl Into<String>,
        input_name: impl Into<String>,
        output_name: impl Into<String>,
        descriptor: MLPacketDescriptor,
    ) -> Result<Self, String> {
        Self::with_epsilon(name, input_name, output_name, descriptor, 1e-12)
    }

    pub fn with_epsilon(
        name: impl Into<String>,
        input_name: impl Into<String>,
        output_name: impl Into<String>,
        descriptor: MLPacketDescriptor,
        epsilon: f32,
    ) -> Result<Self, String> {
        Self::with_epsilon_and_output_dtype(
            name,
            input_name,
            output_name,
            descriptor,
            epsilon,
            MLPacketDataType::Float32,
        )
    }

    pub fn with_output_dtype(
        name: impl Into<String>,
        input_name: impl Into<String>,
        output_name: impl Into<String>,
        descriptor: MLPacketDescriptor,
        output_dtype: MLPacketDataType,
    ) -> Result<Self, String> {
        Self::with_epsilon_and_output_dtype(
            name,
            input_name,
            output_name,
            descriptor,
            1e-12,
            output_dtype,
        )
    }

    pub fn with_epsilon_and_output_dtype(
        name: impl Into<String>,
        input_name: impl Into<String>,
        output_name: impl Into<String>,
        descriptor: MLPacketDescriptor,
        epsilon: f32,
        output_dtype: MLPacketDataType,
    ) -> Result<Self, String> {
        if !matches!(
            descriptor.dtype,
            MLPacketDataType::Float32 | MLPacketDataType::Float16
        ) {
            return Err(format!(
                "L2NormalizeNode only supports Float32 or Float16 input, got {:?}",
                descriptor.dtype
            ));
        }
        if descriptor.shape.is_empty() {
            return Err("L2NormalizeNode requires at least one tensor dimension".to_owned());
        }
        let last_dim = *descriptor
            .shape
            .last()
            .expect("empty shape should be rejected");
        if last_dim == 0 {
            return Err("L2NormalizeNode last tensor dimension must be non-zero".to_owned());
        }
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err("L2NormalizeNode epsilon must be finite and non-negative".to_owned());
        }
        if !matches!(
            output_dtype,
            MLPacketDataType::Float32 | MLPacketDataType::Float16
        ) {
            return Err(format!(
                "L2NormalizeNode only supports Float32 or Float16 output, got {output_dtype:?}"
            ));
        }

        let input_name = input_name.into();
        let output_name = output_name.into();

        let output_descriptor = MLPacketDescriptor {
            dtype: output_dtype,
            ..descriptor.clone()
        };

        Ok(Self {
            name: name.into(),
            input_name: input_name.clone(),
            output_name: output_name.clone(),
            input_descriptors: HashMap::from([(input_name, descriptor.clone())]),
            output_descriptors: HashMap::from([(output_name, output_descriptor)]),
            epsilon,
        })
    }
}

#[async_trait]
impl MLNode for L2NormalizeNode {
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

        let mut values = match payload.to_host_tensor()? {
            HostTensor::Float32(values) => values,
            HostTensor::Float16(values) => values.into_iter().map(f16::to_f32).collect(),
            other => {
                return Err(format!(
                    "L2NormalizeNode expected Float32 or Float16 host tensor, got {other:?}"
                ));
            }
        };

        normalize_rows(
            &mut values,
            *actual_desc.shape.last().expect("shape is non-empty"),
            self.epsilon,
        )?;

        let mut output_desc = self
            .output_descriptors
            .get(&self.output_name)
            .expect("output descriptor should exist")
            .clone();
        output_desc.shape = actual_desc.shape;
        output_desc.dynamic_batch = false;
        output_desc.dynamic_axes = vec![false; output_desc.shape.len()];

        let output_tensor = match output_desc.dtype {
            MLPacketDataType::Float32 => HostTensor::Float32(values),
            MLPacketDataType::Float16 => {
                HostTensor::Float16(values.into_iter().map(f16::from_f32).collect())
            }
            other => {
                return Err(format!(
                    "L2NormalizeNode cannot emit unsupported dtype {other:?}"
                ));
            }
        };
        let output_packet = MLPacket::from_host_tensor(context, output_desc, output_tensor)?;

        Ok(HashMap::from([(self.output_name.clone(), output_packet)]))
    }
}

fn normalize_rows(values: &mut [f32], row_width: usize, epsilon: f32) -> Result<(), String> {
    if row_width == 0 {
        return Err("row width must be non-zero".to_owned());
    }
    if values.len() % row_width != 0 {
        return Err(format!(
            "embedding element count {} is not divisible by row width {}",
            values.len(),
            row_width
        ));
    }

    for row in values.chunks_mut(row_width) {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm <= epsilon {
            continue;
        }

        for value in row {
            *value /= norm;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use half::f16;
    use lumnn::core::{
        context::{MLContext, MLContextOptions},
        node::MLNode,
        packet::{HostTensor, MLPacketDataType, MLPacketDescriptor},
    };

    use super::{L2NormalizeNode, normalize_rows};

    #[test]
    fn normalize_rows_normalizes_each_row() {
        let mut values = vec![3.0, 4.0, 5.0, 12.0];

        normalize_rows(&mut values, 2, 1e-12).unwrap();

        assert_close(values[0], 0.6);
        assert_close(values[1], 0.8);
        assert_close(values[2], 5.0 / 13.0);
        assert_close(values[3], 12.0 / 13.0);
    }

    #[test]
    fn normalize_rows_keeps_zero_rows_zero() {
        let mut values = vec![0.0, 0.0];

        normalize_rows(&mut values, 2, 1e-12).unwrap();

        assert_eq!(values, vec![0.0, 0.0]);
    }

    #[tokio::test]
    async fn node_accepts_float16_input_and_emits_float32_by_default() {
        let context = MLContext::new(MLContextOptions::default()).unwrap();
        let descriptor = MLPacketDescriptor::new(MLPacketDataType::Float16, vec![1, 2]);
        let node = L2NormalizeNode::new("normalize", "input", "output", descriptor.clone())
            .expect("node builds");
        let packet = context
            .packet_from_host_tensor(
                descriptor,
                HostTensor::Float16(vec![f16::from_f32(3.0), f16::from_f32(4.0)]),
            )
            .unwrap();

        let mut outputs = node
            .execute(
                HashMap::from([("input".to_owned(), packet)]),
                context.as_ref(),
            )
            .await
            .unwrap();
        let packet = outputs.remove("output").unwrap();

        assert_eq!(packet.descriptor.dtype, MLPacketDataType::Float32);
        assert_eq!(packet.descriptor.shape, vec![1, 2]);
        assert!(!packet.descriptor.dynamic_batch);
        match packet.to_host_tensor().await.unwrap() {
            HostTensor::Float32(values) => {
                assert_close(values[0], 0.6);
                assert_close(values[1], 0.8);
            }
            other => panic!("unexpected tensor: {other:?}"),
        }
    }

    #[tokio::test]
    async fn node_can_emit_float16_when_configured() {
        let context = MLContext::new(MLContextOptions::default()).unwrap();
        let descriptor = MLPacketDescriptor::new(MLPacketDataType::Float16, vec![1, 2]);
        let node = L2NormalizeNode::with_output_dtype(
            "normalize",
            "input",
            "output",
            descriptor.clone(),
            MLPacketDataType::Float16,
        )
        .expect("node builds");
        let packet = context
            .packet_from_host_tensor(
                descriptor,
                HostTensor::Float16(vec![f16::from_f32(3.0), f16::from_f32(4.0)]),
            )
            .unwrap();

        let mut outputs = node
            .execute(
                HashMap::from([("input".to_owned(), packet)]),
                context.as_ref(),
            )
            .await
            .unwrap();
        let packet = outputs.remove("output").unwrap();

        assert_eq!(packet.descriptor.dtype, MLPacketDataType::Float16);
        assert_eq!(packet.descriptor.shape, vec![1, 2]);
        match packet.to_host_tensor().await.unwrap() {
            HostTensor::Float16(values) => {
                assert_close_f16(values[0], 0.6);
                assert_close_f16(values[1], 0.8);
            }
            other => panic!("unexpected tensor: {other:?}"),
        }
    }

    #[tokio::test]
    async fn node_uses_actual_output_shape_for_dynamic_batch_inputs() {
        let context = MLContext::new(MLContextOptions::default()).unwrap();
        let expected_descriptor =
            MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 2]).with_dynamic_batch();
        let node = L2NormalizeNode::new("normalize", "input", "output", expected_descriptor)
            .expect("node builds");
        let actual_descriptor = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![2, 2]);
        let packet = context
            .packet_from_host_tensor(
                actual_descriptor,
                HostTensor::Float32(vec![3.0, 4.0, 5.0, 12.0]),
            )
            .unwrap();

        let mut outputs = node
            .execute(
                HashMap::from([("input".to_owned(), packet)]),
                context.as_ref(),
            )
            .await
            .unwrap();
        let packet = outputs.remove("output").unwrap();

        assert_eq!(packet.descriptor.dtype, MLPacketDataType::Float32);
        assert_eq!(packet.descriptor.shape, vec![2, 2]);
        assert!(!packet.descriptor.dynamic_batch);
        match packet.to_host_tensor().await.unwrap() {
            HostTensor::Float32(values) => {
                assert_close(values[0], 0.6);
                assert_close(values[1], 0.8);
                assert_close(values[2], 5.0 / 13.0);
                assert_close(values[3], 12.0 / 13.0);
            }
            other => panic!("unexpected tensor: {other:?}"),
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_close_f16(actual: f16, expected: f32) {
        let actual = actual.to_f32();
        assert!(
            (actual - expected).abs() < 5e-4,
            "expected {expected}, got {actual}"
        );
    }
}
