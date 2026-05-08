use std::sync::Arc;

use lumnn::core::{context::MLContext, node::MLNode, pipeline::MLPipeline};

use super::L2NormalizeNode;

/// Builds a CLIP embedding pipeline from a replaceable model-forward node.
///
/// The forward node can be `OrtNode`, a future `RknnNode`, or any other `MLNode`
/// that exposes a Float32 or Float16 embedding output. The pipeline appends L2
/// normalization and returns normalized Float32 embeddings under
/// `embedding_output_name`.
pub fn build_embedding_pipeline(
    name: impl Into<String>,
    context: Arc<MLContext>,
    forward_node: Box<dyn MLNode>,
    forward_output_name: impl Into<String>,
    embedding_output_name: impl Into<String>,
) -> Result<MLPipeline, String> {
    let forward_output_name = forward_output_name.into();
    let embedding_output_name = embedding_output_name.into();
    let embedding_desc = forward_node
        .output_descriptors()
        .get(&forward_output_name)
        .cloned()
        .ok_or_else(|| {
            format!(
                "forward node `{}` does not expose output `{forward_output_name}`",
                forward_node.name()
            )
        })?;

    let normalize = L2NormalizeNode::new(
        "l2_normalize",
        forward_output_name,
        embedding_output_name,
        embedding_desc,
    )?;

    MLPipeline::builder(name, context)
        .then_boxed(forward_node)
        .then(normalize)
        .build()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use async_trait::async_trait;
    use half::f16;
    use lumnn::core::{
        context::{MLContext, MLContextOptions},
        packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor},
    };

    use super::*;

    struct MockForwardNode {
        input_name: String,
        output_name: String,
        input_descriptors: HashMap<String, MLPacketDescriptor>,
        output_descriptors: HashMap<String, MLPacketDescriptor>,
    }

    impl MockForwardNode {
        fn new(input_name: &str, output_name: &str, descriptor: MLPacketDescriptor) -> Self {
            Self {
                input_name: input_name.to_owned(),
                output_name: output_name.to_owned(),
                input_descriptors: HashMap::from([(input_name.to_owned(), descriptor.clone())]),
                output_descriptors: HashMap::from([(output_name.to_owned(), descriptor)]),
            }
        }
    }

    #[async_trait]
    impl MLNode for MockForwardNode {
        fn name(&self) -> &str {
            "mock_forward"
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
            let (context, descriptor, _payload) = packet.into_parts()?;
            let tensor = match descriptor.dtype {
                MLPacketDataType::Float32 => HostTensor::Float32(vec![3.0, 4.0]),
                MLPacketDataType::Float16 => {
                    HostTensor::Float16(vec![f16::from_f32(3.0), f16::from_f32(4.0)])
                }
                other => return Err(format!("unsupported mock dtype {other:?}")),
            };
            let output = MLPacket::from_host_tensor(context, descriptor, tensor)?;

            Ok(HashMap::from([(self.output_name.clone(), output)]))
        }
    }

    #[tokio::test]
    async fn embedding_pipeline_runs_forward_then_normalize() {
        let context = MLContext::new(MLContextOptions::default()).unwrap();
        let descriptor = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 2]);
        let forward = MockForwardNode::new("ready_tensor", "raw_embedding", descriptor.clone());

        let pipeline = build_embedding_pipeline(
            "clip_text_embed",
            context.clone(),
            Box::new(forward),
            "raw_embedding",
            "embedding",
        )
        .unwrap();
        let input_packet = context
            .packet_from_host_tensor(descriptor, HostTensor::Float32(vec![0.0, 0.0]))
            .unwrap();

        let mut outputs = pipeline
            .run(HashMap::from([("ready_tensor".to_owned(), input_packet)]))
            .await
            .unwrap();
        let embedding = outputs.remove("embedding").unwrap();

        match embedding.to_host_tensor().await.unwrap() {
            HostTensor::Float32(values) => {
                assert!((values[0] - 0.6).abs() < 1e-6);
                assert!((values[1] - 0.8).abs() < 1e-6);
            }
            other => panic!("unexpected tensor: {other:?}"),
        }
    }

    #[tokio::test]
    async fn embedding_pipeline_accepts_float16_forward_output() {
        let context = MLContext::new(MLContextOptions::default()).unwrap();
        let descriptor = MLPacketDescriptor::new(MLPacketDataType::Float16, vec![1, 2]);
        let forward = MockForwardNode::new("ready_tensor", "raw_embedding", descriptor.clone());

        let pipeline = build_embedding_pipeline(
            "clip_text_embed",
            context.clone(),
            Box::new(forward),
            "raw_embedding",
            "embedding",
        )
        .unwrap();
        let input_packet = context
            .packet_from_host_tensor(
                descriptor,
                HostTensor::Float16(vec![f16::from_f32(0.0), f16::from_f32(0.0)]),
            )
            .unwrap();

        let mut outputs = pipeline
            .run(HashMap::from([("ready_tensor".to_owned(), input_packet)]))
            .await
            .unwrap();
        let embedding = outputs.remove("embedding").unwrap();

        assert_eq!(embedding.descriptor.dtype, MLPacketDataType::Float32);
        match embedding.to_host_tensor().await.unwrap() {
            HostTensor::Float32(values) => {
                assert!((values[0] - 0.6).abs() < 1e-6);
                assert!((values[1] - 0.8).abs() < 1e-6);
            }
            other => panic!("unexpected tensor: {other:?}"),
        }
    }
}
