use async_trait::async_trait;
use lumnn::core::{
    context::{MLContext, MLContextOptions},
    node::MLNode,
    packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor, RuntimeType},
    pipeline::MLPipeline,
};
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

type TransformFn = Box<dyn Fn(Vec<f32>) -> Vec<f32> + Send + Sync>;

struct MockNode {
    name: String,
    input_name: String,
    output_name: String,
    input_descriptors: HashMap<String, MLPacketDescriptor>,
    output_descriptors: HashMap<String, MLPacketDescriptor>,
    transform: TransformFn,
}

impl MockNode {
    fn new(
        name: impl Into<String>,
        input_name: impl Into<String>,
        output_name: impl Into<String>,
        descriptor: MLPacketDescriptor,
        transform: TransformFn,
    ) -> Self {
        let input_name = input_name.into();
        let output_name = output_name.into();

        Self {
            name: name.into(),
            input_name: input_name.clone(),
            output_name: output_name.clone(),
            input_descriptors: HashMap::from([(input_name, descriptor.clone())]),
            output_descriptors: HashMap::from([(output_name, descriptor)]),
            transform,
        }
    }
}

#[async_trait]
impl MLNode for MockNode {
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
            return Err("unexpected extra inputs".to_string());
        }

        let (context, descriptor, payload) = packet.into_parts()?;
        assert_eq!(payload.runtime(), RuntimeType::Cpu);

        let values = match payload.to_host_tensor()? {
            HostTensor::Float32(values) => values,
            other => return Err(format!("expected Float32 host tensor, got {other:?}")),
        };

        let output_values = (self.transform)(values);
        let output_packet =
            MLPacket::from_host_tensor(context, descriptor, HostTensor::Float32(output_values))?;

        Ok(HashMap::from([(self.output_name.clone(), output_packet)]))
    }
}

struct SharedCounterNode {
    name: String,
    input_descriptors: HashMap<String, MLPacketDescriptor>,
    output_descriptors: HashMap<String, MLPacketDescriptor>,
    executions: AtomicUsize,
}

impl SharedCounterNode {
    fn new(name: impl Into<String>, descriptor: MLPacketDescriptor) -> Self {
        Self {
            name: name.into(),
            input_descriptors: HashMap::from([("input".to_string(), descriptor.clone())]),
            output_descriptors: HashMap::from([("output".to_string(), descriptor)]),
            executions: AtomicUsize::new(0),
        }
    }

    fn execution_count(&self) -> usize {
        self.executions.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl MLNode for SharedCounterNode {
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
        self.executions.fetch_add(1, Ordering::SeqCst);

        let packet = inputs
            .remove("input")
            .ok_or_else(|| "missing required input `input`".to_string())?;

        if !inputs.is_empty() {
            return Err("unexpected extra inputs".to_string());
        }

        let host_tensor = packet.to_host_tensor().await?;
        let (context, _descriptor, _payload) = packet.into_parts()?;
        let output_descriptor = self
            .output_descriptors
            .get("output")
            .expect("output descriptor should exist")
            .clone();
        let output_packet = MLPacket::from_host_tensor(context, output_descriptor, host_tensor)?;

        Ok(HashMap::from([("output".to_string(), output_packet)]))
    }
}

#[tokio::test]
async fn test_linear_pipeline_runs_nodes_in_order() {
    let context = MLContext::new(MLContextOptions::default()).expect("context should initialize");
    let descriptor = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![2, 2]);

    let node_a = MockNode::new(
        "add_ten",
        "image",
        "features",
        descriptor.clone(),
        Box::new(|values| values.into_iter().map(|value| value + 10.0).collect()),
    );
    let node_b = MockNode::new(
        "identity",
        "features",
        "result",
        descriptor.clone(),
        Box::new(|values| values),
    );
    let pipeline = MLPipeline::builder("linear_pipeline", context.clone())
        .then(node_a)
        .then(node_b)
        .build()
        .expect("pipeline should build");

    let initial_packet = context
        .packet_from_host_tensor(descriptor, HostTensor::Float32(vec![1.0, 2.0, 3.0, 4.0]))
        .expect("initial packet creation should succeed");

    let mut outputs = pipeline
        .run(HashMap::from([("image".to_string(), initial_packet)]))
        .await
        .expect("pipeline should run");
    let final_out = outputs.remove("result").expect("result should exist");

    assert_eq!(
        final_out
            .runtime()
            .await
            .expect("final packet should have runtime"),
        RuntimeType::Cpu
    );

    match final_out
        .to_host_tensor()
        .await
        .expect("final packet should materialize to host")
    {
        HostTensor::Float32(values) => assert_eq!(values, vec![11.0, 12.0, 13.0, 14.0]),
        other => panic!("unexpected final host tensor variant: {other:?}"),
    }
}

#[tokio::test]
async fn test_shared_node_can_be_used_by_multiple_pipelines() {
    let context = MLContext::new(MLContextOptions::default()).expect("context should initialize");
    let descriptor = MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1]);
    let shared_node = Arc::new(SharedCounterNode::new("shared", descriptor.clone()));

    let pipeline_a = MLPipeline::builder("pipeline_a", context.clone())
        .then_shared(shared_node.clone())
        .build()
        .expect("pipeline A should build");
    let pipeline_b = MLPipeline::builder("pipeline_b", context.clone())
        .then_shared(shared_node.clone())
        .build()
        .expect("pipeline B should build");

    let input_a = context
        .packet_from_host_tensor(descriptor.clone(), HostTensor::Float32(vec![1.0]))
        .expect("input A packet should be created");
    let input_b = context
        .packet_from_host_tensor(descriptor, HostTensor::Float32(vec![2.0]))
        .expect("input B packet should be created");

    pipeline_a
        .run(HashMap::from([("input".to_string(), input_a)]))
        .await
        .expect("pipeline A should run");
    pipeline_b
        .run(HashMap::from([("input".to_string(), input_b)]))
        .await
        .expect("pipeline B should run");

    assert_eq!(shared_node.execution_count(), 2);
}

#[tokio::test]
async fn test_empty_pipeline_returns_error() {
    let context = MLContext::new(MLContextOptions::default()).expect("context should initialize");

    let error = match MLPipeline::builder("empty", context).build() {
        Ok(_) => panic!("empty pipeline should fail to build"),
        Err(error) => error,
    };

    assert!(error.contains("has no nodes"));
}
