use lumnn::core::{
    context::{MLContext, MLContextOptions},
    node::MLNode,
    packet::{HostTensor, MLPacketDataType, MLPacketDescriptor, RuntimeType},
};
use lumnn::ndarray::node::NdArrayNode;
use std::collections::HashMap;

#[tokio::test]
async fn test_ndarray_node_runs_on_cpu_runtime() {
    let context = MLContext::new(MLContextOptions::default()).expect("context should initialize");

    // Create an NdArrayNode that normalizes input by dividing each value by 2.
    let node = NdArrayNode::new(
        "normalize",
        "input",
        "output",
        MLPacketDescriptor::new(MLPacketDataType::Float32, vec![2, 2]),
        MLPacketDescriptor::new(MLPacketDataType::Float32, vec![2, 2]),
        Box::new(|view| Ok(view.mapv(|value| value / 2.0).into_dyn())),
    )
    .expect("node creation should succeed");

    // Create an input packet with values [2.0, 4.0, 6.0, 8.0]
    let packet = context
        .packet_from_host_tensor(
            MLPacketDescriptor::new(MLPacketDataType::Float32, vec![2, 2]),
            HostTensor::Float32(vec![2.0, 4.0, 6.0, 8.0]),
        )
        .expect("packet creation should succeed");

    // Execute the node with the input packet
    let outputs = node
        .execute(
            HashMap::from([("input".to_string(), packet)]),
            context.as_ref(),
        )
        .await
        .expect("ndarray node should execute successfully");

    // Retrieve the output packet
    let output = outputs.get("output").expect("output packet should exist");
    // Assert that the output runtime is CPU
    assert_eq!(
        output.runtime().await.expect("runtime should be available"),
        RuntimeType::Cpu
    );

    // Check the output values, which should be [1.0, 2.0, 3.0, 4.0] after normalization
    match output
        .to_host_tensor()
        .await
        .expect("output should materialize to host")
    {
        HostTensor::Float32(values) => assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]),
        other => panic!("unexpected output tensor: {other:?}"),
    }
}
