use lumnn::core::{
    context::{MLContext, MLContextOptions},
    node::MLNode,
    packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor, RuntimeType},
};
use lumnn::ort::node::OrtNode;
use ort::memory::{AllocationDevice, AllocatorType, MemoryType};
use prost::Message;
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

const ONNX_TENSOR_TYPE_FLOAT: i32 = 1;
const ONNX_IR_VERSION: i64 = 8;
const ONNX_OPSET_VERSION: i64 = 13;
static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(0);

/// Tests that OrtNode can execute a tiny ONNX model performing matrix multiplication and addition.
#[tokio::test]
async fn test_ort_node_executes_tiny_matmul_add_model() {
    let context = MLContext::new(MLContextOptions::default()).expect("context should initialize");

    // Create a temporary ONNX model file
    let model_path = write_tiny_matmul_add_model();
    // Load the model into an OrtNode
    let node = OrtNode::new(
        context.as_ref(),
        model_path
            .to_str()
            .expect("temporary ONNX path should be valid UTF-8"),
        "tiny_mlp".to_string(),
    )
    .expect("OrtNode should load tiny ONNX model");

    let input_desc = node
        .input_descriptors()
        .get("X")
        .expect("input descriptor should exist");
    assert_eq!(input_desc.dtype, MLPacketDataType::Float32);
    assert_eq!(input_desc.shape, vec![1, 2]);

    let output_desc = node
        .output_descriptors()
        .get("Y")
        .expect("output descriptor should exist");
    assert_eq!(output_desc.dtype, MLPacketDataType::Float32);
    assert_eq!(output_desc.shape, vec![1, 2]);

    let mut inputs = HashMap::new();
    inputs.insert(
        "X".to_string(),
        context
            .packet_from_host_tensor(
                MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 2]),
                HostTensor::Float32(vec![1.0, 2.0]),
            )
            .expect("input packet should be created"),
    );

    let mut outputs = node
        .execute(inputs, context.as_ref())
        .await
        .expect("OrtNode execution should succeed");
    let output = outputs
        .remove("Y")
        .expect("output tensor `Y` should be produced");
    // Check the output runtime and values
    assert!(matches!(
        output.runtime().await.expect("runtime should be readable"),
        RuntimeType::Backend { backend: "ort", .. }
    ));

    let values = extract_f32_values(output).await;
    assert_float_vec_eq(&values, &[7.5, 9.0]);

    let _ = fs::remove_file(model_path);
}

#[tokio::test]
async fn test_ort_node_accepts_ort_payload_from_upstream_node() {
    let context = MLContext::new(MLContextOptions::default()).expect("context should initialize");

    // Create a temporary ONNX model file
    let model_path = write_tiny_matmul_add_model();
    let model_path_str = model_path
        .to_str()
        .expect("temporary ONNX path should be valid UTF-8");
    let first = OrtNode::new(context.as_ref(), model_path_str, "first".to_string())
        .expect("first OrtNode should load tiny ONNX model");
    let second = OrtNode::new(context.as_ref(), model_path_str, "second".to_string())
        .expect("second OrtNode should load tiny ONNX model");

    let mut initial_inputs = HashMap::new();
    initial_inputs.insert(
        "X".to_string(),
        context
            .packet_from_host_tensor(
                MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 2]),
                HostTensor::Float32(vec![1.0, 2.0]),
            )
            .expect("input packet should be created"),
    );

    let mut first_outputs = first
        .execute(initial_inputs, context.as_ref())
        .await
        .expect("first OrtNode execution should succeed");
    // Get the output from first node
    let first_output = first_outputs
        .remove("Y")
        .expect("first node should produce `Y`");
    // Verify it's Ort runtime
    assert!(matches!(
        first_output
            .runtime()
            .await
            .expect("runtime should be readable"),
        RuntimeType::Backend { backend: "ort", .. }
    ));

    let mut second_inputs = HashMap::new();
    second_inputs.insert("X".to_string(), first_output);
    // Execute the second node
    let mut second_outputs = second
        .execute(second_inputs, context.as_ref())
        .await
        .expect("second OrtNode execution should succeed");
    // Get the final output
    let final_output = second_outputs
        .remove("Y")
        .expect("second node should produce `Y`");

    let values = extract_f32_values(final_output).await;
    assert_float_vec_eq(&values, &[35.0, 50.0]);

    let _ = fs::remove_file(model_path);
}

#[tokio::test]
async fn test_ort_node_can_execute_with_io_binding_outputs() {
    let context = MLContext::new(MLContextOptions::default()).expect("context should initialize");
    let model_path = write_tiny_matmul_add_model();
    let node = OrtNode::new(
        context.as_ref(),
        model_path
            .to_str()
            .expect("temporary ONNX path should be valid UTF-8"),
        "tiny_mlp_iobinding".to_string(),
    )
    .expect("OrtNode should load tiny ONNX model")
    .with_io_binding_outputs_to_device(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::CPUOutput,
    );

    let mut inputs = HashMap::new();
    inputs.insert(
        "X".to_string(),
        context
            .packet_from_host_tensor(
                MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 2]),
                HostTensor::Float32(vec![1.0, 2.0]),
            )
            .expect("input packet should be created"),
    );

    let mut outputs = node
        .execute(inputs, context.as_ref())
        .await
        .expect("OrtNode I/O binding execution should succeed");
    let output = outputs
        .remove("Y")
        .expect("output tensor `Y` should be produced");

    assert!(matches!(
        output.runtime().await.expect("runtime should be readable"),
        RuntimeType::Backend { backend: "ort", .. }
    ));
    let values = extract_f32_values(output).await;
    assert_float_vec_eq(&values, &[7.5, 9.0]);

    let _ = fs::remove_file(model_path);
}

/// Extracts f32 values from an MLPacket, panicking if not Float32.
async fn extract_f32_values(packet: MLPacket) -> Vec<f32> {
    match packet
        .to_host_tensor()
        .await
        .expect("packet should be convertible to host tensor")
    {
        HostTensor::Float32(values) => values,
        other => panic!("expected Float32 host tensor, got {other:?}"),
    }
}

/// Asserts that two float vectors are approximately equal within a tolerance.
fn assert_float_vec_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "tensor length mismatch: actual={actual:?}, expected={expected:?}"
    );

    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta <= 1e-5,
            "value mismatch at index {index}: actual={actual}, expected={expected}, delta={delta}"
        );
    }
}

/// Creates and writes a tiny ONNX model for matrix multiplication and addition to a temporary file.
fn write_tiny_matmul_add_model() -> PathBuf {
    let model = ModelProto {
        ir_version: ONNX_IR_VERSION,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: ONNX_OPSET_VERSION,
        }],
        graph: Some(GraphProto {
            node: vec![
                NodeProto {
                    input: vec!["X".to_string(), "W".to_string()],
                    output: vec!["MM".to_string()],
                    name: "matmul".to_string(),
                    op_type: "MatMul".to_string(),
                },
                NodeProto {
                    input: vec!["MM".to_string(), "B".to_string()],
                    output: vec!["Y".to_string()],
                    name: "bias_add".to_string(),
                    op_type: "Add".to_string(),
                },
            ],
            name: "tiny_matmul_add".to_string(),
            initializer: vec![
                TensorProto {
                    dims: vec![2, 2],
                    data_type: ONNX_TENSOR_TYPE_FLOAT,
                    float_data: vec![1.0, 2.0, 3.0, 4.0],
                    name: "W".to_string(),
                },
                TensorProto {
                    dims: vec![1, 2],
                    data_type: ONNX_TENSOR_TYPE_FLOAT,
                    float_data: vec![0.5, -1.0],
                    name: "B".to_string(),
                },
            ],
            input: vec![make_tensor_value_info("X", &[1, 2])],
            output: vec![make_tensor_value_info("Y", &[1, 2])],
        }),
    };

    let path = std::env::temp_dir().join(format!(
        "lumnn-test-tiny-matmul-add-{}-{}-{}.onnx",
        std::process::id(),
        NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after UNIX_EPOCH")
            .as_nanos()
    ));
    fs::write(&path, model.encode_to_vec()).expect("temporary ONNX model should be written");
    path
}

/// Creates a ValueInfoProto for a tensor with the given name and shape.
fn make_tensor_value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(type_proto::Tensor {
                elem_type: ONNX_TENSOR_TYPE_FLOAT,
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .copied()
                        .map(|dim| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(dim)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

/// Represents an ONNX ModelProto.
#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(int64, tag = "1")]
    ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    opset_import: Vec<OperatorSetIdProto>,
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
}

/// Represents an ONNX OperatorSetIdProto.
#[derive(Clone, PartialEq, Message)]
struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    domain: String,
    #[prost(int64, tag = "2")]
    version: i64,
}

/// Represents an ONNX GraphProto.
#[derive(Clone, PartialEq, Message)]
struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    name: String,
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    output: Vec<ValueInfoProto>,
}

/// Represents an ONNX NodeProto.
#[derive(Clone, PartialEq, Message)]
struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    output: Vec<String>,
    #[prost(string, tag = "3")]
    name: String,
    #[prost(string, tag = "4")]
    op_type: String,
}

/// Represents an ONNX TensorProto.
#[derive(Clone, PartialEq, Message)]
struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    data_type: i32,
    #[prost(float, repeated, tag = "4")]
    float_data: Vec<f32>,
    #[prost(string, tag = "8")]
    name: String,
}

/// Represents an ONNX ValueInfoProto.
#[derive(Clone, PartialEq, Message)]
struct ValueInfoProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, optional, tag = "2")]
    r#type: Option<TypeProto>,
}

/// Represents an ONNX TypeProto.
#[derive(Clone, PartialEq, Message)]
struct TypeProto {
    #[prost(message, optional, tag = "1")]
    tensor_type: Option<type_proto::Tensor>,
}

mod type_proto {
    use super::TensorShapeProto;
    use prost::Message;

    /// Represents an ONNX TypeProto Tensor.
    #[derive(Clone, PartialEq, Message)]
    pub struct Tensor {
        #[prost(int32, tag = "1")]
        pub elem_type: i32,
        #[prost(message, optional, tag = "2")]
        pub shape: Option<TensorShapeProto>,
    }
}

/// Represents an ONNX TensorShapeProto.
#[derive(Clone, PartialEq, Message)]
struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    dim: Vec<tensor_shape_proto::Dimension>,
}

mod tensor_shape_proto {
    use prost::Message;

    /// Represents an ONNX TensorShapeProto Dimension.
    #[derive(Clone, PartialEq, Message)]
    pub struct Dimension {
        #[prost(oneof = "dimension::Value", tags = "1, 2")]
        pub value: Option<dimension::Value>,
    }

    pub mod dimension {
        use prost::Oneof;

        #[derive(Clone, PartialEq, Oneof)]
        pub enum Value {
            #[prost(int64, tag = "1")]
            DimValue(i64),
            #[prost(string, tag = "2")]
            DimParam(String),
        }
    }
}
