use lumnn::core::{
    context::{MLContext, MLContextOptions},
    packet::{HostTensor, MLPacket, MLPacketDataType, MLPacketDescriptor, RuntimeType},
    pipeline::MLPipeline,
};
use lumnn::ndarray::node::NdArrayNode;
use lumnn::ort::node::OrtNode;
use prost::Message;
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

const ONNX_TENSOR_TYPE_FLOAT: i32 = 1;
const ONNX_IR_VERSION: i64 = 8;
const ONNX_OPSET_VERSION: i64 = 13;

#[tokio::test]
async fn test_ndarray_preprocess_then_ort_inference() {
    let context = MLContext::new(MLContextOptions::default()).expect("context should initialize");
    let model_path = write_tiny_conv_model();
    let model_path_str = model_path
        .to_str()
        .expect("temporary ONNX path should be valid UTF-8");

    let preprocess = NdArrayNode::new(
        "divide_by_two",
        "image",
        "X",
        MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 1, 2, 2]),
        MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 1, 2, 2]),
        Box::new(|view| Ok(view.mapv(|value| value / 2.0).into_dyn())),
    )
    .expect("preprocess node should be created");
    let infer = OrtNode::new(context.as_ref(), model_path_str, "tiny_conv".to_string())
        .expect("OrtNode should load tiny Conv model");
    let pipeline = MLPipeline::builder("ndarray_to_ort", context.clone())
        .then(preprocess)
        .then(infer)
        .build()
        .expect("pipeline should build");

    let input = context
        .packet_from_host_tensor(
            MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 1, 2, 2]),
            HostTensor::Float32(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .expect("input packet should be created");

    let mut outputs = pipeline
        .run(HashMap::from([("image".to_string(), input)]))
        .await
        .expect("pipeline should run");
    let output = outputs.remove("Y").expect("Conv output should exist");
    assert!(matches!(
        output.runtime().await.expect("runtime should be readable"),
        RuntimeType::Backend { backend: "ort", .. }
    ));

    let values = extract_f32_values(output).await;
    assert_float_vec_eq(&values, &[5.0]);

    let _ = fs::remove_file(model_path);
}

#[tokio::test]
async fn test_two_ort_nodes_chain_runtime_payload() {
    let context = MLContext::new(MLContextOptions::default()).expect("context should initialize");
    let first_model_path = write_tiny_matmul_add_model("input", "X");
    let second_model_path = write_tiny_matmul_add_model("X", "Y");
    let first_model_path_str = first_model_path
        .to_str()
        .expect("temporary ONNX path should be valid UTF-8");
    let second_model_path_str = second_model_path
        .to_str()
        .expect("temporary ONNX path should be valid UTF-8");

    let first = OrtNode::new(context.as_ref(), first_model_path_str, "first".to_string())
        .expect("first OrtNode should load tiny ONNX model");
    let second = OrtNode::new(
        context.as_ref(),
        second_model_path_str,
        "second".to_string(),
    )
    .expect("second OrtNode should load tiny ONNX model");
    let pipeline = MLPipeline::builder("ort_to_ort", context.clone())
        .then(first)
        .then(second)
        .build()
        .expect("pipeline should build");

    let input = context
        .packet_from_host_tensor(
            MLPacketDescriptor::new(MLPacketDataType::Float32, vec![1, 2]),
            HostTensor::Float32(vec![1.0, 2.0]),
        )
        .expect("input packet should be created");

    let mut outputs = pipeline
        .run(HashMap::from([("input".to_string(), input)]))
        .await
        .expect("pipeline should run");
    let output = outputs.remove("Y").expect("final output should exist");
    assert!(matches!(
        output.runtime().await.expect("runtime should be readable"),
        RuntimeType::Backend { backend: "ort", .. }
    ));

    let values = extract_f32_values(output).await;
    assert_float_vec_eq(&values, &[35.0, 50.0]);

    let _ = fs::remove_file(first_model_path);
    let _ = fs::remove_file(second_model_path);
}

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

fn write_tiny_conv_model() -> PathBuf {
    let model = ModelProto {
        ir_version: ONNX_IR_VERSION,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: ONNX_OPSET_VERSION,
        }],
        graph: Some(GraphProto {
            node: vec![NodeProto {
                input: vec!["X".to_string(), "W".to_string()],
                output: vec!["Y".to_string()],
                name: "conv".to_string(),
                op_type: "Conv".to_string(),
            }],
            name: "tiny_conv".to_string(),
            initializer: vec![TensorProto {
                dims: vec![1, 1, 2, 2],
                data_type: ONNX_TENSOR_TYPE_FLOAT,
                float_data: vec![1.0, 1.0, 1.0, 1.0],
                name: "W".to_string(),
            }],
            input: vec![make_tensor_value_info("X", &[1, 1, 2, 2])],
            output: vec![make_tensor_value_info("Y", &[1, 1, 1, 1])],
        }),
    };

    let path = std::env::temp_dir().join(format!(
        "lumnn-test-tiny-conv-{}-{}.onnx",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after UNIX_EPOCH")
            .as_nanos()
    ));
    fs::write(&path, model.encode_to_vec()).expect("temporary ONNX model should be written");
    path
}

fn write_tiny_matmul_add_model(input_name: &str, output_name: &str) -> PathBuf {
    let model = ModelProto {
        ir_version: ONNX_IR_VERSION,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: ONNX_OPSET_VERSION,
        }],
        graph: Some(GraphProto {
            node: vec![
                NodeProto {
                    input: vec![input_name.to_string(), "W".to_string()],
                    output: vec!["MM".to_string()],
                    name: "matmul".to_string(),
                    op_type: "MatMul".to_string(),
                },
                NodeProto {
                    input: vec!["MM".to_string(), "B".to_string()],
                    output: vec![output_name.to_string()],
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
            input: vec![make_tensor_value_info(input_name, &[1, 2])],
            output: vec![make_tensor_value_info(output_name, &[1, 2])],
        }),
    };

    let path = std::env::temp_dir().join(format!(
        "lumnn-test-tiny-matmul-add-{}-{}-{}-{}.onnx",
        input_name,
        output_name,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after UNIX_EPOCH")
            .as_nanos()
    ));
    fs::write(&path, model.encode_to_vec()).expect("temporary ONNX model should be written");
    path
}

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

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(int64, tag = "1")]
    ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    opset_import: Vec<OperatorSetIdProto>,
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
}

#[derive(Clone, PartialEq, Message)]
struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    domain: String,
    #[prost(int64, tag = "2")]
    version: i64,
}

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

#[derive(Clone, PartialEq, Message)]
struct ValueInfoProto {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(message, optional, tag = "2")]
    r#type: Option<TypeProto>,
}

#[derive(Clone, PartialEq, Message)]
struct TypeProto {
    #[prost(message, optional, tag = "1")]
    tensor_type: Option<type_proto::Tensor>,
}

mod type_proto {
    use super::TensorShapeProto;
    use prost::Message;

    #[derive(Clone, PartialEq, Message)]
    pub struct Tensor {
        #[prost(int32, tag = "1")]
        pub elem_type: i32,
        #[prost(message, optional, tag = "2")]
        pub shape: Option<TensorShapeProto>,
    }
}

#[derive(Clone, PartialEq, Message)]
struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    dim: Vec<tensor_shape_proto::Dimension>,
}

mod tensor_shape_proto {
    use prost::Message;

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
