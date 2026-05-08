use candle_onnx::onnx::{
    self, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto,
    TypeProto, ValueInfoProto,
};
use lumnn::{
    candle::node::CandleOnnxNode,
    core::{
        context::{MLContext, MLContextOptions},
        node::MLNode,
        packet::{HostTensor, RuntimeType},
    },
};
use prost::Message;
use std::{collections::HashMap, fs, path::PathBuf};

const ONNX_TENSOR_TYPE_FLOAT: i32 = 1;
const ONNX_IR_VERSION: i64 = 8;
const ONNX_OPSET_VERSION: i64 = 13;

#[tokio::test]
async fn candle_onnx_node_executes_tiny_add_model() {
    let model_path = write_tiny_add_model();
    let model_path_str = model_path
        .to_str()
        .expect("temporary ONNX path should be valid UTF-8");
    let context = MLContext::new(MLContextOptions::default()).expect("context should initialize");
    let node = CandleOnnxNode::new(context.as_ref(), model_path_str, "tiny_add".to_owned())
        .expect("CandleOnnxNode should load tiny ONNX model");

    assert_eq!(node.device_runtime(), "cpu");
    assert!(node.input_descriptors().contains_key("X"));
    assert!(!node.input_descriptors().contains_key("B"));

    let input = context
        .packet_from_f32(vec![1, 3], vec![1.0, 2.0, 3.0])
        .expect("input packet should be created");
    let outputs = node
        .execute(HashMap::from([("X".to_owned(), input)]), context.as_ref())
        .await
        .expect("Candle ONNX execution should succeed");
    let output = outputs.get("Y").expect("Y output should exist");

    assert_eq!(
        output.runtime().await.expect("runtime should exist"),
        RuntimeType::Backend {
            backend: "candle",
            device: "cpu"
        }
    );
    match output
        .to_host_tensor()
        .await
        .expect("output should materialize")
    {
        HostTensor::Float32(values) => assert_eq!(values, vec![11.0, 22.0, 33.0]),
        other => panic!("unexpected output tensor: {other:?}"),
    }

    let _ = fs::remove_file(model_path);
}

fn write_tiny_add_model() -> PathBuf {
    let model = ModelProto {
        ir_version: ONNX_IR_VERSION,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: ONNX_OPSET_VERSION,
        }],
        producer_name: "lumnn-test".to_owned(),
        graph: Some(GraphProto {
            name: "tiny_add".to_owned(),
            node: vec![NodeProto {
                input: vec!["X".to_owned(), "B".to_owned()],
                output: vec!["Y".to_owned()],
                op_type: "Add".to_owned(),
                ..Default::default()
            }],
            initializer: vec![TensorProto {
                name: "B".to_owned(),
                dims: vec![1, 3],
                data_type: ONNX_TENSOR_TYPE_FLOAT,
                float_data: vec![10.0, 20.0, 30.0],
                ..Default::default()
            }],
            input: vec![
                value_info("X", ONNX_TENSOR_TYPE_FLOAT, &[1, 3]),
                value_info("B", ONNX_TENSOR_TYPE_FLOAT, &[1, 3]),
            ],
            output: vec![value_info("Y", ONNX_TENSOR_TYPE_FLOAT, &[1, 3])],
            ..Default::default()
        }),
        ..Default::default()
    };

    let path = std::env::temp_dir().join(format!(
        "lumnn-test-candle-tiny-add-{}-{}.onnx",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos()
    ));
    fs::write(&path, model.encode_to_vec()).expect("temporary ONNX model should be written");
    path
}

fn value_info(name: &str, elem_type: i32, dims: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_owned(),
        r#type: Some(TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(
                onnx::type_proto::Tensor {
                    elem_type,
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|dim| onnx::tensor_shape_proto::Dimension {
                                value: Some(onnx::tensor_shape_proto::dimension::Value::DimValue(
                                    *dim,
                                )),
                                ..Default::default()
                            })
                            .collect(),
                    }),
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    }
}
