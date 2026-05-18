use lumnn::core::{
    context::{MLContext, MLContextOptions},
    node::MLNode,
    packet::{HostTensor, MLPacketDataType},
};
use lumnn::mnn::MnnNode;
use std::collections::HashMap;
use std::time::Instant;

/// Loads the PP-OCRv5 recognition model and runs inference with a dummy input.
/// Verifies that MnnNode loads correctly, produces output, and returns Float32 data.
#[tokio::test]
async fn test_mnn_node_executes_ppocr_recognition() {
    let ctx = MLContext::new(MLContextOptions::accelerated()).expect("MLContext should initialize");

    let model_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../out/pp-ocrv5/mnn/recognition.fp32.mnn"
    );

    let node = MnnNode::new(&ctx, model_path, "ppocr_rec".to_string())
        .expect("MnnNode should load recognition model");

    // ── Inspect descriptors ──
    let input_name = node
        .input_descriptors()
        .keys()
        .next()
        .expect("should have input descriptor")
        .clone();
    let output_name = node
        .output_descriptors()
        .keys()
        .next()
        .expect("should have output descriptor")
        .clone();

    let input_desc = node
        .input_descriptors()
        .get(&input_name)
        .expect("should have input descriptor");
    assert_eq!(input_desc.dtype, MLPacketDataType::Float32);
    assert!(
        input_desc.shape.iter().product::<usize>() > 0,
        "input shape should have positive element count"
    );

    let output_desc = node
        .output_descriptors()
        .get(&output_name)
        .expect("should have output descriptor");
    assert_eq!(output_desc.dtype, MLPacketDataType::Float32);

    println!(
        "Recognition model input shape: {:?}, dynamic: {:?}",
        input_desc.shape, input_desc.dynamic_axes
    );
    println!(
        "Recognition model output shape: {:?}, dynamic: {:?}",
        output_desc.shape, output_desc.dynamic_axes
    );

    // ── Create dummy input (recognition expects NCHW float image) ──
    let input_elements: usize = input_desc.shape.iter().product();
    let dummy_input: Vec<f32> = vec![0.5f32; input_elements];

    let mut inputs = HashMap::new();
    inputs.insert(
        input_name.clone(),
        ctx.packet_from_f32(input_desc.shape.clone(), dummy_input)
            .expect("should create input packet"),
    );

    // ── Execute ──
    let execute_start = Instant::now();

    let mut outputs = node
        .execute(inputs, &ctx)
        .await
        .expect("MnnNode execution should succeed");

    let execute_latency = execute_start.elapsed();

    println!(
        "MnnNode execute latency: {:.3} ms",
        execute_latency.as_secs_f64() * 1000.0
    );

    // ── Check which backend was actually used ──
    let backend_id = node.backend_type().expect("should query backend");
    let backend_name = match backend_id {
        0 => "CPU",
        1 => "Metal",
        2 => "CUDA",
        3 => "OpenCL",
        5 => "CoreML",
        6 => "OpenGL",
        7 => "Vulkan",
        _ => "Unknown",
    };
    println!("Backend used: {} (id={})", backend_name, backend_id);

    let output = outputs
        .remove(&output_name)
        .expect("should produce output packet");

    // ── Verify output ──
    let readback_start = Instant::now();

    let host_tensor = output
        .to_host_tensor()
        .await
        .expect("output should be readable");

    let readback_latency = readback_start.elapsed();

    println!(
        "Output readback latency: {:.3} ms",
        readback_latency.as_secs_f64() * 1000.0
    );

    match host_tensor {
        HostTensor::Float32(values) => {
            println!(
                "Output shape: {:?}, element count: {}",
                output.descriptor.shape,
                values.len()
            );
            // At least some non-zero elements expected (softmax logits)
            assert!(!values.is_empty(), "output should not be empty");
        }
        other => panic!("expected Float32 output, got {:?}", other.dtype()),
    }
}
