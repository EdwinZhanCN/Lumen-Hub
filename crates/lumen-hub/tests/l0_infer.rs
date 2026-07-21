//! L0 inference suite: the data plane over real gRPC against the QA model —
//! raw and tensor inputs, request chunking, and error surfaces.
//!
//! Run: cargo test --features qa --test l0_infer

mod common;

use common::harness::{
    HubOptions, HubProcess, cosine, infer_embedding, infer_raw, parse_embedding, qa_pixels,
    qa_tensor_infer_request,
};
use lumen_hub::backend::default_device;
use lumen_hub::models::qa::{QA_INPUT_NUMEL, QaNet};

/// Reference embedding computed in-process with the same deterministic
/// weights the fixture ships.
fn reference_embedding(pixels: &[f32]) -> Vec<f32> {
    let device = default_device();
    QaNet::deterministic(&device).embed(pixels, 1, &device)
}

#[tokio::test]
async fn tensor_roundtrip_matches_in_process_reference() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    hub.wait_ready().await;
    let mut client = hub.inference().await;

    let pixels = qa_pixels(7);
    let over_grpc = infer_embedding(&mut client, qa_tensor_infer_request("ref", &pixels))
        .await
        .expect("tensor infer");
    let reference = reference_embedding(&pixels);

    let similarity = cosine(&over_grpc, &reference);
    assert!(
        similarity > 0.99999,
        "gRPC result must match the in-process forward, cosine {similarity}"
    );
}

#[tokio::test]
async fn raw_image_roundtrip() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    hub.wait_ready().await;
    let mut client = hub.inference().await;

    // Encode a synthetic 16x16 PNG in-memory.
    let mut png = Vec::new();
    let image = image::RgbImage::from_fn(16, 16, |x, y| {
        image::Rgb([(x * 16) as u8, (y * 16) as u8, ((x + y) * 8) as u8])
    });
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("png encode");

    let request = lumen_hub::daemon::proto::home_native::v1::InferRequest {
        correlation_id: "raw".to_owned(),
        task: lumen_hub::models::qa::QA_EMBED_TASK.to_owned(),
        payload: png,
        meta: std::collections::HashMap::from([("service".to_owned(), "qa".to_owned())]),
        payload_mime: "image/png".to_owned(),
        seq: 0,
        total: 1,
        offset: 0,
    };
    let embedding = infer_embedding(&mut client, request)
        .await
        .expect("raw infer");
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "norm {norm}");
}

#[tokio::test]
async fn chunked_request_reassembles_to_the_same_result() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    hub.wait_ready().await;
    let mut client = hub.inference().await;

    let pixels = qa_pixels(11);
    let whole = qa_tensor_infer_request("chunked", &pixels);
    let single = infer_embedding(&mut client, whole.clone())
        .await
        .expect("single");

    // Split the payload into three chunks with seq/total/offset set.
    let payload = whole.payload.clone();
    let cut_a = payload.len() / 3;
    let cut_b = 2 * payload.len() / 3;
    let parts = [
        (0u64, 0usize, &payload[..cut_a]),
        (1u64, cut_a, &payload[cut_a..cut_b]),
        (2u64, cut_b, &payload[cut_b..]),
    ];
    let requests: Vec<_> = parts
        .iter()
        .map(|(seq, offset, bytes)| {
            let mut message = whole.clone();
            message.payload = bytes.to_vec();
            message.seq = *seq;
            message.total = 3;
            message.offset = *offset as u64;
            message
        })
        .collect();

    let responses = infer_raw(&mut client, requests)
        .await
        .expect("chunked infer");
    assert_eq!(responses.len(), 1);
    let chunked = parse_embedding(&responses[0]).expect("chunked embedding");
    assert!(
        cosine(&single, &chunked) > 0.99999,
        "chunked reassembly must equal the single-message request"
    );
}

#[tokio::test]
async fn unknown_task_is_a_not_found_status() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    hub.wait_ready().await;
    let mut client = hub.inference().await;

    let mut request = qa_tensor_infer_request("nope", &qa_pixels(3));
    request.task = "no_such_task".to_owned();
    let error = infer_raw(&mut client, vec![request])
        .await
        .expect_err("unknown task must fail");
    assert_eq!(error.code(), tonic::Code::NotFound, "got: {error}");
}

#[tokio::test]
async fn tensor_payload_length_mismatch_is_invalid_argument() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    hub.wait_ready().await;
    let mut client = hub.inference().await;

    let mut request = qa_tensor_infer_request("short", &qa_pixels(4));
    request.payload.truncate(QA_INPUT_NUMEL); // 1/4 of the declared bytes
    let error = infer_raw(&mut client, vec![request])
        .await
        .expect_err("length mismatch must fail");
    assert_eq!(error.code(), tonic::Code::InvalidArgument, "got: {error}");
}
