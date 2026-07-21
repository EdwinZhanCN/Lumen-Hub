//! L0 batcher suite: dynamic batching over real gRPC — composition limits and,
//! critically, result integrity (batched results must equal individual ones;
//! no cross-request contamination).
//!
//! Run: cargo test --features qa --test l0_batcher

mod common;

use common::harness::{
    HubOptions, HubProcess, cosine, infer_raw, parse_embedding, qa_pixels, qa_tensor_infer_request,
};
use lumen_hub::backend::default_device;
use lumen_hub::models::qa::QaNet;

/// Fires `count` concurrent single-request Infer streams and returns
/// `(embedding, reported_batch_size)` per request, in request order.
async fn concurrent_infer(hub: &HubProcess, count: usize) -> Vec<(Vec<f32>, Option<usize>)> {
    let mut handles = Vec::new();
    for index in 0..count {
        let mut client = hub.inference().await;
        handles.push(tokio::spawn(async move {
            let request = qa_tensor_infer_request(&format!("bat-{index}"), &qa_pixels(index));
            let responses = infer_raw(&mut client, vec![request]).await.expect("infer");
            assert_eq!(responses.len(), 1);
            let batch_size = responses[0]
                .meta
                .get("lumen.batch_size")
                .and_then(|value| value.parse::<usize>().ok());
            (
                parse_embedding(&responses[0]).expect("embedding"),
                batch_size,
            )
        }));
    }
    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await.expect("join"));
    }
    results
}

#[tokio::test]
async fn batched_results_match_individual_forwards() {
    // A wide queue window forces the concurrent requests into shared batches.
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions {
        queue_latency_ms: 100,
        max_batch_size: 8,
        ..HubOptions::default()
    });
    hub.wait_ready().await;

    let count = 6;
    let results = concurrent_infer(&hub, count).await;

    // Result integrity: every response equals its own single-input reference —
    // the strongest cross-request-contamination check a batcher can get.
    let device = default_device();
    let net = QaNet::deterministic(&device);
    for (index, (embedding, _)) in results.iter().enumerate() {
        let reference = net.embed(&qa_pixels(index), 1, &device);
        let similarity = cosine(embedding, &reference);
        assert!(
            similarity > 0.9999,
            "request {index} was contaminated by batching, cosine {similarity}"
        );
    }

    // Composition: with a 100ms window the six requests must have shared
    // batches at least once.
    let max_batch = results
        .iter()
        .filter_map(|(_, batch)| *batch)
        .max()
        .expect("batch sizes reported");
    assert!(
        max_batch >= 2,
        "expected shared batches under a 100ms queue window"
    );
}

#[tokio::test]
async fn batches_respect_max_batch_size() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions {
        queue_latency_ms: 100,
        max_batch_size: 2,
        ..HubOptions::default()
    });
    hub.wait_ready().await;

    let results = concurrent_infer(&hub, 6).await;
    for (index, (_, batch)) in results.iter().enumerate() {
        let batch = batch.expect("batch size reported");
        assert!(
            batch <= 2,
            "request {index} reported batch size {batch} > max_batch_size 2"
        );
    }
}

#[tokio::test]
async fn disabled_batching_runs_singletons() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions {
        batching_enabled: false,
        queue_latency_ms: 100,
        ..HubOptions::default()
    });
    hub.wait_ready().await;

    let results = concurrent_infer(&hub, 4).await;
    for (index, (_, batch)) in results.iter().enumerate() {
        assert!(
            batch.is_none() || batch == &Some(1),
            "request {index} must not share a batch when batching is disabled, got {batch:?}"
        );
    }
}
