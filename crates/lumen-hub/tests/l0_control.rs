//! L0 control-plane suite: GetStatus/WatchStatus/TailLogs and grpc.health.v1
//! against the running binary.
//!
//! Run: cargo test --features qa --test l0_control

mod common;

use std::time::Duration;

use common::harness::{
    HubOptions, HubProcess, infer_embedding, qa_pixels, qa_tensor_infer_request,
};
use lumen_hub::daemon::proto::lumen::control::v1::{Phase, TailLogsRequest};
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_check_response::ServingStatus;

#[tokio::test]
async fn watch_starts_with_the_current_snapshot_and_seq_is_monotonic() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    let snapshots = hub.wait_ready().await;
    let mut last_seq = 0;
    for snapshot in &snapshots {
        assert!(snapshot.seq > last_seq, "seq must be strictly increasing");
        last_seq = snapshot.seq;
    }

    // A fresh subscription after READY receives the READY snapshot first.
    let mut client = hub.control().await;
    let mut stream = client.watch_status(()).await.expect("watch").into_inner();
    let first = stream.message().await.expect("stream").expect("snapshot");
    assert_eq!(first.phase(), Phase::Ready);
    assert!(!first.version.is_empty());
}

#[tokio::test]
async fn get_status_agrees_with_watch() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    hub.wait_ready().await;

    let mut client = hub.control().await;
    let status = client
        .get_status(())
        .await
        .expect("get_status")
        .into_inner();
    assert_eq!(status.phase(), Phase::Ready);
    assert!(status.services.iter().any(|s| s.service == "qa"));
    assert!(status.error.is_empty());
}

#[tokio::test]
async fn tail_logs_one_shot_replays_backlog_and_closes() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    hub.wait_ready().await;

    let mut client = hub.control().await;
    let mut stream = client
        .tail_logs(TailLogsRequest {
            backlog_lines: 100,
            min_level: String::new(),
            follow: false,
        })
        .await
        .expect("tail_logs")
        .into_inner();

    let mut entries = Vec::new();
    while let Some(entry) = stream.message().await.expect("tail stream") {
        entries.push(entry);
    }
    assert!(
        !entries.is_empty(),
        "startup must have produced log entries"
    );
    assert!(
        entries.iter().any(|e| e.message.contains("ready")),
        "backlog should include the readiness log"
    );
    // One-shot: the stream ended on its own (loop exited without error).
}

#[tokio::test]
async fn tail_logs_follow_streams_new_entries() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    hub.wait_ready().await;

    let mut control = hub.control().await;
    let mut stream = control
        .tail_logs(TailLogsRequest {
            backlog_lines: 0,
            // DEBUG: data-plane traffic logs at debug level (h2 frames); the
            // default INFO floor would filter everything an infer produces.
            min_level: "DEBUG".to_owned(),
            follow: true,
        })
        .await
        .expect("tail_logs")
        .into_inner();

    // Produce fresh log activity through the data plane.
    let mut inference = hub.inference().await;
    infer_embedding(
        &mut inference,
        qa_tensor_infer_request("log", &qa_pixels(9)),
    )
    .await
    .expect("infer");

    let entry = tokio::time::timeout(Duration::from_secs(10), stream.message())
        .await
        .expect("live entry within 10s")
        .expect("stream")
        .expect("entry");
    assert!(!entry.level.is_empty());
    assert!(entry.time_unix_ms > 0);
}

#[tokio::test]
async fn tail_logs_min_level_filters() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    hub.wait_ready().await;

    let mut client = hub.control().await;
    let mut stream = client
        .tail_logs(TailLogsRequest {
            backlog_lines: 500,
            min_level: "ERROR".to_owned(),
            follow: false,
        })
        .await
        .expect("tail_logs")
        .into_inner();
    while let Some(entry) = stream.message().await.expect("stream") {
        assert_eq!(
            entry.level, "ERROR",
            "min_level=ERROR must filter: {entry:?}"
        );
    }

    let error = client
        .tail_logs(TailLogsRequest {
            backlog_lines: 10,
            min_level: "NOT_A_LEVEL".to_owned(),
            follow: false,
        })
        .await
        .expect_err("bad level must be rejected");
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn health_reports_per_service_status() {
    let (hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    hub.wait_ready().await;
    let mut health = hub.health().await;

    for service in ["", "home_native.v1.Inference"] {
        let status = health
            .check(HealthCheckRequest {
                service: service.to_owned(),
            })
            .await
            .unwrap_or_else(|e| panic!("check({service:?}): {e}"))
            .into_inner();
        assert_eq!(
            status.status(),
            ServingStatus::Serving,
            "service {service:?}"
        );
    }

    let error = health
        .check(HealthCheckRequest {
            service: "no.such.Service".to_owned(),
        })
        .await
        .expect_err("unknown service is NOT_FOUND");
    assert_eq!(error.code(), tonic::Code::NotFound);
}
