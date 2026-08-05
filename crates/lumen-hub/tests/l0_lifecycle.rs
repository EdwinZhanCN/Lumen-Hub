//! L0 lifecycle suite: the real `lumen-hub` binary against a mock model
//! repository — control-plane-first startup, caching, failure modes, and
//! shutdown. No real model weights involved (QA fixture model).
//!
//! Run: cargo test --features qa --test l0_lifecycle

mod common;

use std::time::Duration;

use common::harness::{HubOptions, HubProcess, qa_pixels, qa_tensor_infer_request};
use lumen_hub::daemon::proto::lumen::control::v1::Phase;
use tonic_health::pb::HealthCheckRequest;
use tonic_health::pb::health_check_response::ServingStatus;

#[tokio::test]
async fn cold_start_walks_all_phases_and_serves() {
    // Hold artifact downloads until the status watch is subscribed, so the
    // full phase sequence is observable regardless of how fast the tiny QA
    // model installs.
    let (mut hub, repo) = HubProcess::spawn_with_qa_repo_held(&HubOptions::default());
    let mut client = hub.control().await;
    let watch = client.watch_status(()).await.expect("subscribe");
    drop(watch);
    repo.release_downloads();
    let snapshots = hub.wait_ready().await;

    // Phase order: the terminal snapshot is READY and the startup phases
    // appear as an ordered subsequence.
    let phases: Vec<Phase> = snapshots.iter().map(|s| s.phase()).collect();
    for required in [
        Phase::Downloading,
        Phase::Loading,
        Phase::Warmup,
        Phase::Ready,
    ] {
        assert!(
            phases.contains(&required),
            "missing phase {required:?} in {phases:?}"
        );
    }
    let position = |phase: Phase| phases.iter().position(|p| *p == phase).unwrap();
    assert!(position(Phase::Downloading) < position(Phase::Loading));
    assert!(position(Phase::Loading) < position(Phase::Warmup));
    assert!(position(Phase::Warmup) < position(Phase::Ready));

    // Download progress carried real per-file byte/file counters.
    let progress = snapshots
        .iter()
        .filter_map(|s| s.download.as_ref())
        .last()
        .expect("at least one download progress snapshot");
    assert_eq!(progress.model, "qa-tiny");
    assert!(progress.files_total > 0);
    assert!(repo.download_count() > 0, "artifacts came from the repo");

    // Ready snapshot reports the qa service.
    let ready = snapshots.last().unwrap();
    assert!(ready.services.iter().any(|s| s.service == "qa"));
    assert!(ready.seq > 0);

    // grpc.health.v1 flips to SERVING.
    let mut health = hub.health().await;
    let status = health
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(status.status(), ServingStatus::Serving);

    // The data plane serves a real inference.
    let mut inference = hub.inference().await;
    let embedding = common::harness::infer_embedding(
        &mut inference,
        qa_tensor_infer_request("cold", &qa_pixels(1)),
    )
    .await
    .expect("infer after ready");

    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "normalized embedding, norm {norm}"
    );

    assert!(hub.is_running());
}

#[tokio::test]
async fn mDNS_is_advertised_after_ready_when_enabled() {
    // The daemon registers the mDNS advertisement only once the hub reaches
    // READY (see `initialize` in src/main.rs). Reaching READY with mDNS
    // enabled proves the registration step completes; the TXT payload itself
    // (display-only v/runtime keys) is covered by unit tests in daemon/mdns.rs.
    let (mut hub, repo) = HubProcess::spawn_with_qa_repo_held(&HubOptions {
        mdns_enabled: true,
        ..HubOptions::default()
    });
    let mut client = hub.control().await;
    let watch = client.watch_status(()).await.expect("subscribe");
    drop(watch);
    repo.release_downloads();
    let snapshots = hub.wait_ready().await;
    assert_eq!(snapshots.last().unwrap().phase(), Phase::Ready);
}

#[tokio::test]
async fn warm_start_hits_the_cache() {
    let (mut hub, repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    hub.wait_ready().await;
    let downloads_after_cold = repo.download_count();
    assert!(downloads_after_cold > 0);
    hub.shutdown();

    // Same scratch (cache preserved), same repo: no artifact re-downloads.
    let warm = HubProcess::spawn(
        &hub.scratch.clone(),
        &repo.endpoint(),
        &HubOptions::default(),
    );
    warm.wait_ready().await;
    assert_eq!(
        repo.download_count(),
        downloads_after_cold,
        "warm start must be served from the cache"
    );
}

#[tokio::test]
async fn download_failure_keeps_the_control_plane_queryable() {
    let (mut hub, repo) = {
        let scratch = std::env::temp_dir().join(format!(
            "lumen-e2e-fail-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let repo_root = scratch.join("repo");
        lumen_hub::models::qa::write_model_fixture(
            &repo_root.join(lumen_hub::models::qa::QA_MODEL_NAME),
        )
        .expect("qa fixture");
        let repo = common::mock_repo::MockRepo::serve(repo_root);
        repo.fail_resolve("net.fp32.bpk");
        let hub = HubProcess::spawn(&scratch, &repo.endpoint(), &HubOptions::default());
        (hub, repo)
    };
    let _ = &repo;

    let (snapshots, terminal) = hub.watch_until_terminal().await;
    assert_eq!(terminal, Phase::Failed);
    let failed = snapshots.last().unwrap();
    assert!(
        !failed.error.is_empty(),
        "FAILED snapshot must carry the error"
    );

    // The process survives, health reports NOT_SERVING, data plane refuses.
    assert!(hub.is_running(), "failed hub must stay queryable");
    let mut health = hub.health().await;
    let status = health
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(status.status(), ServingStatus::NotServing);

    let mut inference = hub.inference().await;
    let error = common::harness::infer_embedding(
        &mut inference,
        qa_tensor_infer_request("gate", &qa_pixels(2)),
    )
    .await
    .expect_err("data plane must refuse before ready");
    assert_eq!(error.code(), tonic::Code::Unavailable, "got: {error}");
}

#[tokio::test]
async fn corrupt_model_info_fails_with_parse_error() {
    let scratch = std::env::temp_dir().join(format!(
        "lumen-e2e-corrupt-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let repo_root = scratch.join("repo");
    lumen_hub::models::qa::write_model_fixture(
        &repo_root.join(lumen_hub::models::qa::QA_MODEL_NAME),
    )
    .expect("qa fixture");
    let repo = common::mock_repo::MockRepo::serve(repo_root);
    repo.corrupt_resolve("model_info.json");

    let mut hub = HubProcess::spawn(&scratch, &repo.endpoint(), &HubOptions::default());
    let (snapshots, terminal) = hub.watch_until_terminal().await;
    assert_eq!(terminal, Phase::Failed);
    assert!(!snapshots.last().unwrap().error.is_empty());
    assert!(hub.is_running());
}

#[tokio::test]
async fn invalid_config_exits_immediately() {
    // References a service that does not exist in `services`; config
    // validation must reject it before any server binds.
    let scratch = std::env::temp_dir().join(format!(
        "lumen-e2e-badcfg-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&scratch).unwrap();
    let config_path = scratch.join("config.json");
    std::fs::write(
        &config_path,
        serde_json::json!({
            "metadata": {"version": "0.1.0", "region": "other", "cache_dir": scratch.join("cache").to_string_lossy()},
            "deployment": {"mode": "hub", "services": ["nonexistent"]},
            "server": {"host": "127.0.0.1", "port": 51999, "mdns": {"enabled": false}},
            "services": {}
        })
        .to_string(),
    )
    .unwrap();

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_lumen-hub"))
        .arg("--config")
        .arg(&config_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn");
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let code = loop {
        if let Ok(Some(status)) = child.try_wait() {
            break status.code();
        }
        assert!(
            std::time::Instant::now() < deadline,
            "binary should exit on invalid config"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(code, Some(1));
    let _ = std::fs::remove_dir_all(&scratch);
}

#[cfg(unix)]
#[tokio::test]
async fn sigterm_shuts_down_gracefully() {
    let (mut hub, _repo) = HubProcess::spawn_with_qa_repo(&HubOptions::default());
    hub.wait_ready().await;

    let pid = hub.pid();
    let status = std::process::Command::new("kill")
        .arg("-TERM")
        .arg(pid.to_string())
        .status()
        .expect("send SIGTERM");
    assert!(status.success());

    let status = hub.wait_exit(Duration::from_secs(15)).await;
    match status {
        Some(status) => assert!(
            status.success(),
            "graceful shutdown should exit 0, got {status:?}\nlogs:\n{}",
            hub.logs()
        ),
        None => panic!(
            "hub did not exit within 15s of SIGTERM\nlogs:\n{}",
            hub.logs()
        ),
    }
}
