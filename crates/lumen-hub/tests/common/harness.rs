//! Binary-level e2e harness: spawns the real `lumen-hub` binary against a
//! mock model repository and talks to it over real gRPC.

#![allow(dead_code)]

use std::{
    fs,
    net::TcpListener,
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use lumen_hub::daemon::proto::home_native::v1 as datapb;
use lumen_hub::daemon::proto::home_native::v1::inference_client::InferenceClient;
use lumen_hub::daemon::proto::lumen::control::v1 as controlpb;
use lumen_hub::daemon::proto::lumen::control::v1::control_client::ControlClient;
use lumen_hub::models::qa;

use super::mock_repo::MockRepo;

pub const READY_TIMEOUT: Duration = Duration::from_secs(60);

pub struct HubOptions {
    pub precision: &'static str,
    pub batching_enabled: bool,
    pub max_batch_size: usize,
    pub queue_latency_ms: u64,
}

impl Default for HubOptions {
    fn default() -> Self {
        Self {
            precision: "fp32",
            batching_enabled: true,
            max_batch_size: 8,
            queue_latency_ms: 2,
        }
    }
}

/// A spawned `lumen-hub` process plus its scratch dirs; killed on drop.
pub struct HubProcess {
    child: Child,
    pub port: u16,
    pub scratch: PathBuf,
    pub cache_dir: PathBuf,
    pub log_path: PathBuf,
}

impl HubProcess {
    /// Writes the QA fixture into a fresh repo dir, starts a `MockRepo` over
    /// it, and spawns the hub configured to install from that repo.
    pub fn spawn_with_qa_repo(options: &HubOptions) -> (Self, MockRepo) {
        let scratch = scratch_dir();
        let repo_root = scratch.join("repo");
        qa::write_model_fixture(&repo_root.join(qa::QA_MODEL_NAME)).expect("qa fixture");
        let repo = MockRepo::serve(repo_root);
        let hub = Self::spawn(&scratch, &repo.endpoint(), options);
        (hub, repo)
    }

    /// Spawns the hub against an existing repo endpoint. `scratch` owns the
    /// config/cache/log files.
    pub fn spawn(scratch: &PathBuf, repo_endpoint: &str, options: &HubOptions) -> Self {
        fs::create_dir_all(scratch).expect("scratch dir");
        let cache_dir = scratch.join("cache");
        let config_path = scratch.join("config.json");
        let log_path = scratch.join("hub.log");
        let port = free_port();

        let config = serde_json::json!({
            "metadata": {
                "version": "0.1.0",
                "region": "other",
                "cache_dir": cache_dir.to_string_lossy(),
            },
            "deployment": { "mode": "hub", "services": ["qa"] },
            "server": {
                "host": "127.0.0.1",
                "port": port,
                "mdns": { "enabled": false },
                "batching": {
                    "enabled": options.batching_enabled,
                    "max_batch_size": options.max_batch_size,
                    "queue_latency_ms": options.queue_latency_ms,
                },
            },
            "services": {
                "qa": {
                    "enabled": true,
                    "package": "qa",
                    "models": {
                        "default": {
                            "model": qa::QA_MODEL_NAME,
                            "runtime": "burn",
                            "precision": options.precision,
                        }
                    }
                }
            }
        });
        fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).expect("write config");

        let log = fs::File::create(&log_path).expect("create log file");
        let child = Command::new(env!("CARGO_BIN_EXE_lumen-hub"))
            .arg("--config")
            .arg(&config_path)
            .arg("--log-level")
            .arg("DEBUG")
            .env("LUMEN_MODEL_ENDPOINT", repo_endpoint)
            .stdout(Stdio::from(log.try_clone().expect("log clone")))
            .stderr(Stdio::from(log))
            .spawn()
            .expect("spawn lumen-hub");

        Self {
            child,
            port,
            scratch: scratch.clone(),
            cache_dir,
            log_path,
        }
    }

    pub fn grpc_endpoint(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn logs(&self) -> String {
        fs::read_to_string(&self.log_path).unwrap_or_default()
    }

    pub fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Waits until the process exits, returning its status (None = still
    /// running at the deadline). Async on purpose: a blocking poll would
    /// starve the test runtime and keep dropped client channels from closing
    /// their connections — which graceful shutdown waits on.
    pub async fn wait_exit(&mut self, timeout: Duration) -> Option<std::process::ExitStatus> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Ok(Some(status)) = self.child.try_wait() {
                return Some(status);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        None
    }

    /// Connects to the control plane, retrying while the port comes up.
    pub async fn control(&self) -> ControlClient<tonic::transport::Channel> {
        let endpoint = self.grpc_endpoint();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match ControlClient::connect(endpoint.clone()).await {
                Ok(client) => return client,
                Err(err) => {
                    if Instant::now() > deadline {
                        panic!("control plane never came up: {err}\nlogs:\n{}", self.logs());
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }

    pub async fn inference(&self) -> InferenceClient<tonic::transport::Channel> {
        InferenceClient::connect(self.grpc_endpoint())
            .await
            .unwrap_or_else(|err| panic!("inference connect: {err}\nlogs:\n{}", self.logs()))
    }

    /// Streams WatchStatus until the hub reaches READY (or FAILED/timeout →
    /// panic), returning every observed snapshot in order.
    pub async fn wait_ready(&self) -> Vec<controlpb::StatusSnapshot> {
        match self.watch_until_terminal().await {
            (snapshots, controlpb::Phase::Ready) => snapshots,
            (_, phase) => panic!(
                "hub reached {phase:?} instead of READY\nlogs:\n{}",
                self.logs()
            ),
        }
    }

    /// Streams WatchStatus until READY or FAILED, returning all snapshots and
    /// the terminal phase.
    pub async fn watch_until_terminal(&self) -> (Vec<controlpb::StatusSnapshot>, controlpb::Phase) {
        let mut client = self.control().await;
        let mut stream = client
            .watch_status(())
            .await
            .expect("watch_status")
            .into_inner();
        let mut snapshots = Vec::new();
        let deadline = Instant::now() + READY_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let message = tokio::time::timeout(remaining, stream.message())
                .await
                .unwrap_or_else(|_| panic!("WatchStatus timed out\nlogs:\n{}", self.logs()))
                .expect("watch stream");
            let Some(snapshot) = message else {
                panic!("WatchStatus stream ended early\nlogs:\n{}", self.logs());
            };
            let phase = snapshot.phase();
            snapshots.push(snapshot);
            if matches!(phase, controlpb::Phase::Ready | controlpb::Phase::Failed) {
                return (snapshots, phase);
            }
        }
    }

    pub fn shutdown(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for HubProcess {
    fn drop(&mut self) {
        self.shutdown();
        let _ = fs::remove_dir_all(&self.scratch);
    }
}

/// Builds a single-message Infer request for the QA task over the tensor fast
/// path. `pixels` must be `QA_INPUT_NUMEL` values.
pub fn qa_tensor_infer_request(correlation_id: &str, pixels: &[f32]) -> datapb::InferRequest {
    assert_eq!(pixels.len(), qa::QA_INPUT_NUMEL);
    let mut payload = Vec::with_capacity(pixels.len() * 4);
    for value in pixels {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let mut meta: std::collections::HashMap<String, String> = qa::tensor_request_meta()
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect();
    meta.insert("service".to_owned(), "qa".to_owned());
    datapb::InferRequest {
        correlation_id: correlation_id.to_owned(),
        task: qa::QA_EMBED_TASK.to_owned(),
        payload,
        meta,
        payload_mime: "application/octet-stream".to_owned(),
        seq: 0,
        total: 1,
        offset: 0,
    }
}

/// Deterministic per-index test image, distinct across `salt`.
pub fn qa_pixels(salt: usize) -> Vec<f32> {
    (0..qa::QA_INPUT_NUMEL)
        .map(|i| ((i.wrapping_mul(salt + 3).wrapping_add(salt)) % 253) as f32 / 253.0)
        .collect()
}

/// Runs one Infer round-trip and parses the embedding vector.
pub async fn infer_embedding(
    client: &mut InferenceClient<tonic::transport::Channel>,
    request: datapb::InferRequest,
) -> Result<Vec<f32>, tonic::Status> {
    let responses = infer_raw(client, vec![request]).await?;
    assert_eq!(responses.len(), 1, "expected a single final response");
    parse_embedding(&responses[0])
}

/// Sends the given request messages on one Infer stream, collecting responses.
pub async fn infer_raw(
    client: &mut InferenceClient<tonic::transport::Channel>,
    requests: Vec<datapb::InferRequest>,
) -> Result<Vec<datapb::InferResponse>, tonic::Status> {
    let stream = client
        .infer(tonic::Request::new(tokio_stream_iter(requests)))
        .await?
        .into_inner();
    collect_stream(stream).await
}

pub fn parse_embedding(response: &datapb::InferResponse) -> Result<Vec<f32>, tonic::Status> {
    if let Some(error) = &response.error {
        return Err(tonic::Status::internal(format!(
            "in-band error: {} ({})",
            error.message, error.detail
        )));
    }
    let embedding: serde_json::Value =
        serde_json::from_slice(&response.result).expect("embedding json");
    Ok(embedding["vector"]
        .as_array()
        .expect("vector array")
        .iter()
        .map(|v| v.as_f64().expect("f32") as f32)
        .collect())
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

async fn collect_stream(
    mut stream: tonic::Streaming<datapb::InferResponse>,
) -> Result<Vec<datapb::InferResponse>, tonic::Status> {
    let mut responses = Vec::new();
    while let Some(response) = stream.message().await? {
        responses.push(response);
    }
    Ok(responses)
}

fn tokio_stream_iter(
    requests: Vec<datapb::InferRequest>,
) -> impl tonic::codegen::tokio_stream::Stream<Item = datapb::InferRequest> {
    tonic::codegen::tokio_stream::iter(requests)
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("free port probe")
        .local_addr()
        .expect("probe addr")
        .port()
}

fn scratch_dir() -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "lumen-e2e-{}-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

impl HubProcess {
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// grpc.health.v1 client on the hub port (the generated tonic-health
    /// client ships without the transport `connect` helper).
    pub async fn health(
        &self,
    ) -> tonic_health::pb::health_client::HealthClient<tonic::transport::Channel> {
        let channel = tonic::transport::Endpoint::from_shared(self.grpc_endpoint())
            .expect("endpoint")
            .connect()
            .await
            .unwrap_or_else(|err| panic!("health connect: {err}\nlogs:\n{}", self.logs()));
        tonic_health::pb::health_client::HealthClient::new(channel)
    }
}

impl HubProcess {
    /// Like [`Self::spawn_with_qa_repo`] but with downloads held; call
    /// `repo.release_downloads()` once subscribed to WatchStatus.
    pub fn spawn_with_qa_repo_held(options: &HubOptions) -> (Self, MockRepo) {
        let scratch = scratch_dir();
        let repo_root = scratch.join("repo");
        qa::write_model_fixture(&repo_root.join(qa::QA_MODEL_NAME)).expect("qa fixture");
        let repo = MockRepo::serve(repo_root);
        repo.hold_downloads();
        let hub = Self::spawn(&scratch, &repo.endpoint(), options);
        (hub, repo)
    }
}
