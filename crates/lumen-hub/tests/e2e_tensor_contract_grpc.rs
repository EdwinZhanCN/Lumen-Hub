//! Black-box gRPC E2E tests for the thin tensor input wire contract.
//!
//! These tests intentionally use a deterministic in-process fake service instead
//! of real model weights. They exercise the public gRPC protocol boundary:
//! capability reporting, tensor metadata validation, and batching transparency.

use std::{collections::HashMap, net::SocketAddr, sync::Arc, time::Duration};

use async_trait::async_trait;
use lumen_hub::{
    daemon::{
        BatcherConfig, HubGrpcService,
        proto::home_native::v1::{
            Capability, InferRequest, InferResponse, inference_client::InferenceClient,
            inference_server::InferenceServer,
        },
    },
    service::{
        BatchKey, DEFAULT_TENSOR_MIME, FixedShapeTensorValidationOptions, IMAGE_TENSOR_LAYOUT,
        INPUT_KIND_RAW, INPUT_KIND_TENSOR, InferenceService, META_INPUT_KIND, META_PREPROCESS_ID,
        META_PREPROCESS_SKIP, META_TENSOR_BYTE_ORDER, META_TENSOR_DTYPE, META_TENSOR_FORMAT,
        META_TENSOR_LAYOUT, META_TENSOR_SHAPE, PREPROCESS_BIOCLIP2_224_IMAGE,
        PREPROCESS_SIGLIP2_BASE_PATCH16_224_IMAGE, ServiceCapability, ServiceError, ServiceHub,
        ServiceResult, TENSOR_BYTE_ORDER_LITTLE, TENSOR_FORMAT_CONTIGUOUS, TaskHandler,
        TaskRegistry, TaskRequest, TaskResult, TaskSpec, is_tensor_input_request, shape_json,
        validate_fixed_shape_tensor_request,
    },
};
use tokio::sync::oneshot;
use tonic::{Code, Status, transport::Server};

const SERVICE_SIGLIP: &str = "siglip";
const SERVICE_BIOCLIP: &str = "bioclip";
const SERVICE_OCR: &str = "ocr";
const SERVICE_FACE: &str = "face";

const TASK_SEMANTIC_TEXT_EMBED: &str = "semantic_text_embed";
const TASK_SEMANTIC_IMAGE_EMBED: &str = "semantic_image_embed";
const TASK_BIOCLIP_CLASSIFY: &str = "bioclip_classify";
const TASK_OCR: &str = "ocr";
const TASK_FACE_RECOGNITION: &str = "face_recognition";

#[tokio::test]
async fn grpc_reports_thin_tensor_capabilities_and_protocol_version() {
    // Batcher enabled so the reported capability reflects each task's declared
    // tensor fast-path contract; the disabled-batcher masking is covered by
    // `grpc_masks_tensor_batching_when_batcher_disabled`.
    let harness = start_harness(
        contract_hub(),
        BatcherConfig {
            enabled: true,
            max_batch_size: 8,
            queue_latency: Duration::from_millis(2),
        },
    )
    .await;
    let mut client = harness.client().await;

    let capabilities = collect_capabilities(&mut client).await;
    assert_eq!(
        capabilities.len(),
        4,
        "expected one capability per fake service"
    );
    assert!(
        capabilities
            .iter()
            .all(|capability| capability.protocol_version == "1.0"),
        "all capabilities should carry the stable protocol version"
    );

    assert_task_contract(
        &capabilities,
        SERVICE_SIGLIP,
        TASK_SEMANTIC_TEXT_EMBED,
        "",
        false,
    );
    assert_task_contract(
        &capabilities,
        SERVICE_SIGLIP,
        TASK_SEMANTIC_IMAGE_EMBED,
        PREPROCESS_SIGLIP2_BASE_PATCH16_224_IMAGE,
        true,
    );
    assert_task_contract(
        &capabilities,
        SERVICE_BIOCLIP,
        TASK_BIOCLIP_CLASSIFY,
        PREPROCESS_BIOCLIP2_224_IMAGE,
        true,
    );
    assert_task_contract(&capabilities, SERVICE_OCR, TASK_OCR, "", false);
    assert_task_contract(
        &capabilities,
        SERVICE_FACE,
        TASK_FACE_RECOGNITION,
        "",
        false,
    );
}

#[tokio::test]
async fn grpc_masks_tensor_batching_when_batcher_disabled() {
    // When the daemon batcher is off (the default), the Hub must not advertise
    // tensor_batching_supported: clients would otherwise expect concurrent
    // tensor requests to be batched when they will not be. The tensor fast path
    // (preprocess_id) stays advertised so clients still use tensor input.
    let harness = start_harness(contract_hub(), BatcherConfig::disabled()).await;
    let mut client = harness.client().await;

    let capabilities = collect_capabilities(&mut client).await;

    assert_task_contract(
        &capabilities,
        SERVICE_SIGLIP,
        TASK_SEMANTIC_IMAGE_EMBED,
        PREPROCESS_SIGLIP2_BASE_PATCH16_224_IMAGE,
        false,
    );
    assert_task_contract(
        &capabilities,
        SERVICE_BIOCLIP,
        TASK_BIOCLIP_CLASSIFY,
        PREPROCESS_BIOCLIP2_224_IMAGE,
        false,
    );
}

#[tokio::test]
async fn grpc_rejects_malformed_tensor_requests_with_invalid_argument() {
    let harness = start_harness(validating_siglip_hub(false), BatcherConfig::disabled()).await;
    let mut client = harness.client().await;

    let valid = valid_tensor_request("valid", filled_tensor_payload(7));
    let response = infer_ok(&mut client, valid).await;
    assert_eq!(response.correlation_id, "valid");
    assert_eq!(response.result, b"tensor:7");
    assert_eq!(
        response.meta.get(META_INPUT_KIND),
        Some(&INPUT_KIND_TENSOR.to_owned())
    );

    let cases = [
        (
            "wrong-preprocess-id",
            mutated_tensor_request("wrong-preprocess-id", |request| {
                request.meta.insert(
                    META_PREPROCESS_ID.to_owned(),
                    "future_siglip_unknown_v99".to_owned(),
                );
            }),
        ),
        (
            "wrong-dtype",
            mutated_tensor_request("wrong-dtype", |request| {
                request
                    .meta
                    .insert(META_TENSOR_DTYPE.to_owned(), "fp16".to_owned());
            }),
        ),
        (
            "wrong-layout",
            mutated_tensor_request("wrong-layout", |request| {
                request
                    .meta
                    .insert(META_TENSOR_LAYOUT.to_owned(), "NHWC".to_owned());
            }),
        ),
        (
            "wrong-byte-order",
            mutated_tensor_request("wrong-byte-order", |request| {
                request
                    .meta
                    .insert(META_TENSOR_BYTE_ORDER.to_owned(), "big".to_owned());
            }),
        ),
        (
            "wrong-shape",
            mutated_tensor_request("wrong-shape", |request| {
                request.meta.insert(
                    META_TENSOR_SHAPE.to_owned(),
                    shape_json(&[1_usize, 3, 224, 225]),
                );
                request.payload = vec![0; 1 * 3 * 224 * 225 * 4];
            }),
        ),
        (
            "wrong-payload-length",
            mutated_tensor_request("wrong-payload-length", |request| {
                request.payload.pop();
            }),
        ),
    ];

    for (name, request) in cases {
        let status = infer_status(&mut client, request).await;
        assert_eq!(
            status.code(),
            Code::InvalidArgument,
            "{name} should be rejected as InvalidArgument; got {status:?}"
        );
    }
}

#[tokio::test]
async fn grpc_batches_concurrent_tensor_requests_without_client_visible_policy() {
    let harness = start_harness(
        validating_siglip_hub(true),
        BatcherConfig {
            enabled: true,
            max_batch_size: 2,
            queue_latency: Duration::from_secs(60),
        },
    )
    .await;
    let mut first_client = harness.client().await;
    let mut second_client = harness.client().await;

    let first_request = valid_tensor_request("first", filled_tensor_payload(1));
    let second_request = valid_tensor_request("second", filled_tensor_payload(2));

    let (first, second) = tokio::join!(
        infer_ok(&mut first_client, first_request),
        infer_ok(&mut second_client, second_request),
    );

    assert_eq!(first.correlation_id, "first");
    assert_eq!(second.correlation_id, "second");
    assert_eq!(first.result, b"batch:1");
    assert_eq!(second.result, b"batch:2");
}

async fn collect_capabilities(
    client: &mut InferenceClient<tonic::transport::Channel>,
) -> Vec<Capability> {
    let mut stream = client
        .stream_capabilities(())
        .await
        .expect("StreamCapabilities RPC starts")
        .into_inner();
    let mut capabilities = Vec::new();
    while let Some(capability) = stream
        .message()
        .await
        .expect("capability stream message succeeds")
    {
        capabilities.push(capability);
    }
    capabilities
}

fn assert_task_contract(
    capabilities: &[Capability],
    service_name: &str,
    task_name: &str,
    tensor_preprocess_id: &str,
    tensor_batching_supported: bool,
) {
    let capability = capabilities
        .iter()
        .find(|capability| capability.service_name == service_name)
        .unwrap_or_else(|| panic!("missing service capability {service_name}"));
    let task = capability
        .tasks
        .iter()
        .find(|task| task.name == task_name)
        .unwrap_or_else(|| panic!("missing task {service_name}/{task_name}"));

    assert_eq!(task.tensor_preprocess_id, tensor_preprocess_id);
    assert_eq!(task.tensor_batching_supported, tensor_batching_supported);
}

async fn infer_ok(
    client: &mut InferenceClient<tonic::transport::Channel>,
    request: InferRequest,
) -> InferResponse {
    let mut stream = client
        .infer(tonic::codegen::tokio_stream::iter([request]))
        .await
        .expect("Infer RPC starts")
        .into_inner();
    stream
        .message()
        .await
        .expect("Infer response message succeeds")
        .expect("Infer response stream contains a final response")
}

async fn infer_status(
    client: &mut InferenceClient<tonic::transport::Channel>,
    request: InferRequest,
) -> Status {
    match client
        .infer(tonic::codegen::tokio_stream::iter([request]))
        .await
    {
        Err(status) => status,
        Ok(response) => match response.into_inner().message().await {
            Err(status) => status,
            Ok(Some(response)) => panic!("expected gRPC status error, got response {response:?}"),
            Ok(None) => panic!("expected gRPC status error, got empty response stream"),
        },
    }
}

fn contract_hub() -> ServiceHub {
    let mut hub = ServiceHub::new();
    hub.register(ContractService::new(
        SERVICE_SIGLIP,
        "siglip-test-model",
        vec![
            ContractTask::raw(
                TaskSpec::new(TASK_SEMANTIC_TEXT_EMBED, "embed text")
                    .with_input_mimes(["text/plain"])
                    .with_output_mime("application/json;schema=embedding_v1"),
            ),
            ContractTask::raw(tensor_image_spec(
                TASK_SEMANTIC_IMAGE_EMBED,
                PREPROCESS_SIGLIP2_BASE_PATCH16_224_IMAGE,
                true,
            )),
        ],
    ))
    .unwrap();
    hub.register(ContractService::new(
        SERVICE_BIOCLIP,
        "bioclip-test-model",
        vec![ContractTask::raw(tensor_image_spec(
            TASK_BIOCLIP_CLASSIFY,
            PREPROCESS_BIOCLIP2_224_IMAGE,
            true,
        ))],
    ))
    .unwrap();
    hub.register(ContractService::new(
        SERVICE_OCR,
        "ocr-test-model",
        vec![ContractTask::raw(
            TaskSpec::new(TASK_OCR, "ocr raw image")
                .with_input_mimes(["image/jpeg", "image/png"])
                .with_output_mime("application/json;schema=ocr_v1"),
        )],
    ))
    .unwrap();
    hub.register(ContractService::new(
        SERVICE_FACE,
        "face-test-model",
        vec![ContractTask::raw(
            TaskSpec::new(TASK_FACE_RECOGNITION, "face recognition raw image")
                .with_input_mimes(["image/jpeg", "image/png"])
                .with_output_mime("application/json;schema=face_v1"),
        )],
    ))
    .unwrap();
    hub
}

fn validating_siglip_hub(batchable: bool) -> ServiceHub {
    let mut hub = ServiceHub::new();
    hub.register(ContractService::new(
        SERVICE_SIGLIP,
        "siglip-test-model",
        vec![ContractTask::validating_tensor(
            tensor_image_spec(
                TASK_SEMANTIC_IMAGE_EMBED,
                PREPROCESS_SIGLIP2_BASE_PATCH16_224_IMAGE,
                batchable,
            ),
            batchable,
        )],
    ))
    .unwrap();
    hub
}

fn tensor_image_spec(
    task_name: &'static str,
    preprocess_id: &'static str,
    batching_supported: bool,
) -> TaskSpec {
    TaskSpec::new(task_name, "image tensor contract task")
        .with_input_mimes([DEFAULT_TENSOR_MIME, "image/jpeg", "image/png"])
        .with_output_mime("application/json")
        .with_tensor_fast_path(preprocess_id, batching_supported)
}

fn valid_tensor_request(correlation_id: &str, payload: Vec<u8>) -> InferRequest {
    InferRequest {
        correlation_id: correlation_id.to_owned(),
        task: TASK_SEMANTIC_IMAGE_EMBED.to_owned(),
        payload,
        meta: valid_tensor_meta(),
        payload_mime: DEFAULT_TENSOR_MIME.to_owned(),
        seq: 0,
        total: 1,
        offset: 0,
    }
}

fn mutated_tensor_request(
    correlation_id: &str,
    mutate: impl FnOnce(&mut InferRequest),
) -> InferRequest {
    let mut request = valid_tensor_request(correlation_id, filled_tensor_payload(0));
    mutate(&mut request);
    request
}

fn valid_tensor_meta() -> HashMap<String, String> {
    HashMap::from([
        ("service".to_owned(), SERVICE_SIGLIP.to_owned()),
        (META_INPUT_KIND.to_owned(), INPUT_KIND_TENSOR.to_owned()),
        (META_PREPROCESS_SKIP.to_owned(), "true".to_owned()),
        (META_TENSOR_DTYPE.to_owned(), "fp32".to_owned()),
        (
            META_TENSOR_SHAPE.to_owned(),
            shape_json(&[1_usize, 3, 224, 224]),
        ),
        (
            META_TENSOR_LAYOUT.to_owned(),
            IMAGE_TENSOR_LAYOUT.to_owned(),
        ),
        (
            META_TENSOR_FORMAT.to_owned(),
            TENSOR_FORMAT_CONTIGUOUS.to_owned(),
        ),
        (
            META_TENSOR_BYTE_ORDER.to_owned(),
            TENSOR_BYTE_ORDER_LITTLE.to_owned(),
        ),
        (
            META_PREPROCESS_ID.to_owned(),
            PREPROCESS_SIGLIP2_BASE_PATCH16_224_IMAGE.to_owned(),
        ),
    ])
}

fn filled_tensor_payload(marker: u8) -> Vec<u8> {
    vec![marker; 1 * 3 * 224 * 224 * 4]
}

struct ContractService {
    name: String,
    model_id: String,
    tasks: TaskRegistry,
}

impl ContractService {
    fn new(name: &str, model_id: &str, tasks: Vec<ContractTask>) -> Self {
        let mut registry = TaskRegistry::new();
        for task in tasks {
            registry.register(task).unwrap();
        }
        Self {
            name: name.to_owned(),
            model_id: model_id.to_owned(),
            tasks: registry,
        }
    }
}

impl InferenceService for ContractService {
    fn name(&self) -> &str {
        &self.name
    }

    fn tasks(&self) -> &TaskRegistry {
        &self.tasks
    }

    fn capability(&self) -> ServiceCapability {
        self.tasks
            .build_capability(&self.name, vec![self.model_id.clone()], "test-runtime")
    }
}

struct ContractTask {
    spec: TaskSpec,
    validate_tensor: bool,
    batchable: bool,
}

impl ContractTask {
    fn raw(spec: TaskSpec) -> Self {
        Self {
            spec,
            validate_tensor: false,
            batchable: false,
        }
    }

    fn validating_tensor(spec: TaskSpec, batchable: bool) -> Self {
        Self {
            spec,
            validate_tensor: true,
            batchable,
        }
    }

    fn validate_request(&self, request: &TaskRequest) -> ServiceResult<()> {
        if self.validate_tensor && is_tensor_input_request(request) {
            validate_fixed_shape_tensor_request(
                request,
                FixedShapeTensorValidationOptions {
                    dtype: "fp32",
                    layout: IMAGE_TENSOR_LAYOUT,
                    preprocess_id: PREPROCESS_SIGLIP2_BASE_PATCH16_224_IMAGE,
                    expected_shape: &[1, 3, 224, 224],
                },
            )?;
        }
        Ok(())
    }
}

#[async_trait]
impl TaskHandler for ContractTask {
    fn spec(&self) -> &TaskSpec {
        &self.spec
    }

    fn batch_key(&self, request: &TaskRequest) -> ServiceResult<Option<BatchKey>> {
        if self.batchable && is_tensor_input_request(request) {
            return Ok(Some(BatchKey::new(format!("{}:tensor", self.spec.name))));
        }
        Ok(None)
    }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
        self.validate_request(&request)?;
        let marker = request.payload.first().copied().unwrap_or_default();
        let input_kind = if is_tensor_input_request(&request) {
            INPUT_KIND_TENSOR
        } else {
            INPUT_KIND_RAW
        };
        Ok(
            TaskResult::new(format!("{input_kind}:{marker}"), "text/plain")
                .with_meta(META_INPUT_KIND, input_kind),
        )
    }

    async fn handle_batch(&self, requests: Vec<TaskRequest>) -> ServiceResult<Vec<TaskResult>> {
        if !self.batchable {
            let mut results = Vec::with_capacity(requests.len());
            for request in requests {
                results.push(self.handle(request).await?);
            }
            return Ok(results);
        }

        requests
            .into_iter()
            .map(|request| {
                self.validate_request(&request)?;
                let marker = request.payload.first().copied().unwrap_or_default();
                Ok(TaskResult::new(format!("batch:{marker}"), "text/plain")
                    .with_meta(META_INPUT_KIND, INPUT_KIND_TENSOR))
            })
            .collect::<Result<Vec<_>, ServiceError>>()
    }
}

struct GrpcHarness {
    endpoint: String,
    shutdown: Option<oneshot::Sender<()>>,
}

impl GrpcHarness {
    async fn client(&self) -> InferenceClient<tonic::transport::Channel> {
        let mut last_error = None;
        for _ in 0..50 {
            match InferenceClient::connect(self.endpoint.clone()).await {
                Ok(client) => return client,
                Err(err) => {
                    last_error = Some(err);
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            }
        }
        panic!("failed to connect to {}: {last_error:?}", self.endpoint);
    }
}

impl Drop for GrpcHarness {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

async fn start_harness(hub: ServiceHub, batching: BatcherConfig) -> GrpcHarness {
    let addr = reserve_loopback_addr();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let service = HubGrpcService::new(Arc::new(hub), batching);

    tokio::spawn(async move {
        Server::builder()
            .add_service(InferenceServer::new(service))
            .serve_with_shutdown(addr, async {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("gRPC test server exits cleanly");
    });

    GrpcHarness {
        endpoint: format!("http://{addr}"),
        shutdown: Some(shutdown_tx),
    }
}

fn reserve_loopback_addr() -> SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = listener.local_addr().expect("read ephemeral addr");
    drop(listener);
    addr
}
