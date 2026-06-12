use std::{collections::HashMap, pin::Pin, sync::Arc};

use bytes::Bytes;
use tonic::{Request, Response, Status};

use crate::{
    daemon::{BatchFn, Batcher, BatcherConfig, proto::home_native::v1},
    service::{ServiceCapability, ServiceError, ServiceHub, TaskRequest, TaskResult, TaskSpec},
};

const SERVICE_META_KEY: &str = "service";
const RESPONSE_CHUNK_SIZE: usize = 4 * 1024 * 1024;
const META_INPUT_KIND: &str = "lumen.input.kind";
const META_PREPROCESS_SKIP: &str = "lumen.preprocess.skip";
const INPUT_KIND_TENSOR: &str = "tensor";

type ResponseStream<T> =
    Pin<Box<dyn tonic::codegen::tokio_stream::Stream<Item = Result<T, Status>> + Send + 'static>>;

/// gRPC adapter for the protocol-independent `ServiceHub`.
pub struct HubGrpcService {
    hub: Arc<ServiceHub>,
    batcher: Batcher,
}

impl HubGrpcService {
    pub fn new(hub: Arc<ServiceHub>, batching: BatcherConfig) -> Self {
        Self {
            hub,
            batcher: Batcher::new(batching),
        }
    }

    pub fn hub(&self) -> &Arc<ServiceHub> {
        &self.hub
    }

    /// Converts a capability for wire reporting, masking tensor batching
    /// support when the daemon batcher is disabled.
    fn reported_capability(&self, capability: ServiceCapability) -> v1::Capability {
        let mut capability = v1::Capability::from(capability);
        if !self.batcher.config().enabled {
            for task in &mut capability.tasks {
                task.tensor_batching_supported = false;
            }
        }
        capability
    }

    async fn handle_messages(
        &self,
        messages: Vec<v1::InferRequest>,
    ) -> Result<Vec<v1::InferResponse>, Status> {
        let request = assemble_task_request(messages)?;
        let service_name = resolve_service_name(&self.hub, &request.meta)?;
        let task_name = request.task.clone();
        let correlation_id = request.correlation_id.clone();

        let mut meta = request.meta;
        meta.remove(SERVICE_META_KEY);
        let task_request = TaskRequest {
            payload: request.payload,
            payload_mime: request.payload_mime,
            meta,
        };

        let result = self
            .handle_task_request(service_name, task_name, task_request)
            .await
            .map_err(service_error_to_status)?;

        task_result_to_responses(correlation_id, result)
    }

    async fn handle_task_request(
        &self,
        service_name: String,
        task_name: String,
        request: TaskRequest,
    ) -> Result<TaskResult, ServiceError> {
        if !self.batcher.config().enabled || !is_batch_wire_eligible(&request) {
            return self.hub.handle(&service_name, &task_name, request).await;
        }

        let Some(task_key) = self.hub.batch_key(&service_name, &task_name, &request)? else {
            return self.hub.handle(&service_name, &task_name, request).await;
        };

        let batch_key = crate::service::BatchKey::new(format!(
            "service={service_name}\ntask={task_name}\n{}",
            task_key.as_str()
        ));
        let hub = Arc::clone(&self.hub);
        let batch_service = service_name.clone();
        let batch_task = task_name.clone();
        let batch_fn: BatchFn = Arc::new(move |requests| {
            let hub = Arc::clone(&hub);
            let service_name = batch_service.clone();
            let task_name = batch_task.clone();
            Box::pin(async move { hub.handle_batch(&service_name, &task_name, requests).await })
        });

        let receiver = self.batcher.submit(batch_key, request, batch_fn)?;
        receiver
            .await
            .map_err(|_| ServiceError::Unavailable("batch response channel closed".to_owned()))?
    }
}

#[tonic::async_trait]
impl v1::inference_server::Inference for HubGrpcService {
    type InferStream = ResponseStream<v1::InferResponse>;

    async fn infer(
        &self,
        request: Request<tonic::Streaming<v1::InferRequest>>,
    ) -> Result<Response<Self::InferStream>, Status> {
        let mut inbound = request.into_inner();
        let mut messages = Vec::new();

        while let Some(message) = inbound.message().await? {
            messages.push(message);
        }

        let responses = self.handle_messages(messages).await?;
        Ok(Response::new(item_stream(responses)))
    }

    async fn get_capabilities(
        &self,
        _request: Request<()>,
    ) -> Result<Response<v1::Capability>, Status> {
        let mut capabilities = self.hub.capabilities();
        if capabilities.is_empty() {
            return Err(Status::unavailable("no inference services registered"));
        }

        Ok(Response::new(
            self.reported_capability(capabilities.remove(0)),
        ))
    }

    type StreamCapabilitiesStream = ResponseStream<v1::Capability>;

    async fn stream_capabilities(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::StreamCapabilitiesStream>, Status> {
        let capabilities = self
            .hub
            .capabilities()
            .into_iter()
            .map(|capability| self.reported_capability(capability))
            .map(Ok)
            .collect::<Vec<_>>();
        let stream = tonic::codegen::tokio_stream::iter(capabilities);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn health(&self, _request: Request<()>) -> Result<Response<()>, Status> {
        Ok(Response::new(()))
    }
}

#[derive(Debug)]
struct AssembledInferRequest {
    correlation_id: String,
    task: String,
    payload: Bytes,
    payload_mime: String,
    meta: HashMap<String, String>,
}

struct InferChunk {
    seq: u64,
    offset: u64,
    payload: Vec<u8>,
}

fn assemble_task_request(messages: Vec<v1::InferRequest>) -> Result<AssembledInferRequest, Status> {
    let mut task = None;
    let mut correlation_id = None;
    let mut payload_mime = None;
    let mut meta = None;
    let mut expected_total = None;
    let mut chunks = Vec::with_capacity(messages.len());

    for message in messages {
        if message.task.is_empty() {
            return Err(Status::invalid_argument(
                "infer request task must not be empty",
            ));
        }
        if message.payload_mime.is_empty() {
            return Err(Status::invalid_argument(
                "infer request payload_mime must not be empty",
            ));
        }

        validate_stable_field("task", &mut task, &message.task)?;
        validate_stable_field(
            "correlation_id",
            &mut correlation_id,
            &message.correlation_id,
        )?;
        validate_stable_field("payload_mime", &mut payload_mime, &message.payload_mime)?;
        validate_stable_meta(&mut meta, &message.meta)?;

        if message.total > 0 {
            match expected_total {
                Some(total) if total != message.total => {
                    return Err(Status::invalid_argument(
                        "infer request chunks disagree on total chunk count",
                    ));
                }
                Some(_) => {}
                None => expected_total = Some(message.total),
            }
        }

        chunks.push(InferChunk {
            seq: message.seq,
            offset: message.offset,
            payload: message.payload,
        });
    }

    if chunks.is_empty() {
        return Err(Status::invalid_argument(
            "infer stream must contain at least one request",
        ));
    }

    if let Some(total) = expected_total {
        let actual = u64::try_from(chunks.len())
            .map_err(|_| Status::invalid_argument("infer request has too many chunks"))?;
        if actual != total {
            return Err(Status::invalid_argument(format!(
                "infer request expected {total} chunks, got {actual}"
            )));
        }
    }

    if chunks.iter().any(|chunk| chunk.offset != 0) {
        chunks.sort_by_key(|chunk| (chunk.offset, chunk.seq));
    } else {
        chunks.sort_by_key(|chunk| chunk.seq);
    }

    let payload_len = chunks.iter().map(|chunk| chunk.payload.len()).sum();
    let mut payload = Vec::with_capacity(payload_len);
    for chunk in chunks {
        payload.extend_from_slice(&chunk.payload);
    }

    Ok(AssembledInferRequest {
        correlation_id: correlation_id.unwrap_or_default(),
        task: task.expect("non-empty chunks should set task"),
        payload: Bytes::from(payload),
        payload_mime: payload_mime.expect("non-empty chunks should set payload_mime"),
        meta: meta.unwrap_or_default(),
    })
}

fn validate_stable_field(
    field_name: &str,
    current: &mut Option<String>,
    value: &str,
) -> Result<(), Status> {
    match current {
        Some(existing) if existing != value => Err(Status::invalid_argument(format!(
            "infer request chunks disagree on {field_name}"
        ))),
        Some(_) => Ok(()),
        None => {
            *current = Some(value.to_owned());
            Ok(())
        }
    }
}

fn validate_stable_meta(
    current: &mut Option<HashMap<String, String>>,
    value: &HashMap<String, String>,
) -> Result<(), Status> {
    match current {
        Some(existing) if existing != value => Err(Status::invalid_argument(
            "infer request chunks disagree on metadata",
        )),
        Some(_) => Ok(()),
        None => {
            *current = Some(value.clone());
            Ok(())
        }
    }
}

fn resolve_service_name(
    hub: &ServiceHub,
    meta: &HashMap<String, String>,
) -> Result<String, Status> {
    if let Some(service_name) = meta.get(SERVICE_META_KEY).filter(|name| !name.is_empty()) {
        return Ok(service_name.clone());
    }

    let service_names = hub.service_names();
    match service_names.as_slice() {
        [] => Err(Status::unavailable("no inference services registered")),
        [service_name] => Ok(service_name.clone()),
        _ => Err(Status::invalid_argument(format!(
            "request metadata must include `{SERVICE_META_KEY}` when multiple services are registered"
        ))),
    }
}

fn is_batch_wire_eligible(request: &TaskRequest) -> bool {
    normalized_meta(request.meta.get(META_INPUT_KIND)).as_deref() == Some(INPUT_KIND_TENSOR)
        && request.meta.get(META_PREPROCESS_SKIP).map(String::as_str) == Some("true")
}

fn normalized_meta(value: Option<&String>) -> Option<String> {
    value.map(|value| value.trim().to_ascii_lowercase())
}

fn task_result_to_responses(
    correlation_id: String,
    result: TaskResult,
) -> Result<Vec<v1::InferResponse>, Status> {
    let payload = result.payload;
    let total = payload.len().div_ceil(RESPONSE_CHUNK_SIZE).max(1);
    let total_u64 = u64::try_from(total)
        .map_err(|_| Status::internal("response has too many chunks to encode"))?;
    let result_schema = result.result_schema.unwrap_or_default();
    let payload_mime = result.payload_mime;
    let meta = result.meta;

    let mut responses = Vec::with_capacity(total);
    for seq in 0..total {
        let start = seq * RESPONSE_CHUNK_SIZE;
        let end = payload.len().min(start + RESPONSE_CHUNK_SIZE);
        let chunk = if payload.is_empty() {
            Vec::new()
        } else {
            payload.slice(start..end).to_vec()
        };
        responses.push(v1::InferResponse {
            correlation_id: correlation_id.clone(),
            is_final: seq + 1 == total,
            result: chunk,
            meta: meta.clone(),
            error: None,
            seq: u64::try_from(seq)
                .map_err(|_| Status::internal("response chunk index overflow"))?,
            total: total_u64,
            offset: u64::try_from(start)
                .map_err(|_| Status::internal("response chunk offset overflow"))?,
            result_mime: payload_mime.clone(),
            result_schema: result_schema.clone(),
        });
    }

    Ok(responses)
}

fn service_error_to_status(error: ServiceError) -> Status {
    match error {
        ServiceError::DuplicateService(_)
        | ServiceError::DuplicateTask(_)
        | ServiceError::InvalidArgument(_) => Status::invalid_argument(error.to_string()),
        ServiceError::ServiceNotFound { .. } | ServiceError::TaskNotFound { .. } => {
            Status::not_found(error.to_string())
        }
        ServiceError::Unavailable(_) => Status::unavailable(error.to_string()),
        ServiceError::Internal(_) => Status::internal(error.to_string()),
    }
}

fn item_stream<T>(items: Vec<T>) -> ResponseStream<T>
where
    T: Send + 'static,
{
    Box::pin(tonic::codegen::tokio_stream::iter(
        items.into_iter().map(Ok),
    ))
}

// ---- Type conversions: service types → protobuf ----

impl From<&TaskSpec> for v1::IoTask {
    fn from(spec: &TaskSpec) -> Self {
        Self {
            name: spec.name.clone(),
            input_mimes: spec.input_mimes.clone(),
            output_mimes: spec.output_mimes.clone(),
            limits: spec.limits.clone(),
            tensor_preprocess_id: spec.tensor_preprocess_id.clone(),
            tensor_batching_supported: spec.tensor_batching_supported,
        }
    }
}

impl From<ServiceCapability> for v1::Capability {
    fn from(capability: ServiceCapability) -> Self {
        Self {
            service_name: capability.service_name,
            model_ids: capability.model_ids,
            runtime: capability.runtime,
            max_concurrency: capability.max_concurrency,
            precisions: capability.precisions,
            extra: capability.extra,
            tasks: capability.tasks.iter().map(v1::IoTask::from).collect(),
            protocol_version: capability.protocol_version,
        }
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use bytes::Bytes;

    use crate::{
        daemon::{HubGrpcService, grpc::assemble_task_request, proto::home_native::v1},
        service::{
            BatchKey, DEFAULT_TENSOR_MIME, INPUT_KIND_TENSOR, InferenceService, META_INPUT_KIND,
            META_PREPROCESS_SKIP, ServiceCapability, ServiceError, ServiceHub, ServiceResult,
            TaskHandler, TaskRegistry, TaskRequest, TaskResult, TaskSpec,
        },
    };

    #[test]
    fn assemble_task_request_orders_chunks_by_sequence() {
        let request = assemble_task_request(vec![
            infer_chunk("abc", "echo_text", "text/plain", 1, 2, 0, b"llo"),
            infer_chunk("abc", "echo_text", "text/plain", 0, 2, 0, b"he"),
        ])
        .unwrap();

        assert_eq!(request.correlation_id, "abc");
        assert_eq!(request.task, "echo_text");
        assert_eq!(request.payload, Bytes::from_static(b"hello"));
        assert_eq!(request.payload_mime, "text/plain");
    }

    #[test]
    fn assemble_task_request_rejects_inconsistent_chunks() {
        let err = assemble_task_request(vec![
            infer_chunk("abc", "echo_text", "text/plain", 0, 2, 0, b"he"),
            infer_chunk("abc", "other_task", "text/plain", 1, 2, 0, b"llo"),
        ])
        .unwrap_err();

        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("task"));
    }

    #[tokio::test]
    async fn grpc_service_routes_to_hub_and_builds_final_response() {
        let mut hub = ServiceHub::new();
        hub.register(EchoService::new("echo", "echo_text")).unwrap();
        let service = HubGrpcService::new(
            std::sync::Arc::new(hub),
            crate::daemon::BatcherConfig::disabled(),
        );

        let responses = service
            .handle_messages(vec![infer_chunk(
                "abc",
                "echo_text",
                "text/plain",
                0,
                1,
                0,
                b"hello",
            )])
            .await
            .unwrap();
        let response = responses.into_iter().next().unwrap();

        assert_eq!(response.correlation_id, "abc");
        assert!(response.is_final);
        assert_eq!(response.result, b"hello");
        assert_eq!(response.result_mime, "text/plain");
        assert_eq!(
            response.meta.get("handled_by"),
            Some(&"echo_text".to_owned())
        );
    }

    #[tokio::test]
    async fn grpc_service_requires_service_metadata_when_multiple_services_are_registered() {
        let mut hub = ServiceHub::new();
        hub.register(EchoService::new("echo-a", "echo_text"))
            .unwrap();
        hub.register(EchoService::new("echo-b", "echo_text"))
            .unwrap();
        let service = HubGrpcService::new(
            std::sync::Arc::new(hub),
            crate::daemon::BatcherConfig::disabled(),
        );

        let err = service
            .handle_messages(vec![infer_chunk(
                "abc",
                "echo_text",
                "text/plain",
                0,
                1,
                0,
                b"hello",
            )])
            .await
            .unwrap_err();

        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("service"));
    }

    #[tokio::test]
    async fn grpc_service_batches_eligible_tensor_requests() {
        let mut hub = ServiceHub::new();
        hub.register(BatchEchoService::new("echo", "embed"))
            .unwrap();
        let service = HubGrpcService::new(
            std::sync::Arc::new(hub),
            crate::daemon::BatcherConfig {
                enabled: true,
                max_batch_size: 2,
                queue_latency: std::time::Duration::from_secs(60),
            },
        );

        let first = service.handle_messages(vec![tensor_chunk("a", "embed", b"one")]);
        let second = service.handle_messages(vec![tensor_chunk("b", "embed", b"two")]);
        let (first, second) = tokio::join!(first, second);

        let first = first.unwrap().into_iter().next().unwrap();
        let second = second.unwrap().into_iter().next().unwrap();
        assert_eq!(first.correlation_id, "a");
        assert_eq!(second.correlation_id, "b");
        assert_eq!(first.result, b"batch:one");
        assert_eq!(second.result, b"batch:two");
    }

    #[tokio::test]
    async fn grpc_service_bypasses_batcher_for_raw_requests() {
        let mut hub = ServiceHub::new();
        hub.register(BatchEchoService::new("echo", "embed"))
            .unwrap();
        let service = HubGrpcService::new(
            std::sync::Arc::new(hub),
            crate::daemon::BatcherConfig {
                enabled: true,
                max_batch_size: 8,
                queue_latency: std::time::Duration::from_millis(1),
            },
        );

        let response = service
            .handle_messages(vec![infer_chunk(
                "raw",
                "embed",
                "text/plain",
                0,
                1,
                0,
                b"hello",
            )])
            .await
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        assert_eq!(response.result, b"single:hello");
    }

    #[tokio::test]
    async fn grpc_service_bypasses_batcher_when_tensor_skip_is_false() {
        let mut request = tensor_chunk("skip-false", "embed", b"hello");
        request
            .meta
            .insert(META_PREPROCESS_SKIP.to_owned(), "false".to_owned());

        let mut hub = ServiceHub::new();
        hub.register(BatchEchoService::new("echo", "embed"))
            .unwrap();
        let service = HubGrpcService::new(
            std::sync::Arc::new(hub),
            crate::daemon::BatcherConfig {
                enabled: true,
                max_batch_size: 8,
                queue_latency: std::time::Duration::from_millis(1),
            },
        );

        let response = service
            .handle_messages(vec![request])
            .await
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        assert_eq!(response.result, b"single:hello");
    }

    #[test]
    fn maps_service_errors_to_grpc_status_codes() {
        let status =
            super::service_error_to_status(ServiceError::task_not_found("missing", vec![]));
        assert_eq!(status.code(), tonic::Code::NotFound);

        let status =
            super::service_error_to_status(ServiceError::Unavailable("offline".to_owned()));
        assert_eq!(status.code(), tonic::Code::Unavailable);
    }

    #[test]
    fn task_result_to_responses_chunks_large_payloads() {
        let payload = vec![7_u8; super::RESPONSE_CHUNK_SIZE + 3];
        let result = TaskResult::new(payload, "application/octet-stream")
            .with_meta("lumen.output.kind", "tensor")
            .with_meta("lumen.output.tensor.dtype", "fp32");

        let responses = super::task_result_to_responses("abc".to_owned(), result).unwrap();

        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].seq, 0);
        assert_eq!(responses[0].total, 2);
        assert_eq!(responses[0].offset, 0);
        assert!(!responses[0].is_final);
        assert_eq!(responses[1].seq, 1);
        assert_eq!(responses[1].total, 2);
        assert_eq!(responses[1].offset, super::RESPONSE_CHUNK_SIZE as u64);
        assert!(responses[1].is_final);
        assert_eq!(responses[0].meta, responses[1].meta);
        assert_eq!(responses[0].result_mime, responses[1].result_mime);
    }

    fn infer_chunk(
        correlation_id: &str,
        task: &str,
        payload_mime: &str,
        seq: u64,
        total: u64,
        offset: u64,
        payload: &[u8],
    ) -> v1::InferRequest {
        v1::InferRequest {
            correlation_id: correlation_id.to_owned(),
            task: task.to_owned(),
            payload: payload.to_vec(),
            meta: std::collections::HashMap::new(),
            payload_mime: payload_mime.to_owned(),
            seq,
            total,
            offset,
        }
    }

    fn tensor_chunk(correlation_id: &str, task: &str, payload: &[u8]) -> v1::InferRequest {
        let mut request = infer_chunk(correlation_id, task, DEFAULT_TENSOR_MIME, 0, 1, 0, payload);
        request
            .meta
            .insert(META_INPUT_KIND.to_owned(), INPUT_KIND_TENSOR.to_owned());
        request
            .meta
            .insert(META_PREPROCESS_SKIP.to_owned(), "true".to_owned());
        request
    }

    struct EchoService {
        name: String,
        tasks: TaskRegistry,
    }

    impl EchoService {
        fn new(name: &str, task_name: &str) -> Self {
            let mut tasks = TaskRegistry::new();
            tasks.register(EchoTask::new(task_name)).unwrap();
            Self {
                name: name.to_owned(),
                tasks,
            }
        }
    }

    impl InferenceService for EchoService {
        fn name(&self) -> &str {
            &self.name
        }

        fn tasks(&self) -> &TaskRegistry {
            &self.tasks
        }

        fn capability(&self) -> ServiceCapability {
            self.tasks
                .build_capability(&self.name, vec![format!("{}-model", self.name)], "cpu")
        }
    }

    struct EchoTask {
        spec: TaskSpec,
    }

    impl EchoTask {
        fn new(name: &str) -> Self {
            Self {
                spec: TaskSpec::new(name, "echo payload")
                    .with_input_mimes(["text/plain"])
                    .with_output_mime("text/plain"),
            }
        }
    }

    #[async_trait]
    impl TaskHandler for EchoTask {
        fn spec(&self) -> &TaskSpec {
            &self.spec
        }

        async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
            Ok(TaskResult::new(request.payload, "text/plain")
                .with_meta("handled_by", self.spec.name.clone()))
        }
    }

    struct BatchEchoService {
        name: String,
        tasks: TaskRegistry,
    }

    impl BatchEchoService {
        fn new(name: &str, task_name: &str) -> Self {
            let mut tasks = TaskRegistry::new();
            tasks.register(BatchEchoTask::new(task_name)).unwrap();
            Self {
                name: name.to_owned(),
                tasks,
            }
        }
    }

    impl InferenceService for BatchEchoService {
        fn name(&self) -> &str {
            &self.name
        }

        fn tasks(&self) -> &TaskRegistry {
            &self.tasks
        }

        fn capability(&self) -> ServiceCapability {
            self.tasks()
                .build_capability(&self.name, vec!["batch-echo-model".to_owned()], "cpu")
        }
    }

    struct BatchEchoTask {
        spec: TaskSpec,
    }

    impl BatchEchoTask {
        fn new(name: &str) -> Self {
            Self {
                spec: TaskSpec::new(name, "batch echo")
                    .with_input_mimes([DEFAULT_TENSOR_MIME, "text/plain"])
                    .with_output_mime("text/plain"),
            }
        }
    }

    #[async_trait]
    impl TaskHandler for BatchEchoTask {
        fn spec(&self) -> &TaskSpec {
            &self.spec
        }

        fn batch_key(&self, request: &TaskRequest) -> ServiceResult<Option<BatchKey>> {
            if request.meta.get(META_PREPROCESS_SKIP).map(String::as_str) != Some("true") {
                return Ok(None);
            }
            Ok(Some(BatchKey::new("batch-echo")))
        }

        async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> {
            let mut payload = b"single:".to_vec();
            payload.extend_from_slice(&request.payload);
            Ok(TaskResult::new(payload, "text/plain"))
        }

        async fn handle_batch(&self, requests: Vec<TaskRequest>) -> ServiceResult<Vec<TaskResult>> {
            Ok(requests
                .into_iter()
                .map(|request| {
                    let mut payload = b"batch:".to_vec();
                    payload.extend_from_slice(&request.payload);
                    TaskResult::new(payload, "text/plain")
                })
                .collect())
        }
    }

    #[test]
    fn converts_task_spec_to_proto_io_task() {
        let spec = TaskSpec::new("clip_image_embed", "embed image")
            .with_input_mimes(["image/jpeg", "image/png"])
            .with_output_mime("application/json;schema=embedding_v1")
            .with_limit("max_payload_size", "52428800")
            .with_metadata("not_in_proto", "kept_service_side")
            .with_tensor_fast_path("siglip2_base_patch16_224_image_v1", true);

        let proto = v1::IoTask::from(&spec);

        assert_eq!(proto.name, "clip_image_embed");
        assert_eq!(proto.input_mimes, vec!["image/jpeg", "image/png"]);
        assert_eq!(
            proto.output_mimes,
            vec!["application/json;schema=embedding_v1"]
        );
        assert_eq!(
            proto.limits.get("max_payload_size"),
            Some(&"52428800".to_owned())
        );
        assert!(!proto.limits.contains_key("not_in_proto"));
        assert_eq!(
            proto.tensor_preprocess_id,
            "siglip2_base_patch16_224_image_v1"
        );
        assert!(proto.tensor_batching_supported);
    }

    #[test]
    fn converts_service_capability_to_proto_capability() {
        let spec = TaskSpec::new("ocr", "run ocr")
            .with_input_mimes(["image/jpeg"])
            .with_output_mime("application/json;schema=ocr_v1");
        let capability = ServiceCapability::new(
            "ocr-service",
            vec!["ocr-model".to_owned()],
            "onnxrt-cpu",
            vec![spec],
        )
        .with_max_concurrency(2)
        .with_precisions(["fp32"])
        .with_extra("device", "cpu");

        let proto = v1::Capability::from(capability);

        assert_eq!(proto.service_name, "ocr-service");
        assert_eq!(proto.model_ids, vec!["ocr-model"]);
        assert_eq!(proto.runtime, "onnxrt-cpu");
        assert_eq!(proto.max_concurrency, 2);
        assert_eq!(proto.precisions, vec!["fp32"]);
        assert_eq!(proto.extra.get("device"), Some(&"cpu".to_owned()));
        assert_eq!(proto.tasks.len(), 1);
        assert_eq!(proto.tasks[0].name, "ocr");
        assert_eq!(proto.protocol_version, "1.0");
    }
}
