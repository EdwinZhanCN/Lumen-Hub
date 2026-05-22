---
sidebar_position: 2
---

# Request Lifecycle

This document traces an inference request from gRPC ingress to response.

## Stage 1: Streaming Assembly

The gRPC client sends requests over the `Infer` bidirectional stream (`proto/ml_service.proto`). Each `InferRequest` chunk carries:

- `correlation_id` — Shared by all chunks of the same request
- `task` — Registered task name
- `payload` / `payload_mime` — Body and MIME type
- `meta` — String map (`service`, `lumen.input.kind`, `lumen.preprocess.skip`, etc.)
- `seq` / `total` — Optional chunk index and count

In `grpc.rs`, the server collects chunks for the same `correlation_id`, concatenates payloads, merges metadata, and calls `assemble_task_request` to produce a `TaskRequest`.

```mermaid
sequenceDiagram
    participant Client
    participant gRPC as HubGrpcService
    participant Batcher
    participant Hub as ServiceHub
    participant Registry as TaskRegistry
    participant Task as TaskHandler

    Client->>gRPC: InferenceStream chunks
    gRPC->>gRPC: assemble_task_request()
    gRPC->>Batcher: is_batch_wire_eligible?
    alt batch eligible
        Batcher->>Batcher: submit(key, request)
        Batcher->>Hub: handle_batch(requests)
    else not batch eligible
        gRPC->>Hub: handle(request)
    end
    Hub->>Registry: handle(task_name, request)
    Registry->>Task: handle(request)
    Task-->>Registry: TaskResult
    Registry-->>Hub: TaskResult
    Hub-->>gRPC: TaskResult
    gRPC-->>Client: InferResponse stream
```

## Stage 2: Batching Decision

`HubGrpcService::handle_task_request` checks whether to batch:

```mermaid
flowchart TD
    A[TaskRequest] --> B{is_batch_wire_eligible?}
    B -->|No| C[hub.handle]
    B -->|Yes| D{hub.batch_key}
    D -->|None| C
    D -->|Some key| E[batcher.submit]
    E --> F[per-key queue wait]
    F --> G{max_batch_size or timeout?}
    G -->|Trigger| H[hub.handle_batch]
    G -->|Keep waiting| F
    C --> I[TaskResult]
    H --> I
```

Batching eligibility (`grpc.rs:is_batch_wire_eligible`):

1. `server.batching.enabled == true`
2. `lumen.input.kind == "tensor"`
3. `lumen.preprocess.skip == "true"`

Only **preprocessed tensors** are batched. Raw images/text are not — their preprocessing cost is uneven.

## Stage 3: Service Routing

`ServiceHub::handle(service_name, task_name, request)`

1. Look up `InferenceService` by `service_name`
2. Call `service.tasks().handle(task_name, request)`
3. `TaskRegistry` looks up `TaskHandler` by `task_name`

## Stage 4: Task Execution

SigLIP image embedding example:

```
SiglipImageEmbedTask::handle(request)
  ├── image MIME?
  │     → preprocess_image → MLPacket → pipeline.run()
  └── tensor + preprocess.skip?
        → tensor_request_to_packet → pipeline.run()

pipeline.run(packets)
  → ONNX/MNN forward
  → L2 normalize (when configured)
  → embedding_v1 JSON response
```

## Stage 5: Response encoding

After `TaskResult` is returned, `grpc.rs` encodes `InferResponse` chunks (large payloads split at 4 MB).

## Batching Path

Multiple requests are grouped by `BatchKey` in the `Batcher`. When `max_batch_size` or `queue_latency` fires:

```mermaid
flowchart LR
    subgraph Queue[per-BatchKey Queue]
        R1[req1]
        R2[req2]
    end
    Queue --> BatchFn
    BatchFn --> Hub[hub.handle_batch]
    Hub --> Task[handle_batch]
    Task --> Concat[Concat tensors along batch dim]
    Concat --> Forward[Single ONNX forward]
    Forward --> Split[Split results]
    Split --> Results[Vec&lt;TaskResult&gt;]
```

## Key Design Decision

**Why is batching in the daemon layer instead of the service layer?**

The service layer is protocol-agnostic — it only receives `TaskRequest` and returns `TaskResult`. Queue management and timeout triggering are orchestration logic belonging to the transport layer. The service layer just needs to provide `batch_key()` ("I can batch") and `handle_batch()` ("run this batch").
