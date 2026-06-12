# Lumen Thin Tensor Input Contract

## Status

Implemented across Lumen-Hub, Lumen-SDK, and Lumilio Photos.

This document records the stable wire contract and integration rules for client-side tensor fast paths. It replaces the earlier implementation plan and should be treated as the reference for future changes.

## Goal

Eliminate Tensor Path drift between Lumen-Hub, Lumen-SDK, and Lumilio Photos without turning the protobuf contract into a model preprocessing configuration language.

The protocol answers only two client-side questions:

1. Does this task expose a client-side tensor fast path?
2. If yes, which versioned SDK preprocessor should produce that tensor, and can the Hub batch those tensor requests?

Everything else stays out of the wire contract.

## Design Principle

Raw input does not require a preprocessing contract.

For raw image/text inputs, the client sends only a payload and MIME type. Lumen-Hub owns decode, resize, crop, normalization, tokenization, and all model-specific preprocessing.

Tensor input is different: the client must produce a tensor that exactly matches Hub expectations. Therefore, tensor input needs a stable identifier, but not a full preprocessing schema.

The stable identifier is `tensor_preprocess_id`.

The SDK owns a registry of known preprocessors keyed by `tensor_preprocess_id`. If the SDK does not recognize an ID, the caller must fall back to the raw path.

## Wire Contract

`IOTask` is extended with two fields:

```proto
message IOTask {
  string name = 1;
  repeated string input_mimes = 2;
  repeated string output_mimes = 3;
  map<string, string> limits = 4;

  // Empty means this task has no client-side tensor fast path.
  string tensor_preprocess_id = 5;

  // True means tensor requests for this task can be dynamically batched by the Hub.
  bool tensor_batching_supported = 6;
}
```

No general-purpose `ImagePreprocessContract` belongs in protobuf. Resize, crop, normalization, tokenizer, tensor layout details, and model-specific validation are implementation details of versioned SDK preprocessors and Hub validators.

## Field Semantics

### `tensor_preprocess_id`

A stable, versioned identifier for a tensor fast path.

Examples:

- `siglip2_base_patch16_224_image_v1`
- `siglip2_so400m_patch14_384_image_v1`
- `bioclip2_224_image_v1`
- `ppocr_det_v1`
- `insightface_scrfd_640_letterbox_v1`

An empty value means the task only supports raw client input from the client's perspective.

The ID is the contract. It implies the exact preprocessing algorithm, output tensor dtype/layout/shape, request metadata, and compatibility expectations.

### `tensor_batching_supported`

A hint that tensor requests for this task can be dynamically batched by Lumen-Hub.

Clients do not need `max_batch_size`. Batch size, queue latency, and scheduling are Hub runtime policy. Clients should use the tensor path when available and issue normal concurrent requests.

## Current Task Reporting

| Service | Task | `tensor_preprocess_id` | `tensor_batching_supported` | Notes |
|---|---|---|---:|---|
| `siglip` | `semantic_text_embed` | empty | false | Raw text only. |
| `siglip` | `semantic_image_embed` | `siglip2_base_patch16_224_image_v1` or `siglip2_so400m_patch14_384_image_v1` | true | Selected from model image tensor shape. |
| `bioclip` | `bioclip_classify` | `bioclip2_224_image_v1` | true | Uses BioCLIP/CLIP-style 224 image preprocessing. |
| `ocr` | `ocr` | empty | false | Raw-first. Do not expose tensor path until benchmarks justify it. |
| `face` | `face_recognition` | empty | false | Raw-first. Do not expose tensor path until benchmarks justify it. |

OCR and face may still contain internal tensor handling, but they should not advertise a client-side tensor fast path until the SDK has a deliberate versioned preprocessor and the Photos workload shows a benefit.

## Hub Responsibilities

Lumen-Hub is the source of truth for task capability reporting.

Hub must:

1. Report `tensor_preprocess_id` only for tasks that are safe for client-side tensor input.
2. Report `tensor_batching_supported` only when tensor requests for that task can be dynamically batched.
3. Keep raw input behavior independent from `tensor_preprocess_id`.
4. Validate tensor requests against the exact expected preprocess ID, dtype, layout, shape, byte order, and payload length.
5. Keep preprocessing internals out of protobuf.

Implemented Hub touchpoints:

- `crates/lumen-hub/proto/ml_service.proto`
- `crates/lumen-hub/src/daemon/home_native.v1.rs`
- `crates/lumen-hub/src/service/task.rs`
- `crates/lumen-hub/src/daemon/grpc.rs`
- `crates/lumen-hub/src/service/tensor.rs`
- `crates/lumen-hub/src/models/siglip/task.rs`
- `crates/lumen-hub/src/models/bioclip/task.rs`

## SDK Responsibilities

Lumen-SDK owns client-side interpretation of tensor fast-path capability.

SDK must:

1. Keep generated protobuf bindings in sync with `IOTask`.
2. Preserve task availability behavior based on task name.
3. Expose helpers for tensor path detection:

```go
func (t TaskContract) HasTensorPath() bool
func (t TaskContract) TensorPreprocessID() string
func (t TaskContract) TensorBatchingSupported() bool
```

4. Provide a built-in preprocessor registry keyed by `tensor_preprocess_id`.
5. Return raw-path fallback behavior when a task has no tensor ID or the SDK does not know the ID.

Implemented SDK touchpoints:

- `proto/ml_service.proto`
- `proto/ml_service.pb.go`
- `pkg/types/task_contract_helpers.go`
- `pkg/types/tensor_preprocessor.go`
- `pkg/types/task_contract.go`
- `pkg/client/client.go`
- `pkg/client/lumen_balancer.go`
- `pkg/client/pool.go`

Current built-in preprocessor IDs:

- `siglip2_base_patch16_224_image_v1`
- `siglip2_so400m_patch14_384_image_v1`
- `bioclip2_224_image_v1`

## Photos Responsibilities

Lumilio Photos must not understand model preprocessing internals.

Photos should:

1. Query capabilities through the SDK-backed Lumen service.
2. For each ML task, check whether a known tensor preprocessor exists.
3. Use tensor path only when:
   - the task reports `tensor_preprocess_id`, and
   - the SDK registry recognizes the ID.
4. Otherwise use the raw path exactly as before.

Implemented Photos touchpoint:

- `Lumilio-Photos/server/internal/service/lumen_service.go`

Current Photos behavior:

- `semantic_image_embed`: tries SDK tensor path first, falls back to raw.
- `bioclip_classify`: tries SDK tensor path first, falls back to raw.
- `ocr`: raw path.
- `face_recognition`: raw path.

## Fallback Rule

The fallback rule is mandatory:

```go
contract, ok := client.FindTaskContract(taskName)
if !ok || !contract.HasTensorPath() {
    // use raw path
}

preprocessor, ok := registry.Lookup(contract.TensorPreprocessID())
if !ok {
    // use raw path
}

// use tensor path
```

Unknown preprocess IDs must not be treated as errors by high-level clients such as Photos. They indicate Hub/SDK version skew and should degrade to raw input.

## Dynamic Batching Rule

`tensor_batching_supported = true` means clients may issue normal concurrent tensor requests and Hub may batch compatible requests internally.

Clients must not depend on:

- batch size
- queue latency
- worker count
- scheduling strategy

Those remain Hub runtime policy.
