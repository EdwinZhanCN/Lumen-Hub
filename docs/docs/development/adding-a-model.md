---
sidebar_position: 1
---

# Adding a New Model

This document describes how to integrate a new inference model into Lumen Hub.

## Step-by-Step

### 1. Create the Model Directory

```
crates/lumen-hub/src/models/<name>/
  mod.rs
  factory.rs
  service.rs
  pipeline.rs
  nodes.rs     # (optional custom postprocessing)
  task.rs
```

### 2. Implement ModelFactory

```rust
// factory.rs
pub struct MyModelFactory;

impl ModelFactory for MyModelFactory {
    fn build(
        &self,
        config: &str,
        context: Arc<MLContext>,
    ) -> ServiceResult<InferenceServiceInstance> {
        // 1. Parse model_info.json
        // 2. Load model weights
        // 3. Create MyModelService
    }
}
```

### 3. Implement InferenceService

```rust
// service.rs
pub struct MyModelService {
    name: String,
    registry: Arc<TaskRegistry>,
}

impl InferenceService for MyModelService {
    fn name(&self) -> &str { &self.name }
    fn capability(&self) -> ServiceCapability { ... }
    fn tasks(&self) -> Arc<TaskRegistry> { Arc::clone(&self.registry) }
}
```

### 4. Build Pipeline

```rust
// pipeline.rs
pub fn build_my_pipeline(
    model: Arc<MLModel>,
    context: &MLContext,
) -> ServiceResult<MLPipeline> {
    let mut pipeline = MLPipeline::new();
    pipeline.add_node(model); // forward node
    // Optional: add postprocessing nodes
    Ok(pipeline)
}
```

### 5. Implement TaskHandler

```rust
// task.rs
pub struct MyModelTask { ... }

#[async_trait]
impl TaskHandler for MyModelTask {
    fn spec(&self) -> &TaskSpec { &self.spec }

    // To support batching, override these two methods:
    fn batch_key(&self, request: &TaskRequest) -> ServiceResult<Option<BatchKey>> { ... }
    async fn handle_batch(&self, requests: Vec<TaskRequest>) -> ServiceResult<Vec<TaskResult>> { ... }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> { ... }
}
```

### 6. Register the Module

In `crates/lumen-hub/src/models/mod.rs`:

```rust
#[cfg(feature = "my_model")]
pub mod my_model;
```

### 7. Add a Feature Gate

In `crates/lumen-hub/Cargo.toml`:

```toml
[features]
my_model = []
```

### 8. Register the Service Name

Register the new service name in `lumen-schema`'s config validation logic (if you want config-driven loading).

## Supporting Batching

To support batching, you need:

1. **Define compatibility rules**: Return a key containing model ID, tensor shape, and dtype in `batch_key()`
2. **Implement concatenation logic**: In `handle_batch()`, concatenate input tensors from multiple requests along the batch dimension
3. **Split outputs**: Split the single batch output back into per-request results

See CLIP/SigLIP's `batched_tensor_packets()` implementation for reference.
