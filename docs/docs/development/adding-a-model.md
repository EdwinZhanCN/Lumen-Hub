---
sidebar_position: 1
---

# 添加新模型

本文档描述如何在 Lumen Hub 中集成一个新的推理模型。

## 步骤清单

### 1. 创建模型目录

```
crates/lumen-hub/src/models/<name>/
  mod.rs
  factory.rs
  service.rs
  pipeline.rs
  nodes.rs     # （如需自定义后处理）
  task.rs
```

### 2. 实现 ModelFactory

```rust
// factory.rs
pub struct MyModelFactory;

impl ModelFactory for MyModelFactory {
    fn build(
        &self,
        config: &str,
        context: Arc<MLContext>,
    ) -> ServiceResult<InferenceServiceInstance> {
        // 1. 解析 model_info.json
        // 2. 加载模型权重
        // 3. 创建 MyModelService
    }
}
```

### 3. 实现 InferenceService

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

### 4. 构建 Pipeline

```rust
// pipeline.rs
pub fn build_my_pipeline(
    model: Arc<MLModel>,
    context: &MLContext,
) -> ServiceResult<MLPipeline> {
    let mut pipeline = MLPipeline::new();
    pipeline.add_node(model); // forward node
    // 可选：添加后处理节点
    Ok(pipeline)
}
```

### 5. 实现 TaskHandler

```rust
// task.rs
pub struct MyModelTask { ... }

#[async_trait]
impl TaskHandler for MyModelTask {
    fn spec(&self) -> &TaskSpec { &self.spec }

    // 如果要支持批处理，覆盖这两个方法：
    fn batch_key(&self, request: &TaskRequest) -> ServiceResult<Option<BatchKey>> { ... }
    async fn handle_batch(&self, requests: Vec<TaskRequest>) -> ServiceResult<Vec<TaskResult>> { ... }

    async fn handle(&self, request: TaskRequest) -> ServiceResult<TaskResult> { ... }
}
```

### 6. 注册模块

在 `crates/lumen-hub/src/models/mod.rs`：

```rust
#[cfg(feature = "my_model")]
pub mod my_model;
```

### 7. 添加 Feature Gate

在 `crates/lumen-hub/Cargo.toml`：

```toml
[features]
my_model = []
```

### 8. 注册服务名

在 `lumen-schema` 的配置校验逻辑中注册新的服务名（如需配置驱动加载）。

## 支持批处理

如果要支持批处理，需要：

1. **定义兼容性规则**：在 `batch_key()` 中返回包含模型 ID、张量形状、数据类型的 key
2. **实现拼接逻辑**：在 `handle_batch()` 中沿 batch 维度拼接多个请求的输入张量
3. **拆分输出**：将单个 batch 输出拆分为每个请求的独立结果

参考 CLIP/SigLIP 的 `batched_tensor_packets()` 实现。
