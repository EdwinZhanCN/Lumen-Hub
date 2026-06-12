# Lumen-Hub Burn Runtime Memory Decision

## 背景

在 Metal 后端运行 `lumen-hub` 时，启动 warmup 后曾观察到 `Physical footprint` 常驻约 `5.3G`。继续调用 `semantic_text_embed` 时，早期实现还会看到 footprint 继续抬升，甚至在使用过程中到达 `9G` 到 `10G`。

本轮排查使用的典型构建命令：

```bash
cargo run --release --no-default-features --features metal,siglip,ppocr,insightface,clip -p lumen-hub -- --config ~/.lumen/config.yaml
```

服务配置包含 SigLIP、PP-OCR、InsightFace，精度为 `fp16q8`。CPU 后端已从 Burn legacy NdArray 切换为 Burn Flex。

## 现象拆分

这次问题实际由两个不同原因组成：

1. Warmup 后常驻内存过高
2. 多次推理后 memory pool 继续扩张

它们都表现为 macOS `vmmap` 中的 graphics memory 变大，但根因和修复方式不同。

## 常驻内存过高

根因是 Burn 0.21 的 Metal/Vulkan/wgpu 后端走 CubeCL/wgpu。CubeCL wgpu runtime 默认使用 `MemoryConfiguration::SubSlices`。该策略会创建较大的 page，并在 page 内切 slice 复用 GPU buffer。

在 CubeCL 0.10 的默认实现中，`SubSlices` 的 sliced pools 使用 `dealloc_period: None`。这意味着 warmup 推理过程中触发的大 page 会长期留在 pool 里，即使当前没有活跃请求，macOS 仍会把这些 GPU allocations 计入 `owned unmapped (graphics)`。

旧状态下观察到：

```text
Physical footprint:        5.3G
owned unmapped (graphics): 4.8G
```

决策：在 Metal/Vulkan/wgpu 后端启动时显式初始化 Burn wgpu runtime，并使用：

```rust
RuntimeOptions {
    memory_config: MemoryConfiguration::ExclusivePages,
    ..RuntimeOptions::default()
}
```

`ExclusivePages` 的 pool 使用 bucketed exclusive allocations，并带有回收周期。结合每次推理后的 `Backend::memory_cleanup()` 和 `sync()`，warmup 期间的大量临时 allocation 不再长期压在 sliced page pool 中。

新状态下观察到：

```text
Physical footprint:        1.8G
owned unmapped (graphics): 1.3G
```

常驻 footprint 下降约 `3.5G`。

## Memory Pool 继续扩张

CubeCL stream 与线程上下文相关。默认 `streaming.max_streams` 为 `128`，并且每个 stream 拥有独立 memory pool。

旧实现中，模型推理通过 Tokio blocking pool 执行。不同请求可能落到不同 blocking worker 线程，进而触发多个 CubeCL stream。每个 stream 都会建立并增长自己的 memory pool，于是表现为每次推理后 graphics memory 继续扩大。

决策：

1. 引入 dedicated inference worker，将模型构建和推理固定到单个 OS 线程。
2. 启动最早阶段设置 `CubeClRuntimeConfig.streaming.max_streams = 1`。
3. `ServiceHub` 层保留单 permit semaphore，确保同一进程内只有一个推理任务在执行。
4. 每次 inference worker job 结束后调用 backend cleanup 和 sync。

这些改动让 CubeCL 长期只使用一个 stream/pool，防止多线程调度造成 pool 复制扩张。

累计多次 `semantic_text_embed` 后观察到：

```text
Physical footprint:        1.8G
owned unmapped (graphics): 1.3G
```

未再出现线性增长。

## CPU 后端

CPU 后端从 Burn legacy NdArray 切换到 Burn Flex，并显式启用 `rayon` 和 `simd`：

```toml
cpu = ["burn/flex", "burn/simd", "dep:burn-flex"]
burn-flex = { version = "0.21.0", default-features = false, features = ["std", "simd", "rayon"], optional = true }
```

原因是 NdArray 后端不适合作为后续 CPU baseline。Flex 是 Burn 0.21 推荐的新 CPU 后端，能使用多线程和 SIMD 执行路径。

CPU/Flex 实测没有类似 Metal graphics pool 常驻问题；多次 text 推理后未观察到线性增长。

## 性能取舍

`max_streams = 1` 和 dedicated worker 会降低高并发吞吐潜力，因为多个请求不能并行提交到多个 GPU stream。

`ExclusivePages` 可能比 `SubSlices` 增加 allocation/free 管理开销，因为它减少了大 page 切片复用。

但 Lumen-Hub 的目标是本地常驻 ML 微服务，不是高吞吐 GPU batch server。当前更重要的是稳定、低常驻、低泄露风险。实测单请求 latency 没有异常波动：

```text
semantic_text_embed   avg 28.35ms  p95 29.92ms
semantic_image_embed  avg 68.28ms  p95 72.28ms
ocr                   avg 433.87ms p95 461.98ms
face_recognition      avg 419.75ms p95 426.24ms
```

因此当前默认选择是：

```text
single inference worker
CubeCL max_streams = 1
WGPU MemoryConfiguration::ExclusivePages
cleanup_memory + sync after each inference job
```

## 后续方向

如果未来需要高吞吐部署，可以将 GPU runtime 策略下沉为配置项：

```yaml
runtime:
  gpu:
    memory_strategy: exclusive_pages
    max_streams: 1
    inference_workers: 1
```

本地常驻模式默认使用 `exclusive_pages + max_streams=1`。高吞吐模式可以单独 A/B `SubSlices + max_streams>1 + batching`，但必须配套内存上限和长时间 soak test。

另一个独立问题是 SigLIP text embedding table 仍无法安全量化到 CubeCL q8 gather 路径，因为 Burn/CubeCL 当前 `q_gather` 尚未实现。该问题影响模型权重常驻体积，但不是本轮 graphics pool 扩张的根因。
