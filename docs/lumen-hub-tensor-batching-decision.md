# Lumen-Hub Tensor Batching Decision

## 背景

Tensor path 引入后,SigLIP `semantic_image_embed` 的 e2e benchmark(COCO sample-500,Metal,`fp16q8`,M2 Pro)出现了两个问题:

1. tensor path 吞吐整体不如 raw path;
2. `concurrency=4` 时 tensor path 出现稳定退化(~10 img/s),且 batch sweep 呈非单调(c=2 好,c=3/4/5 差,c=6/7 回升)。

本文记录根因定位过程与最终决策。结论先行:

```text
daemon dynamic batching 默认关闭(default_batching_enabled = false)
吞吐依靠并发请求在单线程 inference worker 上的 batch-1 流水线
ServiceHub 移除单 permit semaphore,CPU 预处理与 GPU 推理重叠
inference worker 先回复再 cleanup_memory,清理移出请求关键路径
```

## 根因:Metal/CubeCL 上 batch>1 是纯负收益

在 batcher 中加入 `batch_size` + `elapsed_ms` debug 日志后,实测每个 batch size 的真实 GPU 耗时(同一进程、稳定复现):

| batch | 每 batch 耗时 | 摊到每张 |
|---:|---:|---:|
| 1 | 26–50 ms | ~35–40 ms |
| 2 | ~190 ms | ~95 ms |
| 4 | ~385 ms | ~96 ms |
| 8 | ~360 ms | ~45 ms |

batch=2/4 比串行单张慢约 2.4 倍,batch=8 也没有摊销收益。该曲线与 Burn 0.21 / CubeCL 0.10 在 Metal 上对特定 GEMM 形状(`M = batch × 196` tokens)的 kernel 选择有关,属于上游后端问题,不是 batcher 实现问题。

### 排除矩阵(全部 A/B 实测,batch-2 耗时为判据)

| 嫌疑因素 | 实验 | batch-2 | 结论 |
|---|---|---:|---|
| 基线(ExclusivePages + cleanup + streams=1 + fp16q8 + fusion) | — | ~190 ms | — |
| memory decision 内存策略 | `LUMEN_GPU_MEMORY_STRATEGY=subslices` + `LUMEN_SKIP_INFER_CLEANUP=1` | 189 ms | 无关 |
| `max_streams = 1` | `LUMEN_GPU_MAX_STREAMS=128` | 212 ms | 无关(单线程 worker 本来只用一个 stream) |
| q8 量化 kernel | `precision: fp16` | 195 ms | 无关 |
| Burn fusion JIT | 去掉 `burn/fusion` feature 重编 | 194 ms | 无关 |

也排除了 autotune/编译开销:数百个相同 shape 的 batch 稳定在同一耗时,是稳态而非首次编译。

剩下的指向 CubeCL/wgpu Metal 的核心 kernel 本身。耗时形状很说明问题(fp16):`t(B) ≈ ~160 ms 固定开销 + ~27 ms × B`(B≥2):b2=195、b3=226、b8=376 都吻合。即 batch>1 让某个(些)算子切换到一条带巨大固定成本的执行路径;路径内部的边际成本(27 ms/张)其实低于 batch-1 的 40 ms——batch-8 的"看起来还行"(47 ms/张)只是把同一面墙摊薄了,不是 batching 真正生效。

`LUMEN_GPU_MEMORY_STRATEGY`、`LUMEN_SKIP_INFER_CLEANUP`、`LUMEN_GPU_MAX_STREAMS` 三个环境变量留作后续后端实验的开关。低常驻内存策略可以保留,没有为它付出 batch 性能代价。

注意一个测量陷阱:此前 sweep 里 "c=2 吞吐最好(~22 img/s,75–85 ms)" 实际上**不是 batch-2**。日志显示 c=2 时 batcher 全程在打 batch-1 ping-pong(两个 client 错峰到达,2 ms 窗口永远凑不齐第二个请求),吞吐来自流水线重叠而非 batching。真正的 batch-2 没有快过 190 ms。

## 决策与实测

测试矩阵:COCO sample-500,warmup 20,`semantic_image_embed`,Metal `fp16q8`。

关闭 batching、移除 hub 级 semaphore 后:

| Mode | C | 吞吐 (img/s) | 改动前 (img/s) |
|---|---:|---:|---:|
| raw | 1 | 21.9 | 21.8 |
| raw | 4 | 25.4 | 20.5 |
| raw | 8 | 25.4 | 20.6 |
| raw | 16 | 25.4 | 20.7 |
| tensor | 1 | 19.4 | 12.2 |
| tensor | 4 | 23.6 | 9.8–10.5 |
| tensor | 8 | 24.6 | 15.4–20.0 |
| tensor | 16 | 24.5 | 21.1–22.2 |

Hub 常驻内存(单 siglip 服务):`Physical footprint ~1.1G`,`graphics ~940M`,与 batching 开启时持平或略低(batch 激活内存更小)。

### 1. dynamic batching 默认关闭

`lumen-schema` 的 `default_batching_enabled()` 改为 `false`。任何 batch size 的摊销成本都不低于 batch-1,batching 只会把请求绑在坏 kernel 形状上。配置项保留,未来后端修复或换 CUDA 等后端时可按部署重新开启并 benchmark。

`HubGrpcService` 在 batcher 关闭时将 capability 上报中的 `tensor_batching_supported` 掩蔽为 `false`,与 thin tensor input contract 的语义保持一致(该字段是 Hub runtime policy 的提示)。

### 2. 移除 ServiceHub 单 permit semaphore

GPU 提交安全性本来就由单线程 `inference_worker` 保证(CubeCL stream 绑定线程,worker 串行执行所有 forward)。hub 级 semaphore 是冗余的,而且把 raw path 的 CPU 解码/预处理也串行化了,GPU 在每个请求解码期间空转。移除后 raw path 饱和吞吐 20.5 → 25.4 img/s(+24%),CPU 预处理与 GPU 推理重叠,GPU 持续被喂满。

### 3. inference worker 先回复再 cleanup

`cleanup_memory`(含 device sync)原先在 `reply.send` 之前执行,即每个请求的关键路径上。改为先回复再清理:worker 是单线程,下一个 job 仍然在 cleanup 完成后才开始,内存回收保证不变,但调用方不再为清理买单。

### 4. SDK 预处理顺序修正(Lumen-SDK)

`imageTensorPreprocessor` 原先在**原始分辨率**上做逐像素 `At()/RGBA()` 接口调用转 NRGBA,然后才 resize。改为直接把解码图喂给 `imaging.Resize`(对 JPEG 的 `*image.YCbCr` 有 fast path),并对 resize 输出的 `*image.NRGBA` 走 `Pix` 直读转换。输出与旧实现逐字节一致(已用对照测试验证),preprocess p95 从 ~35 ms 降到 ~13 ms。

## 预处理对齐(Photos → SDK → Hub)

固化推理链路时发现并修复了三处预处理 drift,全部由新的 Photos 端 conformance 测试
(`Lumilio-Photos/server/internal/service/lumen_tensor_conformance_test.go`)守护:

1. **Hub/SDK 的 SigLIP resize 语义偏离训练时预处理**。HF `SiglipImageProcessor`
   (`do_center_crop=false`)是一步直接 resize 到 224×224(squash);Hub 与 SDK 此前
   实现为"短边 224 → 再拉伸 224×224"两步,多了一次训练时不存在的重采样。已统一改为:
   `do_center_crop=true` → 短边 resize + 中心裁剪(CLIP/BioCLIP);否则一步直接 resize。
2. **Photos 的 vips 预处理用错了几何与 kernel**。原 `mlOptions` 对 semantic 用中心裁剪
   (应为 squash),且 vips thumbnail 路径固定 lanczos3(契约为 bilinear/bicubic)。
   新增 `imaging.DecodeRGBResizeExact`(bilinear,SigLIP)与
   `imaging.DecodeRGBShortestEdgeCenterCrop`(bicubic,BioCLIP)。
3. **SDK 忽略调用方已解码的像素**。`Preprocess` 此前优先 `Encoded`,Photos 用 vips
   产出的 224×224 像素被丢弃后在 Go 里二次解码整张 thumbnail。现在尺寸匹配的
   `Data` 优先,Photos 链路上 SDK 预处理只剩 normalize + NCHW(~1 ms)。

对齐后 conformance(COCO 12 张抽样,vips tensor 路径 vs Hub raw):下采样 worst
cosine `0.9992`(gate 0.995),上采样(源图小于 224,罕见退化样本)`0.9798`
(gate 0.97;libvips 与 image crate 的上采样半像素约定不同,已知差异)。
SDK Go 路径 vs Hub raw:cosine `0.9997+`,max_abs_diff ≤ `0.0038`。

对齐后在全家桶 Hub(siglip + ppocr + insightface + bioclip/TreeOfLife200MCore 同时
加载,`examples/photos-full.yaml`)上的吞吐:SigLIP raw 与 tensor 均为 ~25 img/s
(c=2 即饱和);BioCLIP ~4 img/s;混合并发(semantic c=2 + bioclip c=1)时 semantic
降至 ~6 img/s(单 inference worker 的 head-of-line 排队,既有取舍)。Hub 常驻
`Physical footprint ~2.2G`(峰值 4.4G,出现在启动 warmup),16GB 设备安全。

注意:resize 语义修正会让新生成的 embedding 与旧索引存在 ~0.99 cosine 级别的偏移,
属于一次性预处理修正;Photos 的 `MLPreprocessVersion` 维持 V1,不强制全库重建。

## 残留问题

1. **上游 kernel 问题**:batch=2/4 在 Metal/CubeCL 上的 2.4 倍退化值得向 Burn/CubeCL 上报最小复现(`[B,3,224,224]` ViT forward,fp16q8)。修复后 batching 才有重新开启的价值。
2. **conformance 边缘超标**:SDK tensor path 与 Hub raw path 的 embedding 一致性测试在个别 COCO 图片上 `max_abs_diff ≈ 0.00506 > 0.005`(cosine 0.99949 仍通过)。根因是 Go `imaging.Linear` 与 Rust `image` crate bilinear 的 resize 舍入差异,先于本轮改动存在,对检索质量无实际影响。
3. **raw path 解码占用 tokio worker**:semaphore 移除后,raw 解码在 async 线程上同步执行(每张 ~5–10 ms)。当前并发规模无碍;若未来出现大并发 raw 请求,可移到 `spawn_blocking`。

## 复现

```bash
# Hub
cargo run --release --no-default-features --features metal,siglip -p lumen-hub -- \
  --config crates/lumen-hub/examples/bench-siglip.yaml --log-level info

# SDK bench
go run ./cmd/lumen-bench \
  --image-dir "$LUMEN_BENCH_DATA_ROOT/siglip-coco-val2017/sample-500" \
  --task semantic_image_embed --mode both --concurrency 1,4,8,16 \
  --limit 500 --warmup 20 --hub-pid <pid> --out bench-results/<name>
```

batch size 观测:`--log-level debug` 后 grep `batch flushed`。
