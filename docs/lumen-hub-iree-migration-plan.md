# Lumen Hub: Burn → IREE Migration Plan

| | |
|---|---|
| Branch | `migration/iree`, created from `main` @ `cc51e8d`. It is **not** merged into `main` before §10 (Definition of Done) holds in full. |
| Revision | 1 (2026-10-08) |
| Pinned toolchain | IREE **3.12.0** (git tag `v3.12.0`, commit `2b05c5dbb2f2ecb27c0d3941e80ee8d2f16e890d`); onnx 1.23.2; onnxruntime 1.30.0; numpy 2.5.3 |
| Already on the branch | this plan; `tools/iree/` (conversion toolchain + recipes + self-tests); `fixtures/iree/qa-tiny/` (committed test fixture); `docs/iree-migration/reference/` (verified reference implementation of `lumen-iree-sys` and `lumen-iree`) |

## 0. How to use this document

- The plan is executed as §5 orders it. A phase is complete only when every acceptance criterion of that phase passes, and a phase starts only when its prerequisites are complete:

  | Phase | Prerequisites |
  |---|---|
  | 1 Runtime crates | — |
  | 2 Real inputs and artifacts | 1 |
  | 3 Schema, config, download | 1 |
  | 4 IREE engine and QA | 3 |
  | 5 Production models | 4; its L1 acceptance also needs 2 |
  | 6 Remove Burn | 5 |
  | 7 Packaging and release | 6 |
  | 8 Hardware validation and merge | 7 |
- "MUST", "MUST NOT", "SHOULD" are used as in RFC 2119. Every decision in §3 is final for this migration; anything not described here is out of scope and MUST NOT be done on this branch.
- Phases marked **USER** need the maintainer's own machines or credentials (real ONNX files, Hugging Face upload, GPU hardware). Every other phase is executable in a cloud session (Linux x86-64, no GPU).
- §2 records the measurements behind the decisions. Appendix C explains how to reproduce them.
- The capability names follow `AGENTS.md`: 图像语义分析 / Image Semantic Analysis (`siglip`), 人物识别 / Person Recognition (`face`), OCR文字识别 / OCR Text Recognition (`ocr`), BioCLIP物种识别 / BioCLIP Species Recognition (`bioclip`).

## 1. Scope

### 1.1 Goals

1. IREE becomes the only inference runtime of `lumen-hub`. Burn, burn-store, burn-flex, CubeCL, the burn-onnx generated code in `crates/lumen-hub/src/model_arch/`, and `crates/lumen-quant-core` are removed.
2. Two first-party crates provide the IREE binding; no third-party IREE binding crate is used:
   - `crates/lumen-iree-sys`: builds the IREE runtime from the pinned source with CMake and statically links it together with a small C shim (`csrc/lumen_iree.{h,c}`); exposes the shim as raw FFI.
   - `crates/lumen-iree`: safe, inference-only Rust wrapper.
3. `tools/iree` converts the maintainer's fp32 ONNX files into verified, unambiguously named artifact sets and generates the committed QA fixture.
4. Every published model has exactly one precision (§3.2), one quantization scheme, and a fixed set of entry points and targets.

### 1.2 Invariants (MUST hold during and after the migration)

1. **No network-layer change.** Files under `crates/lumen-hub/proto/`, the gRPC services, request/response messages, task names, result payload schemas (`embedding_v1`, `face_v1`, `ocr_v1`, `labels_v1`), capability fields, thin tensor preprocess IDs, control-plane messages and mDNS TXT keys are unchanged. `just contract` stays green. The backend name reported by the status bus, capabilities and mDNS stays one of `cpu`, `cuda`, `metal`.
2. **No new inference protocol.**
3. **Preprocessing is unchanged**, with exactly one exception: the OCR detection resize rule of §3.6.2.
4. **`lumen-schema` changes are additive**: a new enum variant `Runtime::Iree` and new pure functions. The JSON shape of `model_info.json` (`ModelInfo`, `RuntimeSpec`) does not change at all, because already-released hubs parse the same Hugging Face files with `deny_unknown_fields`.
5. Out of scope for this migration: fp16/bf16 compute, dynamic shapes, compiled batch sizes other than 1, Vulkan, ROCm/HIP, WebGPU, CUDA on Windows, Intel GPUs, ONNX opsets above 17, any IREE version other than 3.12.0.

## 2. Findings (evidence for §3)

Measured while preparing this plan with the pinned toolchain on a 4-vCPU Intel Xeon (AVX-512 VNNI) cloud VM. Real weights: antelopev2 (`scrfd_10g_bnkps`, `glintr100`, insightface release `v0.7`), PP-OCRv6 small detection/recognition (bundled in the `rapidocr` 3.10.0 wheel), SigLIP2 B/16-224 vision and text towers (Google `big_vision` checkpoint, re-expressed as ONNX in the same operator style as the Hugging Face/optimum exports: `MatMul` weights `[K, N]`, `LayerNormalization`, tanh-GELU, MAP head, last-token text pooling). BioCLIP-2 and SigLIP2 so400m were not downloadable from the research environment; they use the identical ViT code path and are covered by the SigLIP2 B/16 measurements and by the mandatory verification gates of §3.12.

| # | Finding |
|---|---|
| F1 | **fp32 import is exact.** SCRFD, ArcFace R100, PP-OCRv6 det/rec and SigLIP2 vision/text import, compile and match onnxruntime with cosine 1.000000 (max abs error ≤ 7e-5) on CPU. |
| F2 | **Dynamic H/W does not compile.** PP-OCRv6 detection with dynamic `H, W` fails with three distinct compiler errors (`MaxPool auto_pad=SAME_UPPER` legalization; after rewriting that, `workgroup_count_hint` failures on `Conv`/`ConvTranspose`). Static shapes compile. |
| F3 | **Static buckets must not be zero-padded.** Padding a detection input to a larger bucket changes the probability map by up to 0.99 and changes boxes (the PP-LCNet squeeze-excitation blocks average over the padded area). Each entry must therefore receive an image resized to exactly its shape. |
| F4 | **Several entry points share one weight set.** Importing many static-shape graphs into one MLIR module with deduplicated named parameters works: 2 OCR entries → 141 parameters, the same `.irpa` as one entry. |
| F5 | **IREE fuses int8 compute only for per-tensor weights.** torch-mlir `FuseQuantizedOps` (pinned commit `77139e3`) bails out on per-channel `DequantizeLinear` ("Partially traced quantized operands. This op will remain in QDQ form."). |
| F6 | **Accuracy of quantized variants (real weights, onnxruntime, vs fp32).** SCRFD: W8A32 detection F1 1.000, keypoint error 0.055 px; W8A8 per-tensor F1 1.000, 0.34–0.56 px. ArcFace: **W8A32 cosine mean 0.99968 / min 0.99777**; W8A8 per-tensor mean 0.946–0.973, min 0.775–0.865. OCR det: W8A32 box F1 0.939; W8A8 0.749. OCR rec: W8A32 string match 72.7 % (93.8 % on confident crops), W8A8 15–20 %; the recognizer flips 7 % of strings under 0.1 % relative weight noise but 0 % with fp16-rounded weights. SigLIP2 vision: **W8A32 cosine mean 0.99926 / min 0.99776** (32 real images); W8A8 per-tensor 0.679 / 0.500. SigLIP2 text incl. int8 token-embedding table: **W8A32 min cosine 0.99769** (24 token sequences). |
| F7 | **Latency on IREE 3.12 CPU (4 threads).** SCRFD: fp32 350 ms, W8A32 5698 ms, W8A8 2258 ms. ArcFace: fp32 391 ms, W8A32 4438 ms, W8A8 1532 ms. ViT-B/16: fp32 595 ms (283 ms with data tiling), W8A32 857 ms (471 ms with data tiling), W8A8 219 ms with data tiling. Quantized convolutions are 4–16× slower than fp32; quantized transformer matmuls are fine. |
| F8 | **Memory.** Parameters memory-mapped with `LUMEN_IREE_PARAMS_MMAP` are aliased by the CPU device: ArcFace fp32 resident anonymous memory 17 MB (weights stay reclaimable page cache) vs 273 MB with file reads. W8A32 weights stay int8: ViT-B with data tiling 7 MB anonymous + 92 MB page cache, vs fp32 with data tiling 359 MB anonymous (packed copies). A 256 000 × 768 int8 embedding table stays mapped; only gathered rows are touched (8 MB resident). |
| F9 | **Block-wise int8 is neither supported nor useful.** `DequantizeLinear` with `block_size` (opset 21) fails to legalize in the importer; block sizes 64/32/16 do not improve OCR recognition over per-channel scales. |
| F10 | **CPU data tiling is off by default** in 3.12; `--iree-opt-data-tiling` halves ViT latency, makes SCRFD 8 % faster and ArcFace 13 % slower. |
| F11 | **GPU targets.** CUDA embeds PTX ISA 7.6 that the driver JIT-compiles (`cuModuleLoadDataEx`), so one `sm_75` artifact serves compute capability ≥ 7.5 (incl. Jetson Orin, sm_87); the CUDA driver library is `dlopen`ed (`libcuda.so`/`nvcuda.dll`), the CUDA Toolkit ≥ 12 is needed only to build. Metal embeds MSL source compiled at load time, so it is produced on Linux. Every fp32 model compiles for Metal in seconds. **W8A32 matmuls with more than one row never finish compiling for Metal** (M ≥ 16 times out; M = 1 compiles in 2 s; the hang is in SPIR-V code generation after `EliminateEmptyTensorsPass`); CUDA compiles them in 8 s. The vendor-neutral Vulkan target (`vp_android_baseline_2022`) fails on SCRFD; IREE has no Intel GPU target. |
| F12 | **OCR detection buckets with a 32-pixel side do not compile** (`linalg.generic` shape-inference error in a stride-2 depthwise convolution on a ~1-pixel feature map); 64…960 compile. The 57-entry detection module compiles in ≈10 min for `cpu-x86_64`, `cpu-aarch64` and `cuda-sm_75` (10.6–24.8 MB). For `metal-macos` one entry compiles in 10 s but module compile time grows superlinearly with the entry count (4 → 34 s, 8 → 93 s, 16 → 256 s); the full 57-entry module compiles in 62 min (single-threaded, 3.4 GB peak resident memory) to 32.0 MB, and it carries ≈4.7 k MSL kernels (102 per entry) that the Metal driver compiles when the module is loaded. |
| F13 | **The C shim is correct.** A runtime-only IREE build plus the shim links into one 2.4 MB static archive in 36 s; all paths (both parameter modes, wrong shape, wrong byte length, unknown function, missing file, wrong CPU ISA, 4 threads sharing a model) behave correctly; AddressSanitizer/LeakSanitizer report nothing. |
| F14 | **The toolchain works on real models.** `tools/iree` converted antelopev2 (4 targets) and SigLIP2 B/16 vision+text (+ synthetic aesthetic head) end-to-end, with parity cosine ≥ 0.9999999999 and quality cosine ≥ 0.9977. The real SigLIP2 B/16 `w8a32` vision, text and aesthetic modules compile for `cuda-sm_75` in 2–9 s each. |

## 3. Decisions (normative)

### 3.1 Toolchain pin

- Compiler: Python packages pinned in `tools/iree/requirements.txt` (`iree-base-compiler==3.12.0`, `iree-base-runtime==3.12.0`, `onnx==1.23.2`, `onnxruntime==1.30.0`, `numpy==2.5.3`), Python 3.12.
- Runtime: built by `lumen-iree-sys` from IREE git commit `2b05c5dbb2f2ecb27c0d3941e80ee8d2f16e890d` (tag `v3.12.0`) with submodule `third_party/flatcc` at the commit recorded in that IREE tree.
- `.vmfb` files are only guaranteed to load in the runtime of the same IREE release. Changing the IREE version is a new plan revision that regenerates every artifact and the QA fixture; it is not part of this migration.

### 3.2 Precision tags and per-model precision

Precision tags for `runtime: iree`:

| Tag | Meaning |
|---|---|
| `fp32` | float32 weights, float32 activations. |
| `w8a32` | Weight-only int8: symmetric int8 weights (range −127…127, zero point 0) with one float32 scale per output channel (per row for embedding tables), float32 activations. Rule: §3.3. |

No other tag is valid for `runtime: iree` (in particular not `fp16`, `fp16q8`, `int8`, `w8a8`, `w8a16`). Each production model has exactly one precision:

| Model | Capability | Precision | Quantized components | Not quantized |
|---|---|---|---|---|
| `siglip2-base-patch16-224` | 图像语义分析 | `w8a32` | `vision`, `text` | `aesthetic` |
| `siglip2-so400m-patch14-384` | 图像语义分析 | `w8a32` | `vision`, `text` | `aesthetic` |
| `bioclip-2` | BioCLIP物种识别 | `w8a32` | `vision` | — |
| `antelopev2` | 人物识别 | `fp32` | — | `detection`, `recognition` |
| `pp-ocrv6-small` | OCR文字识别 | `fp32` | — | `detection`, `recognition`, `classification` |
| `qa-tiny` (tests only) | — | `fp32` and `w8a32` | `net` (in `w8a32`) | — |

Rationale: transformers keep ≥ 0.9977 cosine and 4× smaller weights under W8A32 with acceptable latency (F6, F7, F8). Convolutional models are both slower (F7) and less accurate (F6) when quantized in any form, and their fp32 weights are small (17–260 MB, memory-mapped per F8). W8A8 is rejected for every model (F5, F6). The aesthetic head stays fp32 because its final linear layer produces the score directly (the existing Burn policy, `load_aesthetic_head`).

### 3.3 W8A32 rule

Implemented by `tools/iree/lumen_iree_tools/quantize.py` (normative). A float32 initializer with rank ≥ 2 and ≥ 4096 elements is quantized iff **all** its uses are one of: input 1 of `Conv` (axis 0), input 1 of `ConvTranspose` (axis 1), input 1 of a rank-2 `MatMul` weight (axis 1), input 1 of `Gemm` (axis 0 if `transB = 1`, else 1), input 0 of `Gather` with `axis = 0` on a rank-2 table (axis 0); and all uses agree on the axis. Scale `s = max|w| / 127` over the channel (`1.0` for an all-zero channel), `q = round_half_to_even(w / s)` clipped to ±127. The initializer `W` is replaced by `W__w8q` (int8), `W__w8s` (float32 scales), `W__w8z` (int8 zeros) and one `DequantizeLinear(axis)` node producing `W__w8dq`. Components with `"quantize": false` in their recipe keep all weights in float32.

### 3.4 Targets and drivers

| Target | `iree-compile` flags | Runtime driver | Published for |
|---|---|---|---|
| `cpu-x86_64` | `--iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=x86_64-unknown-unknown-eabi-elf --iree-llvmcpu-target-cpu=x86-64-v3` | `local-task` | `fp32`, `w8a32` |
| `cpu-aarch64` | `… --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-llvmcpu-target-cpu=generic` | `local-task` | `fp32`, `w8a32` |
| `cuda-sm_75` | `--iree-hal-target-device=cuda --iree-cuda-target=sm_75` | `cuda` | `fp32`, `w8a32` |
| `metal-macos` | `--iree-hal-target-device=metal --iree-metal-target-platform=macos` | `metal` | `fp32` only (F11) |

- CPU targets add `--iree-opt-data-tiling` exactly when the recipe sets `"cpu_data_tiling": true` (all `w8a32` recipes); GPU targets never get it.
- CPU executables are embedded ELF and therefore OS-independent: `cpu-x86_64` serves Linux and Windows x86-64 (x86-64-v3 = Haswell or newer), `cpu-aarch64` serves Linux and macOS arm64. A module for the wrong ISA is rejected at load with `INCOMPATIBLE` (F13).
- **Placement rule in lumen-hub** (the only place a target is chosen):
  1. Build without GPU feature → every model on `local-task` with the host CPU target.
  2. Build with feature `cuda` → every model on `cuda` with `cuda-sm_75`.
  3. Build with feature `metal` → `fp32` models on `metal` with `metal-macos`; `w8a32` models on `local-task` with `cpu-aarch64`.
- A GPU build whose GPU device cannot be created fails startup with an error naming the driver (status phase `FAILED`, health `NOT_SERVING`); it never falls back silently. The launcher keeps choosing the profile from detected hardware.
- Not supported in this migration: Vulkan (per-vendor artifacts would be required and are untestable; Intel has no target), ROCm/HIP, CUDA on Windows, compute capability < 7.5.

### 3.5 Artifacts, naming and publishing

Per model repository `Lumilio-Photos/<model>` on Hugging Face (and the identical local cache layout `<cache_dir>/<model>/`):

```
model_info.json                                  # runtimes.iree added (shape unchanged)
iree/<component>.<precision>.irpa                # weights, shared by all targets
iree/<component>.<precision>.<target>.vmfb       # one per published target (§3.4)
iree/BUILD.<precision>.json                      # provenance + verification report (not downloaded)
burn/...                                         # kept untouched for released hubs
```

`model_info.json` gains exactly:

```json
"runtimes": {
  "burn": { "...": "unchanged" },
  "iree": { "available": true, "components": ["<component>", "..."], "precisions": ["<precision>"] }
}
```

Publishing is all-or-nothing: an artifact set is uploaded only when `tools/iree` produced every target of §3.4 for its precision and its verification passed (the tool refuses to write `model_info.json` otherwise). The `burn/` directory and the `runtimes.burn` entry stay in the repositories so already-released hubs keep working.

### 3.6 Entry points

Every component module exports functions named below; all have batch size 1 and the input/outputs listed (dtype float32 unless noted). Source output indices refer to the maintainer's ONNX files as used to generate the current Burn code; `tools/iree` checks every shape.

| Model / component | Function(s) | Input | Outputs (as returned) |
|---|---|---|---|
| siglip2-base / `vision` | `main` | `[1,3,224,224]` | `[1,768]` (source output 0) |
| siglip2-base / `text` | `main` | int64 `[1,64]` | `[1,768]` (source output 0) |
| siglip2-base / `aesthetic` | `main` | `[1,768]` | `[1]` |
| siglip2-so400m / `vision` | `main` | `[1,3,384,384]` | `[1,1152]` (source output 1, pooled) |
| siglip2-so400m / `text` | `main` | int64 `[1,64]` | `[1,1152]` (source output 1, pooled) |
| siglip2-so400m / `aesthetic` | `main` | `[1,1152]` | `[1]` |
| bioclip-2 / `vision` | `main` | `[1,3,224,224]` | `[1,768]` |
| antelopev2 / `detection` | `main` | `[1,3,640,640]` | 9 outputs `[12800,1] [3200,1] [800,1] [12800,4] [3200,4] [800,4] [12800,10] [3200,10] [800,10]` |
| antelopev2 / `recognition` | `main` | `[1,3,112,112]` | `[1,512]` |
| pp-ocrv6-small / `detection` | `main_h{H}_w{W}` (57 functions, §3.6.2) | `[1,3,H,W]` | `[1,1,H,W]` |
| pp-ocrv6-small / `recognition` | `main` | `[1,3,48,320]` | `[1,40,18710]` |
| pp-ocrv6-small / `classification` | `main` | `[1,3,80,160]` | `[1,2]` |
| qa-tiny / `net` | `main` | `[1,3072]` | `[1,16]` (L2-normalized) |

#### 3.6.1 Batching

Compiled entry points have batch size 1. Model wrappers that receive `batch > 1` (dynamic batcher, tensor path) invoke the entry once per item in order and concatenate the results. The batcher, `tensor_batching_supported` reporting and all gRPC semantics are unchanged.

#### 3.6.2 OCR detection input shapes (the only preprocessing change)

`det_preprocess` in `crates/lumen-hub/src/models/ppocr/task.rs` changes from "scale only when the long side exceeds `limit_side_len`" to "always scale so the long side equals `limit_side_len`", and the short-side floor changes from 32 to 64. In the existing f32 arithmetic:

```rust
let ratio = limit_side_len as f32 / h.max(w);                    // h, w: f32; always applied
let resize_h = (((h * ratio) as u32).div_ceil(32) * 32).max(64);
let resize_w = (((w * ratio) as u32).div_ceil(32) * 32).max(64);
```

with `limit_side_len = 960`. This yields exactly the 57 shapes `{960} × {64, 96, …, 960}` and `{64, 96, …, 928} × {960}`; the entry is `main_h{resize_h}_w{resize_w}`. The image is resized to exactly that shape with the existing `FilterType::CatmullRom` call (never padded, F3). `ratio_h = resize_h / h` and `ratio_w = resize_w / w` keep their definitions, so box mapping is unchanged. Compared with today, the input tensor is identical whenever the long side is ≥ 960 and `(short * ratio) as u32 > 32` (aspect ratio below 960/33 ≈ 29.1:1). Images with a long side < 960 are now upscaled (RapidOCR's default behaviour); extremely elongated images get a 64-pixel short side instead of 32 (F12).

### 3.7 Runtime policies in lumen-hub

- One `lumen_iree::Runtime` per driver per process, created lazily by the placement rule (a Metal build may hold a `metal` and a `local-task` runtime).
- Every component is one `lumen_iree::Model` loaded with `ParamsMode::Mmap`.
- All inference still runs through `inference_worker::run` on the single worker thread (unchanged). IREE's `local-task` driver uses its own worker threads for intra-op parallelism.
- After the startup warmup completes, `Model::trim()` is called once on every loaded model. There is no per-job cleanup; `backend::cleanup_memory`, `LUMEN_SKIP_INFER_CLEANUP`, `LUMEN_GPU_MAX_STREAMS` and `LUMEN_GPU_MEMORY_STRATEGY` are removed.
- `RUNTIME_STACK_SIZE` (main.rs) and `INFERENCE_WORKER_STACK_SIZE` (inference_worker.rs) become 16 MiB (the 256 MiB values existed for Burn's monolithic generated `forward` frames).

### 3.8 Build of the runtime

- `cargo xtask iree-fetch` clones IREE at the pinned commit into `<workspace>/third_party/iree` (git-ignored), initializes only `third_party/flatcc`, verifies `git rev-parse HEAD` equals the pinned commit and is idempotent. Network is needed once.
- `lumen-iree-sys/build.rs` builds `csrc/` with CMake (≥ 3.21) and Ninja; the source directory is `LUMEN_IREE_SOURCE_DIR` or `<workspace>/third_party/iree`. A missing checkout fails the build with the message "Run `cargo xtask iree-fetch`". Build machines need CMake, Ninja and a C/C++ compiler (gcc/clang on Linux, Xcode clang on macOS, MSVC on Windows); feature `driver-cuda` additionally needs the CUDA Toolkit ≥ 12 (headers only).
- `local-task` is always compiled in; features `driver-cuda` and `driver-metal` add the respective HAL driver.

### 3.9 Cargo features, release profiles, launcher, Docker

`crates/lumen-hub/Cargo.toml` features become:

```toml
default = ["cpu", "clip", "insightface", "ppocr", "siglip"]
cpu = []                                    # marker; the CPU driver is always present
cuda = ["lumen-iree/driver-cuda"]
metal = ["lumen-iree/driver-metal"]
clip = []
qa = []
insightface = []
ppocr = ["imageproc", "rten-imageproc"]
siglip = []
```

Features `wgpu`, `vulkan` and `rocm` are deleted. Enabling both `cuda` and `metal` is a `compile_error!`. `BACKEND_NAME` is `"cuda"`, `"metal"` or `"cpu"`.

Release profiles (`crates/xtask/src/main.rs` `PROFILES`, `.github/workflows/release.yml`, `lumen-schema` manifest data, `lumen-launcher` `backend_choices`):

| Profile | Target triple | Features | Status after migration |
|---|---|---|---|
| `darwin-arm64-metal` | aarch64-apple-darwin | `metal` + models | kept |
| `darwin-arm64-cpu` | aarch64-apple-darwin | `cpu` + models | kept |
| `windows-x64-cpu` | x86_64-pc-windows-msvc | `cpu` + models | kept |
| `linux-x64-cpu` | x86_64-unknown-linux-gnu | `cpu` + models | kept |
| `linux-x64-cuda` | x86_64-unknown-linux-gnu | `cuda` + models | kept |
| `linux-arm64-cpu` | aarch64-unknown-linux-gnu | `cpu` + models | kept |
| `linux-arm64-jetson` | aarch64-unknown-linux-gnu | `cuda` + models | kept (source-build recipe, not released) |
| `windows-x64-gpu`, `linux-x64-gpu`, `linux-arm64-gpu`, `linux-x64-rocm` | — | — | **removed** |

Launcher choices: `darwin-arm64` → metal, cpu; `windows-x64` → cpu; `linux-x64` → cuda (when NVIDIA is detected), cpu; `linux-arm64` → cpu. Docker image tags: `cpu` and `cuda`; the `vulkan` tag is removed.

### 3.10 Configuration and schema compatibility

- `lumen_schema::Runtime` gains `Iree` (serde `"iree"`). `Burn` stays as a deprecated variant so existing `config.yaml` files keep parsing.
- New pure function in `crates/lumen-schema/src/preset.rs`: `pub fn iree_precision(model: &str) -> Option<&'static str>`. It returns the precision of §3.2 for every production model **whose service has been ported** (it starts empty in Phase 3, Phase 5 adds one model per port, and after Phase 5 it is exactly the five rows of §3.2). It is the single switch between the two runtimes during the transition.
- Rendering (`crates/lumen-schema/src/config/render.rs`): a model with `iree_precision(model) = Some(p)` is rendered as `runtime: iree`, `precision: p`; otherwise as today (`runtime: burn`, `precision: fp16q8`). The constant `MODEL_PRECISION` is removed in Phase 6, when every production model renders as `iree`.
- Hub startup (in `main.rs`, before downloads): a model config with `runtime: burn` and `iree_precision(model) = Some(p)` is rewritten in memory to `runtime: iree`, `precision: p`, and one warning is logged naming the model. A config with `runtime: iree` and `iree_precision(model) = Some(p)` MUST have `precision` absent or equal to `p`, else startup fails with a configuration error. For models without a table entry (`qa-tiny`) the precision must be listed in `model_info.json` `runtimes.iree.precisions` (already enforced by the downloader). After Phase 6 a config with `runtime: burn` for a model without a table entry is a configuration error.
- `fixtures/config/*.yaml`, `crates/lumen-hub/examples/*.yaml`, `packaging/docker/*.yaml` and the exported JSON schemas under `schemas/` are regenerated with the existing commands whenever the rendered output changes.
- `model_download.rs`: for `Runtime::Iree` the planned files per component are `iree/<component>.<precision>.irpa` and `iree/<component>.<precision>.<target>.vmfb`, `<target>` given by `Engine::placement(precision)` (§3.4); `Runtime::Burn` keeps today's behavior until Phase 6 removes it from the hub.
- Capabilities and status keep reporting the build-level `BACKEND_NAME` (also for `w8a32` models placed on the CPU in a Metal build), exactly as today.

### 3.11 Fixtures

- `fixtures/iree/qa-tiny/` (committed, ≈350 KB): `model_info.json`, `expected.json`, `iree/net.fp32.{irpa,cpu-x86_64.vmfb,cpu-aarch64.vmfb,metal-macos.vmfb}`, `iree/net.w8a32.{irpa,cpu-x86_64.vmfb,cpu-aarch64.vmfb}`, `iree/BUILD.{fp32,w8a32}.json`. It is regenerated only with `python -m lumen_iree_tools qa-fixture --out fixtures/iree` (deterministic weights, identical formula to today's Burn QA model). `tools/iree` output is byte-reproducible: `iree-compile` runs from the work directory on the bare `.mlir` file name, so no build-machine path is embedded and a regeneration with unchanged inputs produces identical files (self-test). This replaces the "no binary fixtures" rule of `models/qa/fixture.rs`: IREE artifacts need the compiler, and tests must not depend on Python.
- `write_model_fixture(model_dir)` copies `fixtures/iree/qa-tiny/{model_info.json,iree/*}` into `model_dir` (skipping `BUILD.*.json`).

### 3.12 Verification gates

`tools/iree convert` aborts unless, on the host CPU target, for every component and every entry point:
- **parity**: IREE vs onnxruntime on the exact imported graph, cosine ≥ 0.9999 on every output for two deterministic synthetic inputs;
- **quality** (quantized components only, mandatory): IREE `w8a32` vs onnxruntime fp32 on the real inputs passed with `--inputs`, cosine ≥ the recipe's `min_quantized_cosine` (0.995) for every sample.

Hub-level gates are in §6.

## 4. Components

### 4.1 `crates/lumen-iree-sys`

Created by copying `docs/iree-migration/reference/lumen-iree-sys/` verbatim (Phase 1), then adding it to the workspace. Contents:

| File | Role |
|---|---|
| `Cargo.toml` | `links = "lumen_iree"`, features `driver-cuda`, `driver-metal`, `[package.metadata.dist] dist = false` |
| `build.rs` | locate IREE source, `cmake -G Ninja -S csrc -B $OUT_DIR/cmake -DCMAKE_BUILD_TYPE=Release -DIREE_SOURCE_DIR=… -DLUMEN_IREE_DRIVERS=…` (on MSVC also `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` when the `crt-static` target feature is on, else `MultiThreadedDLL`), `cmake --build … --target lumen_iree`, link `static=lumen_iree` plus the system libraries listed by CMake (`lumen_iree_link_libs.txt`) and: Linux `dl pthread m rt`; macOS with Metal the frameworks `Foundation Metal CoreGraphics` and `objc` |
| `csrc/CMakeLists.txt` | runtime-only IREE (`IREE_BUILD_COMPILER=OFF`, tests/samples/benchmarks/TFLite bindings OFF, `IREE_HAL_DRIVER_DEFAULTS=OFF`, only the requested drivers ON, executable loader and plugin = embedded ELF only) and one static archive with the shim, every transitive IREE object of `iree::runtime::impl`, `iree::io::formats::parser_registry`, `iree::io::parameter_index_provider`, `iree::modules::io::parameters`, and `flatcc_parsing` |
| `csrc/lumen_iree.h` / `.c` | the C ABI (Appendix A) |
| `src/lib.rs` | `#[repr(C)]` mirrors of the four opaque handles and `lumen_iree_tensor_t`, the constants, and the 15 `unsafe extern "C"` declarations |

### 4.2 `crates/lumen-iree`

Created by copying `docs/iree-migration/reference/lumen-iree/` verbatim. Public API (complete):

```rust
pub const PARAM_SCOPE: &str = "model";
pub const MAX_RANK: usize = 8;
pub struct Error;                        // Display + std::error::Error; fn message(&self) -> &str
pub type Result<T> = std::result::Result<T, Error>;
pub enum Driver { LocalTask, Cuda, Metal }   // fn as_str(self) -> &'static str
pub struct Runtime;                      // Send + Sync
impl Runtime { pub fn new(driver: Driver) -> Result<Arc<Runtime>>; pub fn driver(&self) -> Driver }
pub enum ParamsMode { Mmap, Read }
pub struct Model;                        // Send + Sync; keeps its Runtime alive; serializes invocations
impl Model {
    pub fn load(runtime: &Arc<Runtime>, vmfb: &Path, irpa: &Path, params: ParamsMode) -> Result<Model>;
    pub fn has_function(&self, function: &str) -> bool;
    pub fn invoke(&self, function: &str, inputs: &[TensorRef<'_>]) -> Result<Vec<Tensor>>;
    pub fn trim(&self) -> Result<()>;
}
pub enum TensorRef<'a> { F32 { dims: &'a [i64], data: &'a [f32] }, I64 { .. }, I32 { .. } }
pub enum TensorData { F32(Vec<f32>), I64(Vec<i64>), I32(Vec<i32>) }
pub struct Tensor { pub dims: Vec<i64>, pub data: TensorData }  // fn into_f32(self) -> Result<Vec<f32>>
```

Safety invariants (documented at each `unsafe` block): every handle is non-null and released exactly once; a `Model` holds `Arc<Runtime>` so the runtime outlives all models; outputs are released by RAII after copying; the shim validates shapes and byte lengths again; `TensorRef` validation rejects rank > 8, negative or overflowing dims and element-count mismatches before any FFI call.

### 4.3 `cargo xtask iree-fetch`

New subcommand in `crates/xtask/src/main.rs`. Algorithm: if `third_party/iree/.git` exists and `HEAD` is the pinned commit and `third_party/flatcc/include/flatcc/flatcc_verifier.h` exists → print "up to date" and exit 0. Otherwise: `git init third_party/iree`; `git -C third_party/iree fetch --depth 1 https://github.com/iree-org/iree 2b05c5dbb2f2ecb27c0d3941e80ee8d2f16e890d`; `git -C third_party/iree checkout --detach FETCH_HEAD`; `git -C third_party/iree submodule update --init --depth 1 third_party/flatcc`; verify `HEAD`. `.gitignore` gains `/third_party/`. The pinned commit is a `const` in xtask and MUST equal `IREE_GIT_COMMIT` in `tools/iree/lumen_iree_tools/constants.py` (a unit test in xtask reads that file and compares).

### 4.4 lumen-hub runtime layer

Final state (after Phase 6):

| Today | After |
|---|---|
| `src/backend.rs` (Burn backend selection, CubeCL/wgpu setup) | deleted; replaced by `src/runtime.rs` |
| `crate::backend::{Backend, Device, BACKEND_NAME, configure_runtime, default_device, cleanup_memory}` | `crate::runtime::{Engine, BACKEND_NAME}` |
| `Arc<Device>` passed by `build_service_hub_from_config` to every `*Service::from_config` | `Arc<Engine>` |
| `ModelFactory::create(&self, config, device: Arc<Device>)` (`service/factory.rs`, used only by its own tests) | `create(&self, config, engine: Arc<Engine>)` |
| `model_arch::load_burnpack`, `load_aesthetic_head`, `conv_fwd` | deleted |
| `models/*/model.rs` Burn wrappers | IREE wrappers (§4.5) |

`src/runtime.rs` (complete contract):

```rust
pub const BACKEND_NAME: &str;            // final: "cuda" | "metal" | "cpu" by feature
pub struct Engine { /* Mutex<HashMap<Driver, Arc<lumen_iree::Runtime>>>, created lazily */ }
impl Engine {
    pub fn new() -> Engine;               // creates no device yet
    /// Placement rule of §3.4; precision is "fp32" or "w8a32".
    pub fn placement(precision: &str) -> Result<(lumen_iree::Driver, &'static str /* target */), String>;
    /// Loads <model_dir>/iree/<component>.<precision>.irpa and
    /// <model_dir>/iree/<component>.<precision>.<target>.vmfb with ParamsMode::Mmap.
    pub fn load(&self, model_dir: &Path, component: &str, precision: &str) -> Result<lumen_iree::Model, String>;
}
```

`main.rs` drops `configure_runtime()`; `inference_worker.rs` drops `cleanup_memory`; `warmup.rs` calls a new `InferenceService::trim_memory(&self)` (default no-op; ported services call `Model::trim()` on each of their models) once after the mandatory warmup.

Transition (Phases 4–5): `backend.rs` and `runtime.rs` coexist. `build_service_hub_from_config` creates both `Arc<Device>` (Burn) and `Arc<Engine>` (IREE) and passes `Engine` to services already ported, `Device` to the others. `runtime::BACKEND_NAME` re-exports `backend::BACKEND_NAME` until Phase 6. Feature mapping during the transition: `cpu` → Burn Flex; `metal` → Burn Metal + `lumen-iree/driver-metal`; `cuda` → Burn CUDA + `lumen-iree/driver-cuda`; `wgpu`/`vulkan`/`rocm` → Burn only (ported services run on the IREE CPU driver). Phase 6 applies the final feature table of §3.9.

### 4.5 Model wrappers

Each wrapper keeps the existing public methods used by its `task.rs` so task code (pre/post-processing, gRPC handling) is untouched apart from the OCR resize rule:

| File | New implementation |
|---|---|
| `models/siglip/model.rs` | `SiglipTextModel::encode(&[i64])` → `main` with `I64 [1,64]`, returns `[D]`; `SiglipVisionModel::encode(pixels, batch, h, w)` → loop `main` with `[1,3,S,S]`; when the aesthetic head is loaded, `head.main([1,D])` per row. `S`/`D`: base 224/768, so400m 384/1152. |
| `models/bioclip/model.rs` | `BioClipVisionModel::encode` → loop `main` `[1,3,224,224]` → `[768]`. |
| `models/insightface/model.rs` | detection `main [1,3,640,640]` → 9 `TensorOutput`s in source order; recognition `main [1,3,112,112]` → `[512]`. |
| `models/ppocr/model.rs` | detection `main_h{H}_w{W}`; recognition `main [1,3,48,320]`; classification `main [1,3,80,160]`. Supported OCR model: `pp-ocrv6-small` only. |
| `models/qa/model.rs` | `QaNet` → loop `main [1,3072]` → `[16]`. |

Every wrapper checks the output count and dims against the table of §3.6 and returns `ServiceError::Internal` on mismatch.

### 4.6 `lumen-hub dump-inputs`

New subcommand (in `crates/lumen-hub/src/main.rs` next to the `config` render command) producing the real inputs required by `tools/iree convert --inputs` for quantized components:

```
lumen-hub dump-inputs --model <siglip2-base-patch16-224|siglip2-so400m-patch14-384|bioclip-2> \
    --cache-dir <dir> --images <dir> [--texts <file>] --out <dir>
```

It runs only the existing preprocessing (no inference): for every `*.jpg|*.jpeg|*.png|*.webp` in `--images` it writes `<out>/vision/<n>.npy` (float32 `[1,3,S,S]`, the exact tensor today's task feeds to the vision encoder); for every non-empty line of `--texts` (SigLIP only) it writes `<out>/text/<n>.npy` (int64 `[1,64]`, the exact token ids the text task produces). `.npy` is written as format version 1.0, little-endian, C order. Inputs are then packed with `python -m lumen_iree_tools pack-inputs --out vision.npz <out>/vision/*.npy`. Minimum: 32 images and 24 texts.

### 4.7 `tools/iree`

Already on the branch. Usage (Python 3.12):

```bash
cd tools/iree
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -m unittest discover -s tests            # self-tests
python -m lumen_iree_tools check-recipes        # validate recipes/*.json
python -m lumen_iree_tools convert --recipe recipes/<model>.json \
    --onnx <component>=<fp32.onnx> [...]        # one per recipe component
    [--inputs <component>=<npz> ...]            # required for quantized components
    --model-info <current model_info.json> --out <out-dir> --targets all
python -m lumen_iree_tools reference --recipe recipes/<model>.json --component <c> \
    --onnx <fp32.onnx> --inputs <npz> --out <file.json>  # fp32 references for L1
python -m lumen_iree_tools qa-fixture --out ../../fixtures/iree
```

`convert` normalizes every entry (one runtime input of the recipe dtype, static shape, kept outputs, opset 17 with IR version 8, non-negative `Concat` axes, onnxruntime basic optimizations, standard-domain operators only, recipe output shapes), applies §3.3 for quantized components, imports all entries of a component into one module with shared parameters, compiles every target, verifies (§3.12) and writes `<out>/<model>/{model_info.json,iree/…}`. ONNX files above 2 GiB must use external data next to the `.onnx` file. Recipes (`tools/iree/recipes/*.json`) are the single source of truth for §3.2–§3.6; `model_info_checks` fail the conversion when `model_info.json` disagrees with them (OCR shapes).

## 5. Phases

### Phase 1 — Runtime crates (cloud)

1. `cargo xtask iree-fetch` (§4.3) and `/third_party/` in `.gitignore`.
2. Copy the reference crates to `crates/lumen-iree-sys` and `crates/lumen-iree` (§4.1, §4.2); they become workspace members through the existing `crates/*` glob. `lumen-hub` does not depend on them yet.
3. `justfile`: recipe `iree-fetch` (`cargo xtask iree-fetch`); recipes `test`, `l0`, `l0-backend`, `check-backend`, `l1-backend` depend on it.
4. CI (`.github/workflows/ci.yml`): every job that builds the workspace installs Ninja (`sudo apt-get install -y ninja-build`, `brew install ninja`, `choco install ninja`) and runs `cargo xtask iree-fetch` before building; `actions/cache` caches `third_party/iree` keyed on the pinned commit.
5. New CI job `iree-runtime` with the matrix `ubuntu-22.04` (x86-64), `ubuntu-22.04-arm` (aarch64), `macos-15` (arm64), `windows-2025` (x86-64, MSVC environment from `ilammy/msvc-dev-cmd@v1`, `RUSTFLAGS=-C target-feature=+crt-static` to match the release profiles): `cargo test -p lumen-iree --release` (runs the `cpu-x86_64` or `cpu-aarch64` fixture modules natively). On `macos-15` additionally `cargo test -p lumen-iree --release --features driver-metal` with a new test (cfg feature `driver-metal`) that runs `net.fp32.metal-macos.vmfb` on `Driver::Metal` against `expected.json` (cosine > 0.99999).
6. The release workflow's `linux-x64-cuda` job (CUDA Toolkit installed) gains a step `cargo build -p lumen-iree-sys --release --features driver-cuda`.

Acceptance: the `iree-runtime` job is green on all four runners; clippy (`-D warnings`) and rustfmt are clean; the `driver-cuda` build step succeeds; `xtask` unit test comparing its pinned commit with `tools/iree/lumen_iree_tools/constants.py` passes.

### Phase 2 — Real inputs and artifact production (**USER**)

1. (cloud) Implement `lumen-hub dump-inputs` (§4.6) in the current code base — preprocessing does not depend on the runtime. Unit test: dumping `warmup/semantic/bus.jpg` yields exactly the tensor the SigLIP task passes to its encoder, and dumping a text yields exactly the token ids of the text task.
2. (USER, on a machine with the fp32 ONNX files) Create the venv (§4.7). For each production model run `convert` with every recipe component, `--inputs` for every quantized component (≥ 32 own photos; ≥ 24 search phrases for SigLIP, both produced by `dump-inputs` + `pack-inputs`), `--model-info` pointing at the model's current `model_info.json`, and `--targets all`. The machine needs ≥ 8 GB RAM; `pp-ocrv6-small` takes ≈ 1.5 h on 4 cores (the `metal-macos` detection module alone ≈ 1 h, F12), every other model takes minutes.
3. (USER) Produce the fp32 references for the quantized components used by the L1 suites and commit them under `crates/lumen-hub/tests/golden/`:

   | Model | Component | L1 input | Reference file |
   |---|---|---|---|
   | siglip2-base-patch16-224 / siglip2-so400m-patch14-384 | `vision` | `crates/lumen-hub/warmup/semantic/bus.jpg` | `<model>.vision.bus.fp32-reference.json` |
   | same | `text` | `a photo of a bus` | `<model>.text.bus.fp32-reference.json` |
   | same | `text` | `a photo of a kitten` | `<model>.text.kitten.fp32-reference.json` |
   | bioclip-2 | `vision` | `crates/lumen-hub/warmup/bio/abyssinian.jpg` | `bioclip-2.vision.abyssinian.fp32-reference.json` |

   Each file is produced by `dump-inputs` (one image or one text), `pack-inputs`, and `python -m lumen_iree_tools reference --recipe … --component … --onnx <fp32.onnx> --inputs … --out …`.
4. (USER) Inspect `iree/BUILD.<precision>.json` (every gate passed) and upload `iree/` plus the updated `model_info.json` to `Lumilio-Photos/<model>`; never delete `burn/`.

Acceptance: all five repositories contain the complete file set of §3.5 for their precision; `ModelInfo::from_json_str` of `lumen-schema` at `main` parses each updated `model_info.json`.

### Phase 3 — Schema, config and download (cloud)

Implement §3.10 with an empty `iree_precision` table: `Runtime::Iree`, `iree_precision`, render rule, startup rewrite/validation, `model_download.rs` planning for `Runtime::Iree`. `lumen-hub` gains the dependency `lumen-iree = { path = "../lumen-iree" }` and a `src/runtime.rs` containing `Engine::placement` (§3.4, §4.4); the rest of `Engine` follows in Phase 4.

Acceptance: `just ci` green (nothing renders or runs as `iree` yet); new unit tests: with a test-only table entry the renderer emits `runtime: iree` and the §3.2 precision; `runtime: burn` configs for table models are rewritten; a mismatched `iree` precision is rejected; the download plan for each placement (cpu/cuda/metal × fp32/w8a32) lists exactly the `.irpa` and one `.vmfb` per component.

### Phase 4 — IREE engine and QA model (cloud)

1. `src/runtime.rs` (§4.4, transition form); `Engine` created in `build_service_hub_from_config` next to the Burn device; transition feature mapping (§4.4).
2. Port `models/qa` to `Engine` (§4.5) and `models/qa/fixture.rs` (§3.11); `tests/common/harness.rs` renders `"runtime": "iree"`; `fixture_roundtrips_through_all_precisions` covers `fp32` and `w8a32` (cosine to `expected.json` > 0.99999 each; `w8a32` vs the fp32 reference > 0.999).
3. `warmup.rs` trims (§3.7) for ported services; the Burn `cleanup_memory` stays for Burn services until Phase 6.

Acceptance: `just ci` (incl. `just l0`) green on Linux; L0 on macOS with `--features metal` green; `just check-backend cpu|metal|wgpu` compiles.

### Phase 5 — Production models (cloud for code; L1 in CI/USER)

Port, in this order, one model family per commit series, each leaving the branch green: InsightFace (人物识别, `antelopev2`), PP-OCR (OCR文字识别, `pp-ocrv6-small`, including §3.6.2), SigLIP (图像语义分析, both models, vision/text/aesthetic), BioCLIP (BioCLIP物种识别, `bioclip-2`). For each:

1. Wrapper per §4.5; the service's `from_config` takes `Arc<Engine>`; `main.rs` passes the engine.
2. Add the model to `iree_precision` (§3.10); regenerate config fixtures, examples, Docker configs, JSON schemas.
3. `tests/common/mod.rs` `require_model_precision` looks for `iree/<component>.<precision>.irpa` and the placement `.vmfb`; `l1_models.rs`/`l1_parity.rs` use `runtime: iree` and the §3.2 precision.
4. `l1_parity.rs`: `w8a32` components are compared with the committed `*.fp32-reference.json` files (cosine ≥ 0.995 per sample); `fp32` components need no reference file (their parity is covered by the conversion gate).
5. OCR only, before porting: extend `check_ocr` in `tests/l1_models.rs` so that with `LUMEN_GOLDEN_WRITE=1` it writes the `ocr_v1` payload to `tests/golden/<model>.<image-stem>.ocr.json`, add a `pp-ocrv6-small` case for `tests/test_sample/ocr_test_1.jpeg` next to the existing `warmup/ocr/border.png` case, and (USER, where the Burn weights are available) run `LUMEN_GOLDEN_WRITE=1 LUMEN_MODELS_DIR=… cargo test -p lumen-hub --release --test l1_models ppocr -- --test-threads=1` on the still-Burn code and commit the two files. After porting, `check_ocr` asserts that at least 90 % of the recorded boxes are matched by an IREE box with IoU ≥ 0.5 and that matched boxes have equal text. The `pp-ocrv5` and `pp-ocrv5-server` L1 cases are deleted (those models are not in the catalog and are not converted).
6. `.github/workflows/nightly-models.yml` fetches, for the ported model, `model_info.json`, tokenizer/dictionary files as today, `iree/*.irpa`, `iree/*.cpu-aarch64.vmfb` and (for `fp32` models) `iree/*.metal-macos.vmfb`.

Acceptance per model: unit tests and L0 green in CI; `nightly-models` run via `workflow_dispatch` on `migration/iree` green for `cpu` and `metal`; golden files regenerated with `just golden` and the regeneration report (cosine of each new golden vs the previous Burn golden ≥ 0.99) pasted into the commit message. L1 needs the model artifacts from Hugging Face; a session without Hugging Face access runs everything else and MUST report L1 as not run.

### Phase 6 — Remove Burn (cloud)

1. Delete `src/model_arch/`, `src/backend.rs`, `crates/lumen-quant-core/`, `examples/{quantize_check.rs,repro_quant_roundtrip.rs,siglip_smoke.rs}`, `crates/lumen-hub/tools/check_onnx_candle_ops.py`, and the burn/burn-flex/burn-store/cubecl dependencies; drop the Burn `Device` from `build_service_hub_from_config`; `ModelFactory` takes `Arc<Engine>`.
2. Final feature table and `BACKEND_NAME` of §3.9; `runtime.rs` no longer re-exports from `backend.rs`.
3. `render.rs` loses `MODEL_PRECISION`; `model_download.rs` loses the Burn branch; startup rejects `runtime: burn` for models without a table entry (§3.10).
4. Stack sizes and cleanup env vars of §3.7.
5. Mark `docs/lumen-hub-runtime-memory-decision.md` and `docs/lumen-hub-tensor-batching-decision.md` as historical (Burn) with one line at the top.

Acceptance: `cargo tree -i burn` and `cargo tree -i cubecl-runtime` report that the package is not in the graph; `just ci` green; nightly L1 green.

### Phase 7 — Packaging, release, docs (cloud)

Implement §3.9 in `crates/xtask/src/main.rs`, `.github/workflows/release.yml` (install Ninja; run `cargo xtask iree-fetch`; the CUDA toolkit step stays), `crates/lumen-launcher/src/setup.rs`, `crates/lumen-schema/src/manifest.rs`, `packaging/docker/Dockerfile` and its README (tags `cpu`, `cuda`); update `README.md` (runtime IREE, features, build prerequisites CMake ≥ 3.21 + Ninja + `cargo xtask iree-fetch`) and `crates/lumen-hub/tools/*/model_info.example.json` (`runtimes.iree`).

Acceptance: `cargo xtask dist --profile linux-x64-cpu` builds, and the binary started with the rendered `minimal` preset downloads the Phase 2 artifacts and reaches `PHASE_READY` (run where Hugging Face is reachable); `just ci` and `just contract` green; `git diff main -- crates/lumen-hub/proto` is empty.

### Phase 8 — Hardware validation (**USER**) and merge

On M2 Pro (`darwin-arm64-metal` and `darwin-arm64-cpu`), Jetson Orin Nano (`linux-arm64-jetson`), Intel N100 (`linux-x64-cpu`):
- `just l1-backend <cpu|metal|cuda>` passes;
- cross-device consistency: SigLIP image embedding of `warmup/semantic/bus.jpg` and ArcFace embedding of `warmup/face/face.jpg` agree with the cloud Linux CPU result at cosine ≥ 0.999;
- resident memory after warmup for the `basic` preset does not exceed the Burn numbers in `docs/lumen-hub-tensor-batching-decision.md` ("Preset 内存画像") by more than 10 %;
- record `semantic_image_embed`, `ocr`, `face_recognition` p50/p95 with the SDK bench (same command as the batching decision doc) in `docs/lumen-hub-iree-runtime-decision.md` (new);
- Metal load cost (F12): on the M2 Pro, with artifacts already downloaded and after a reboot, measure the time from process start to `PHASE_READY` for the `basic` preset with `darwin-arm64-metal` and with `darwin-arm64-cpu`. If the Metal build is slower by more than 60 s, the `metal-macos` target gains `--iree-metal-compile-to-metallib` (precompiled metallib instead of MSL source; `metal-macos` artifacts and the QA fixture's Metal module are then produced on macOS with the Xcode Command Line Tools), every `metal-macos` artifact is regenerated and re-uploaded, §3.4 is updated in a plan revision, and the measurement is repeated.

Then open a PR `migration/iree` → `main`.

## 6. Test plan

| Layer | What | Where / command | Runs in |
|---|---|---|---|
| Toolchain | recipe validation, W8A32 rule, QA weight formula, tiny end-to-end convert | `python -m unittest discover -s tests` in `tools/iree` | cloud |
| FFI | qa-tiny both precisions, both param modes, error paths, wrong ISA, 4-thread sharing; Metal variant on macOS | `cargo test -p lumen-iree` | CI Linux x86-64, Linux arm64, macOS, Windows |
| Hub unit | placement rule, download plan, config rewrite/validation, OCR resize rule (exactly the 57 entry shapes over a sweep of image sizes; identical to the old rule whenever §3.6.2 says so), wrapper shape checks, `dump-inputs` | `cargo test -p lumen-hub` | CI |
| L0 e2e | lifecycle, control, batcher (batch>1 loops), contract, infer with qa-tiny | `just l0` | CI Linux (cpu) + macOS (metal) |
| L1 | real weights, goldens, fp32 references, OCR vs Burn | `just l1-backend …` | `nightly-models` (macOS cpu/metal, also `workflow_dispatch` on the branch) + maintainer hardware |
| Conversion gates | §3.12 | `tools/iree convert` | maintainer |

## 7. Risks and fixed responses

| Risk | Response (already decided) |
|---|---|
| Maintainer ONNX differs from the recipe (input dtype/shape, output count/shape, opset > 17, `model_info` OCR shapes) | `convert` stops with the exact mismatch; fix the ONNX export or `model_info.json`, never the recipe values of §3.6. |
| A production graph fails to import or compile (an operator pattern not covered by F1) | Re-export that ONNX at opset ≤ 17 without custom domains and rerun `convert`. If it still fails, record the failing operator and the compiler error in `docs/lumen-hub-iree-runtime-decision.md`; Phase 2 stays incomplete for that model and the branch is not merged (no partial migration). |
| Quality gate fails for a W8A32 component | The gate is not lowered. The component's recipe switches to `"quantize": false` (fp32 weights inside the `w8a32` set), with the measured cosine recorded in the recipe commit message. |
| Metal (or CUDA) runtime numerics differ from CPU | Phase 8 cross-device check; a failing target is removed from §3.4 for that precision in a plan revision before merge. |
| Metal cold start too slow (MSL compiled at load) | Phase 8 rule: > 60 s slower than the CPU build ⇒ switch `metal-macos` to precompiled metallib. |
| OCR quality change from §3.6.2 | Phase 5 OCR acceptance (same text on shared boxes); upscaling small images is the RapidOCR default behavior. |
| Old hubs read updated `model_info.json` | `runtimes` keeps its schema; `burn/` stays; Phase 2 acceptance checks parsing with the released schema. |

## 8. File inventory

Created: `crates/lumen-iree-sys/**`, `crates/lumen-iree/**`, `crates/lumen-hub/src/runtime.rs`, `docs/lumen-hub-iree-runtime-decision.md` (Phase 8). Already present on the branch: `tools/iree/**`, `fixtures/iree/qa-tiny/**`, `docs/iree-migration/reference/**`, this document.

Modified: `Cargo.toml` (workspace unchanged except members via glob), `.gitignore`, `justfile`, `.github/workflows/{ci,release,nightly-models}.yml`, `crates/xtask/src/main.rs`, `crates/lumen-schema/src/{config/lumen_config.rs,config/render.rs,preset.rs,manifest.rs}`, `crates/lumen-launcher/src/setup.rs`, `crates/lumen-hub/{Cargo.toml,src/main.rs,src/inference_worker.rs,src/warmup.rs,src/status.rs,src/model_download.rs,src/service/*.rs,src/models/**,tests/**}`, `fixtures/config/*.yaml`, `crates/lumen-hub/examples/*.yaml`, `packaging/docker/*`, `schemas/*`, `README.md`.

Deleted: `crates/lumen-hub/src/backend.rs`, `crates/lumen-hub/src/model_arch/**`, `crates/lumen-quant-core/**`, `crates/lumen-hub/examples/{quantize_check,repro_quant_roundtrip,siglip_smoke}.rs`, `crates/lumen-hub/tools/check_onnx_candle_ops.py`.

## 9. Glossary

entry point: an exported function of a compiled module with fixed input shape. component: one ONNX graph of a model (`vision`, `text`, …), compiled to one `.vmfb` per target sharing one `.irpa`. target: `cpu-x86_64`, `cpu-aarch64`, `cuda-sm_75`, `metal-macos`. driver: IREE HAL driver (`local-task`, `cuda`, `metal`).

## 10. Definition of Done

1. Phases 1–8 complete with all acceptance criteria.
2. `cargo tree` contains no `burn*`/`cubecl*` package; no file references `fp16q8` outside historical docs.
3. `just ci`, `just contract`, `just l0` green on the branch; nightly L1 green.
4. The diff `main...migration/iree` contains no change under `crates/lumen-hub/proto/` and no change to `crates/lumen-hub/src/daemon/*.v1.rs` generated files.

## Appendix A — Shim ABI and the IREE C API behind it

Header: `docs/iree-migration/reference/lumen-iree-sys/csrc/lumen_iree.h` (normative). Implementation: `lumen_iree.c` in the same directory.

| Shim function | IREE C API sequence |
|---|---|
| `lumen_iree_runtime_create(driver)` | `iree_runtime_instance_options_initialize` → `iree_runtime_instance_options_use_all_available_drivers` → `iree_runtime_instance_create(&opts, iree_allocator_system())` → `iree_runtime_instance_try_create_default_device(instance, driver)` |
| `lumen_iree_runtime_release` | `iree_hal_device_release`, `iree_runtime_instance_release` |
| `lumen_iree_model_load(vmfb, irpa, scope, mode)` | `iree_runtime_session_options_initialize` → `iree_runtime_session_create_with_device` (registers the `hal` module) → params module: `iree_io_parameter_index_create`; `iree_io_file_handle_open(READ\|SHARE_READ)`; for `MMAP`: `iree_io_file_map_view(…, 0, IREE_HOST_SIZE_MAX, EXCLUDE_FROM_DUMPS)` + `iree_io_file_mapping_contents_ro` + `iree_io_file_handle_wrap_host_allocation` (release callback releases the mapping); `iree_io_parse_file_index`; `iree_io_parameter_index_provider_create(scope, …, IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS)`; `iree_io_parameters_module_create` → `iree_runtime_session_append_module(params)` → `iree_io_file_contents_map(vmfb, READ)` → `iree_runtime_session_append_bytecode_module_from_memory(…, iree_io_file_contents_deallocator(contents))` |
| `lumen_iree_model_has_function` | `iree_runtime_session_lookup_function("module.<fn>")` |
| `lumen_iree_model_invoke` | `iree_runtime_call_initialize_by_name("module.<fn>")` → per input `iree_hal_buffer_view_allocate_buffer_copy(device, allocator, rank, dims, {FLOAT_32\|INT_64\|INT_32}, DENSE_ROW_MAJOR, {DEVICE_LOCAL, ALL, DEFAULT}, bytes)` + `iree_runtime_call_inputs_push_back_buffer_view` → `iree_runtime_call_invoke` → `iree_vm_list_size(iree_runtime_call_outputs)` × `iree_runtime_call_outputs_pop_front_buffer_view` → `iree_runtime_call_deinitialize` |
| `lumen_iree_outputs_describe` | `iree_hal_buffer_view_element_type`, `_shape_rank`, `_shape_dim`, `_byte_length` |
| `lumen_iree_outputs_read` | `iree_hal_device_transfer_d2h(device, iree_hal_buffer_view_buffer(view), 0, dst, len, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout())` |
| `lumen_iree_model_trim` | `iree_runtime_session_trim` |
| errors | `iree_status_to_string` (fallback `iree_status_code_string(iree_status_code(s))`), `iree_status_free` |

Element types at the ABI: `F32` ↔ `IREE_HAL_ELEMENT_TYPE_FLOAT_32`, `I64` ↔ `INT_64` (accepts `SINT_64` on output), `I32` ↔ `INT_32` (accepts `SINT_32`).

## Appendix B — Recipes

`tools/iree/recipes/{siglip2-base-patch16-224,siglip2-so400m-patch14-384,bioclip-2,antelopev2,pp-ocrv6-small,qa-tiny}.json`; schema documented at the top of `tools/iree/lumen_iree_tools/recipes.py`.

## Appendix C — Reproducing the findings

All numbers in §2 come from the toolchain in `tools/iree` plus small scripts run during research:
- fp32 parity, conversion and gates: `python -m lumen_iree_tools convert …` (see `iree/BUILD.*.json` for the recorded cosines).
- Latency: `iree-benchmark-module --module=<vmfb> --parameters=model=<irpa> --function=<fn> --input=<shape>xf32=0.1 --device=local-task --benchmark_repetitions=5`.
- Memory: load through the shim with `ParamsMode::Mmap` vs `Read` and read `RssAnon`/`RssFile` from `/proc/self/status` after one invocation.
- Quantized accuracy: onnxruntime on W8A32 graphs from `quantize.py` and W8A8 graphs from `onnxruntime.quantization.quantize_static` (QDQ, `Conv/MatMul/Gemm`, per-tensor and per-channel, MinMax/Percentile 99.999, `MatMulConstBOnly`), evaluated with task metrics (detection F1 at IoU ≥ 0.5, keypoint L2, embedding cosine, CTC greedy string match).
- Compile failures: `iree-compile` with the flags of §3.4 on the dynamic-shape OCR graph, on the 32-pixel OCR buckets, and on a single W8A32 `MatMul` with `M ≥ 16` for `metal-macos`.
