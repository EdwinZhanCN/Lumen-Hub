# Lumen Hub in Docker

Container images for running the hub on a Linux server or NAS. The image runs
`lumen-hub --config /etc/lumen/config.yaml`, then applies the strict Docker
environment contract described below. Models are cached on the `/models`
volume.

## Pick a tag

| Tag | Hardware | Host setup |
|---|---|---|
| `cpu` (= `latest`) | anything, x64 + arm64 | none |
| `vulkan` | Intel iGPU (Skylake or newer) / AMD GPU | `/dev/dri` passthrough |
| `cuda` | NVIDIA GPU (compute ≥ what the CUDA build targets) | [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) |

The `vulkan` image ships Mesa's open Vulkan drivers (ANV for Intel, RADV for
AMD), so one small image covers both vendors — no OpenVINO or ROCm stack.
NVIDIA-in-Vulkan is deliberately not supported; use the `cuda` tag.

```bash
docker run -d --name lumen-hub --network host -v lumen-models:/models \
  -e LUMEN_REGION=other -e LUMEN_PRESET=basic \
  ghcr.io/edwinzhancn/lumen-hub:cpu
```

For `vulkan` add `--device /dev/dri`; for `cuda` add `--gpus all`.

Ready-to-import Compose files are published with Lumilio Photos under
[`deploy/compose`](https://github.com/EdwinZhanCN/Lumilio-Photos/tree/main/deploy/compose):

- `lumen-cpu.compose.yml`
- `lumen-vulkan.compose.yml`
- `lumen-cuda.compose.yml`

They are complete deployment files rather than templates: no separate `.env`
file or manual YAML editing is required. The Lumilio documentation wizard
embeds the selected values directly in the downloaded Compose. Host networking
is intentional because Lumen discovery uses mDNS on the LAN.

## Verifying GPU pickup

On startup the hub logs the adapter wgpu selected:

```
backend: wgpu adapter "Intel(R) Iris(R) Xe Graphics (ADL GT2)" (IntegratedGpu, Vulkan), driver ...
```

If it reports a `Cpu` device type (llvmpipe), the GPU was not passed through —
the hub prints a warning; fix the device/group flags or fall back to the `cpu`
tag. Pre-Skylake Intel iGPUs have no Vulkan 1.3 driver in current Mesa: use
`cpu`.

## Config

Published Compose files set these variables:

| Variable | Values | Meaning |
|---|---|---|
| `LUMEN_REGION` | `other`, `cn` | Official source or mainland-China model mirror |
| `LUMEN_PRESET` | `minimal`, `basic`, `brave`, `custom` | Complete supported capability plan |
| `LUMEN_SERVICES` | comma-separated `siglip`, `face`, `ocr`, `bioclip` | Required with `custom` |
| `LUMEN_SIGLIP_MODEL` | `siglip2-base-patch16-224`, `siglip2-so400m-patch14-384` | Optional custom semantic model |
| `LUMEN_FACE_MODEL` | `antelopev2` | Optional custom face model |
| `LUMEN_OCR_MODEL` | `pp-ocrv6-small` | Optional custom OCR Text Recognition model |
| `LUMEN_BIOCLIP_MODEL` | `bioclip-2` | Optional custom species model |
| `LUMEN_BIOCLIP_DATASET` | `TreeOfLife200MCore`, `TreeOfLife200M` | Optional custom species catalog |

Model variables are legal only with `LUMEN_PRESET=custom`, and only for a
service listed in `LUMEN_SERVICES`. Invalid combinations fail before model
download. `basic` is the published default and enables Image Semantic Analysis,
Person Recognition, OCR Text Recognition, and BioCLIP Species Recognition with
the Core catalog.

Advanced operators can still mount a complete yaml over
`/etc/lumen/config.yaml` (see `config.default.yaml` or
`crates/lumen-hub/examples/`). Remove the Docker configuration variables when
the mounted file should remain authoritative; otherwise the allow-listed
variables intentionally update it.

The image exposes a Docker health check backed by the standard gRPC health
service. It remains in `starting` during the first model download and becomes
`healthy` only after model loading and warmup complete.

## Building locally

```bash
cargo xtask dist --profile linux-x64-gpu   # or linux-x64-cpu / linux-x64-cuda
docker build -f packaging/docker/Dockerfile \
  --build-arg DIST=dist/lumen-hub-linux-x64-gpu \
  --build-arg FLAVOR=vulkan \
  -t lumen-hub:vulkan .
```

`cuda` builds pass `--build-arg BASE=nvidia/cuda:<ver>-base-ubuntu24.04`.
The Dockerfile installs only the matching NVRTC runtime package CubeCL needs,
instead of inheriting the full CUDA math-library runtime. Keep its version in
step with the CUDA toolkit used by the release workflow. CI builds and pushes
all tags on tag push — see `.github/workflows/release.yml`.
