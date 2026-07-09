# Lumen Hub in Docker

Container images for running the hub on a Linux server or NAS. The stable
contract is unchanged — the image just runs `lumen-hub --config
/etc/lumen/config.yaml` with a baked-in default config (SigLIP + face + OCR,
models cached on the `/models` volume).

## Pick a tag

| Tag | Hardware | Host setup |
|---|---|---|
| `cpu` (= `latest`) | anything, x64 + arm64 | none |
| `vulkan` | Intel iGPU (Skylake or newer) / AMD GPU | `/dev/dri` passthrough + render group |
| `cuda` | NVIDIA GPU (compute ≥ what the CUDA build targets) | [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) |

The `vulkan` image ships Mesa's open Vulkan drivers (ANV for Intel, RADV for
AMD), so one small image covers both vendors — no OpenVINO or ROCm stack.
NVIDIA-in-Vulkan is deliberately not supported; use the `cuda` tag.

```bash
docker run -d --name lumen-hub -p 50051:50051 -v lumen-models:/models \
  ghcr.io/edwinzhancn/lumen-hub:cpu
```

For `vulkan` add `--device /dev/dri --group-add "$(getent group render | cut -d: -f3)"`;
for `cuda` add `--gpus all`. Or start from the compose file next to this README.

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

Everything is overridable by mounting your own yaml over
`/etc/lumen/config.yaml` (see `config.default.yaml` here, or
`crates/lumen-hub/examples/`). Notably: `region: cn` for the HF mirror, and
the `bioclip` service (off by default — large catalog download).

mDNS discovery generally does not cross the container network boundary; point
clients (e.g. Lumilio Photos' `LUMEN_DISCOVERY_HUB_URL`) directly at
`<host>:50051`.

## Building locally

```bash
cargo xtask dist --profile linux-x64-gpu   # or linux-x64-cpu / linux-x64-cuda
docker build -f packaging/docker/Dockerfile \
  --build-arg DIST=dist/lumen-hub-linux-x64-gpu \
  --build-arg FLAVOR=vulkan \
  -t lumen-hub:vulkan .
```

`cuda` builds pass `--build-arg BASE=nvidia/cuda:<ver>-runtime-ubuntu24.04`
(the runtime flavor ships NVRTC; keep the version in step with the CUDA
toolkit the release workflow installs). CI builds and pushes all tags on tag
push — see `.github/workflows/release.yml`.
