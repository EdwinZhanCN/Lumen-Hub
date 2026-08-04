# syntax=docker/dockerfile:1.10

# Lumen Hub container image.
#
# This Dockerfile does NOT compile anything: it repackages a prebuilt
# `cargo xtask dist` staging directory (bin/ + warmup/ + licenses + README).
# Build it from the repository root, pointing DIST at an unzipped dist dir:
#
#   cargo xtask dist --profile linux-x64-cpu
#   docker build -f packaging/docker/Dockerfile \
#     --build-arg DIST=dist/lumen-hub-linux-x64-cpu \
#     --build-arg FLAVOR=cpu -t lumen-hub:cpu .
#
# Flavors (see packaging/docker/README.md):
#   cpu    — BASE=ubuntu:24.04, profile linux-{x64,arm64}-cpu
#   vulkan — BASE=ubuntu:24.04, profile linux-x64-gpu; installs Mesa's Vulkan
#            ICDs (ANV = Intel iGPU, RADV = AMD), so one image covers both
#            vendors with a plain `/dev/dri` passthrough — no vendor stack.
#   cuda   — BASE=nvidia/cuda:<ver>-base-ubuntu24.04 plus NVRTC only, profile
#            linux-x64-cuda; host needs the nvidia-container-toolkit.
#
# Linux release binaries are built on Ubuntu 22.04 (glibc 2.35), so the
# Ubuntu 24.04 CPU, Vulkan, and CUDA runtime images satisfy their ABI floor.
ARG BASE=ubuntu:24.04
FROM ${BASE}

ARG FLAVOR=cpu
ARG CUDA_NVRTC_VERSION=12.6.85-1
ARG VERSION=dev
ARG REVISION=unknown

LABEL org.opencontainers.image.title="Lumen Hub" \
    org.opencontainers.image.description="Zero-configuration LAN AI services for Lumilio" \
    org.opencontainers.image.source="https://github.com/EdwinZhanCN/Lumen-Hub" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.version="${VERSION}" \
    org.opencontainers.image.revision="${REVISION}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && if [ "$FLAVOR" = "vulkan" ]; then \
        apt-get install -y --no-install-recommends mesa-vulkan-drivers; \
    elif [ "$FLAVOR" = "cuda" ]; then \
        apt-get install -y --no-install-recommends \
            "cuda-nvrtc-12-6=${CUDA_NVRTC_VERSION}"; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Keep the dist layout (bin/ + sibling warmup/): the hub resolves its warmup
# fixtures relative to the executable's parent-of-parent directory.
# DIST defaults to the per-platform staging layout the release workflow
# prepares (dist/docker/linux-{amd64,arm64}), so one multi-platform buildx
# invocation picks the right binaries per arch; local single-arch builds
# override DIST with an unzipped profile dir instead.
ARG TARGETARCH
ARG DIST=dist/docker/linux-${TARGETARCH}
COPY --link --chmod=755 ${DIST}/bin/ /opt/lumen/bin/
COPY --link ${DIST}/warmup/ /opt/lumen/warmup/
COPY --link ${DIST}/LICENSE ${DIST}/README.md /opt/lumen/

# Complete Docker service catalog. Published Compose files use the strict
# LUMEN_REGION / LUMEN_PRESET environment contract to resolve the active set.
# With no Docker variables, this remains a valid standalone file config.
COPY --link packaging/docker/config.default.yaml /etc/lumen/config.yaml

# Model weights are fetched on first start into the cache dir; keep them on a
# volume so they survive image updates.
VOLUME /models
EXPOSE 50051

# The gRPC control plane binds before model download and warmup complete. Its
# standard health service switches to SERVING only when inference is ready.
# A generous first-start window keeps Docker in `starting` while weights are
# downloaded; any successful probe transitions to `healthy` immediately.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --start-interval=10s --retries=3 \
    CMD ["/opt/lumen/bin/lumen-hub", "healthcheck"]

ENTRYPOINT ["/opt/lumen/bin/lumen-hub"]
CMD ["--config", "/etc/lumen/config.yaml"]
