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
#   cuda   — BASE=nvidia/cuda:<ver>-runtime-ubuntu24.04 (runtime flavor ships
#            the NVRTC CubeCL needs), profile linux-x64-cuda; host needs the
#            nvidia-container-toolkit.
#
# The base must be a glibc >= 2.39 distro (binaries are built on ubuntu-24.04
# runners); ubuntu:24.04 and the ubuntu24.04 CUDA images satisfy that.
ARG BASE=ubuntu:24.04
FROM ${BASE}

ARG FLAVOR=cpu
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && if [ "$FLAVOR" = "vulkan" ]; then \
    apt-get install -y --no-install-recommends mesa-vulkan-drivers; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Keep the dist layout (bin/ + sibling warmup/): the hub resolves its warmup
# fixtures relative to the executable's parent-of-parent directory.
# DIST defaults to the per-platform staging layout the release workflow
# prepares (dist/docker/linux-{amd64,arm64}), so one multi-platform buildx
# invocation picks the right binaries per arch; local single-arch builds
# override DIST with an unzipped profile dir instead.
# The dist zips do not preserve the executable bit, so restore it.
ARG TARGETARCH
ARG DIST=dist/docker/linux-${TARGETARCH}
COPY ${DIST}/ /opt/lumen/
RUN chmod +x /opt/lumen/bin/*

# Default config: SigLIP + face + OCR on the models volume. Mount your own
# yaml over /etc/lumen/config.yaml to change services, region, or batching.
COPY packaging/docker/config.default.yaml /etc/lumen/config.yaml

# Model weights are fetched on first start into the cache dir; keep them on a
# volume so they survive image updates.
VOLUME /models
EXPOSE 50051

ENTRYPOINT ["/opt/lumen/bin/lumen-hub"]
CMD ["--config", "/etc/lumen/config.yaml"]
