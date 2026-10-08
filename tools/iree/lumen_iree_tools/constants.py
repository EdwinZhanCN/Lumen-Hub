"""Fixed facts of the Lumen IREE toolchain.

Everything here is part of the artifact contract shared with lumen-hub
(crates/lumen-hub) and lumen-iree-sys (crates/lumen-iree-sys). Changing any
value requires regenerating every published artifact and the committed QA
fixtures; see docs/lumen-hub-iree-migration-plan.md.
"""

from __future__ import annotations

from dataclasses import dataclass

# The IREE release used for compilation. lumen-iree-sys builds the runtime from
# the same release tag; vmfb bytecode is only guaranteed to load in the runtime
# of the compiler release that produced it.
IREE_VERSION = "3.12.0"
IREE_GIT_TAG = "v3.12.0"
IREE_GIT_COMMIT = "2b05c5dbb2f2ecb27c0d3941e80ee8d2f16e890d"

# Graphs are normalized to this ONNX opset before import.
ONNX_OPSET = 17

# Parameter scope used by the importer and by the runtime parameter provider.
PARAM_SCOPE = "model"

# Initializers with at least this many elements become named parameters in the
# .irpa archive; smaller ones (shape vectors, scalars) are inlined into the
# module.
PARAM_MIN_ELEMENTS = 32

# Precision tags. A tag names the whole artifact set of a model:
#   fp32  - float32 weights, float32 activations.
#   w8a32 - symmetric int8 weights with one float32 scale per output channel
#           (per row for embedding tables), float32 activations. Components
#           whose recipe sets "quantize": false keep float32 weights.
PRECISION_FP32 = "fp32"
PRECISION_W8A32 = "w8a32"
PRECISIONS = (PRECISION_FP32, PRECISION_W8A32)

# Data-tiling flag; only ever passed to CPU targets (recipe "cpu_data_tiling").
CPU_DATA_TILING_FLAG = "--iree-opt-data-tiling"


@dataclass(frozen=True)
class Target:
    """One compilation target = one .vmfb per component."""

    name: str
    flags: tuple[str, ...]
    is_cpu: bool
    # IREE runtime driver that executes this target.
    driver: str


TARGETS: dict[str, Target] = {
    # Embedded ELF (OS-independent) for x86-64-v3 (AVX2/FMA/BMI2, Haswell+).
    "cpu-x86_64": Target(
        name="cpu-x86_64",
        flags=(
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-llvmcpu-target-triple=x86_64-unknown-unknown-eabi-elf",
            "--iree-llvmcpu-target-cpu=x86-64-v3",
        ),
        is_cpu=True,
        driver="local-task",
    ),
    # Embedded ELF (OS-independent) for baseline ARMv8.0-A with NEON.
    "cpu-aarch64": Target(
        name="cpu-aarch64",
        flags=(
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
            "--iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf",
            "--iree-llvmcpu-target-cpu=generic",
        ),
        is_cpu=True,
        driver="local-task",
    ),
    # PTX (ISA 7.6) for sm_75; the CUDA driver JIT-compiles it for any GPU with
    # compute capability >= 7.5 (Turing, Ampere incl. Jetson Orin, Ada, ...).
    "cuda-sm_75": Target(
        name="cuda-sm_75",
        flags=(
            "--iree-hal-target-device=cuda",
            "--iree-cuda-target=sm_75",
        ),
        is_cpu=False,
        driver="cuda",
    ),
    # Metal Shading Language source compiled by the Metal driver at load time.
    "metal-macos": Target(
        name="metal-macos",
        flags=(
            "--iree-hal-target-device=metal",
            "--iree-metal-target-platform=macos",
        ),
        is_cpu=False,
        driver="metal",
    ),
}

ALL_TARGETS = tuple(TARGETS)

# Published target set per precision. metal-macos is excluded for w8a32:
# IREE 3.12's SPIR-V/Metal code generation does not terminate for int8-weight
# matmuls with more than one row (measured: M>=16 never finishes, M=1 compiles),
# so lumen-hub runs w8a32 models on the CPU driver in Metal builds.
TARGETS_BY_PRECISION: dict[str, tuple[str, ...]] = {
    PRECISION_FP32: ("cpu-x86_64", "cpu-aarch64", "cuda-sm_75", "metal-macos"),
    PRECISION_W8A32: ("cpu-x86_64", "cpu-aarch64", "cuda-sm_75"),
}


def vmfb_name(component: str, precision: str, target: str) -> str:
    return f"{component}.{precision}.{target}.vmfb"


def irpa_name(component: str, precision: str) -> str:
    return f"{component}.{precision}.irpa"
