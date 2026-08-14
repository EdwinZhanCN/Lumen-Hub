# Lumen Hub developer recipes.
#
# Pin matches CI (extractions/setup-just). Local install:
#   cargo binstall just@1.58.0
#   brew install just

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

just-version := "1.58.0"
model-features := "siglip,ppocr,insightface,clip"
l0-tests := "--test l0_lifecycle --test l0_infer --test l0_batcher --test l0_control --test l0_contract"

default:
    @just --list

# PR workspace job: fmt, workspace tests, config fixtures, L0 e2e (cpu + qa).
ci: fmt-check test config-fixtures-check l0

fmt-check:
    cargo fmt --all -- --check

test:
    cargo test --workspace

config-fixtures-check:
    cargo run -q -p xtask -- config-fixtures --check

# L0 e2e on the default cpu backend with the qa fixture model (no weight downloads).
l0:
    cargo test -p lumen-hub --features qa {{ l0-tests }}

# Verify lumen-hub compiles for a backend and the full model set.
check-backend backend:
    cargo check -p lumen-hub --no-default-features --features {{ backend }},{{ model-features }}

# L0 e2e on an explicit backend (metal CI job).
l0-backend backend:
    cargo test -p lumen-hub --no-default-features --features {{ backend }},{{ model-features }},qa {{ l0-tests }}

# CI backends matrix: compile check; L0 only on metal where a GPU is available.
ci-backend backend:
    #!/usr/bin/env bash
    set -euo pipefail
    just check-backend "{{ backend }}"
    if [ "{{ backend }}" = "metal" ]; then
        just l0-backend metal
    fi

# Deterministic proto contract check (requires `buf` on PATH).
contract: contract-check daemon-sync-check

contract-check:
    cargo run -q -p xtask -- contract-check

daemon-sync-check:
    cargo build -p lumen-hub
    git diff --exit-code -- crates/lumen-hub/src/daemon

# Remote SDK provenance + WIRE_JSON baseline (network; proto-contract workflow).
contract-verify:
    cargo run -q -p xtask -- contract-verify

# L1 nightly: real weights — semantic quality + golden regression.
l1-backend backend:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{ backend }}" = "metal" ]; then
        cargo test -p lumen-hub --release --no-default-features \
            --features metal,{{ model-features }} \
            --test l1_models --test l1_parity -- --test-threads=1
    else
        cargo test -p lumen-hub --release \
            --test l1_models --test l1_parity -- --test-threads=1
    fi

# --- xtask passthroughs (complex logic stays in Rust) ---

golden *args:
    cargo xtask golden {{ args }}

config-fixtures *args:
    cargo xtask config-fixtures {{ args }}

dist profile:
    cargo xtask dist --profile {{ profile }}
