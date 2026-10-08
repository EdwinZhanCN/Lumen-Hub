# Reference implementation: lumen-iree-sys and lumen-iree

Verified starting point for Phase 1 of `docs/lumen-hub-iree-migration-plan.md`.
Phase 1 copies both directories verbatim to `crates/lumen-iree-sys` and
`crates/lumen-iree`. They live under `docs/` until then so the workspace
(`members = ["crates/*"]`) and its CI are not affected.

Verified on Linux x86-64 (Rust 1.94.1, clang 18 / gcc 13, CMake 3.28, Ninja)
against IREE `v3.12.0` (`2b05c5dbb2f2ecb27c0d3941e80ee8d2f16e890d`) with only
`third_party/flatcc` initialized:

- `cargo test --release` in a workspace containing both crates, with
  `LUMEN_IREE_SOURCE_DIR` pointing at the IREE checkout and
  `LUMEN_IREE_FIXTURE_DIR` at `fixtures/iree/qa-tiny`: 3/3 tests pass
  (both precisions × both parameter modes, error paths, wrong-ISA rejection,
  4 threads sharing one model).
- `cargo clippy --all-targets` reports no warnings; `cargo fmt --check` is clean.
- The shim (`lumen-iree-sys/csrc`) built together with the IREE runtime under
  `-fsanitize=address` and driven through runtime creation, model loading in
  both parameter modes, function lookup, successful and rejected invocations
  (wrong byte length, wrong shape), output description/reads, trim and every
  release function reports no AddressSanitizer or LeakSanitizer findings.

Not verifiable in that environment (covered by Phase 1 CI and Phase 8):
the `driver-metal` build on macOS, the `driver-cuda` build (needs the CUDA
Toolkit headers) and Windows/MSVC builds.
