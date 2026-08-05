# Lumen Hub proto module

- `ml_service.proto` — the data-plane contract. **Vendored byte-for-byte from
  [Lumen-SDK](https://github.com/EdwinZhanCN/Lumen-SDK)** (`proto/ml_service.proto`);
  see `provenance.json` for the pinned source tag/commit/SHA-256. Never edit
  this file in Lumen-Hub — update it with `cargo xtask contract-check --sync-sdk <tag>`.
- `control.proto` — the control-plane contract. **Authoritative in this
  repository**; the Lumilio-Photos Desktop vendors it byte-for-byte and
  verifies it against the pinned Hub release.

Go import-path differences (e.g. `option go_package`) are handled with
`protoc --go_opt=M...` at the consumer, never by editing the proto bytes.
`cargo xtask contract-check` enforces: byte-for-byte provenance, `buf lint`,
and `buf breaking` (WIRE_JSON) against the fixed current-major baseline tag.
