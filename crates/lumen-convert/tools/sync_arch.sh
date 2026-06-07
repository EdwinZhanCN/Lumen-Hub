#!/usr/bin/env bash
# Sync burn-onnx generated architectures from lumen-convert's OUT_DIR into
# lumen-hub's committed model_arch, applying the `conv_fwd` pointwise-conv
# workaround (the same patch the existing mobile pp_ocrv5 arch carries — it routes
# 1x1 convolutions through matmul to dodge a burn-wgpu/metal correctness bug).
#
# The only transformation applied to the generated code is:
#   self.conv2dN.forward(ARG)  ->  crate::model_arch::conv_fwd(&self.conv2dN, ARG)
# which is mathematically equivalent (1x1 conv == matmul) and idempotent.
#
# Usage (after `cargo build -p lumen-convert` has run ModelGen):
#   tools/sync_arch.sh <repo_module> <component>...
#   tools/sync_arch.sh pp_ocrv5_server detection recognition classification
set -euo pipefail

REPO="${1:?usage: sync_arch.sh <repo_module> <component>...}"
shift
COMPONENTS=("$@")
[ "${#COMPONENTS[@]}" -gt 0 ] || { echo "give at least one component"; exit 1; }

CRATE_DIR="$(cd "$(dirname "$0")/.." && pwd)"          # crates/lumen-convert
WS="$(cd "$CRATE_DIR/../.." && pwd)"                   # workspace root
DEST="$WS/crates/lumen-hub/src/model_arch/$REPO"

OUT=$(find "$WS/target" -type d -path "*lumen-convert-*/out/$REPO" 2>/dev/null | head -1)
[ -n "$OUT" ] || { echo "no generated arch for '$REPO' under target/; run: cargo build -p lumen-convert"; exit 1; }
echo "source: $OUT"

mkdir -p "$DEST"
for c in "${COMPONENTS[@]}"; do
  src="$OUT/$c/$c.rs"
  [ -f "$src" ] || { echo "missing generated $src"; exit 1; }
  sed -E 's/self\.(conv2d[0-9]+)\.forward\(/crate::model_arch::conv_fwd(\&self.\1, /g' \
    "$src" > "$DEST/$c.rs"
  patched=$(grep -c 'conv_fwd(&self' "$DEST/$c.rs" || true)
  leftover=$(grep -cE 'self\.conv2d[0-9]+\.forward\(' "$DEST/$c.rs" || true)
  echo "  $c.rs: $patched conv_fwd, $leftover leftover raw"
done

{
  echo "//! Generated architectures for \`$REPO\`, synced by lumen-convert"
  echo "//! tools/sync_arch.sh with the \`conv_fwd\` pointwise-conv patch applied"
  echo "//! (see super::conv_fwd). Do not hand-edit; re-run the script to update."
  printf 'pub mod %s;\n' $(printf '%s\n' "${COMPONENTS[@]}" | sort)
} > "$DEST/mod.rs"
echo "wrote $DEST/{$(IFS=,; echo "${COMPONENTS[*]}").rs, mod.rs}"
echo "remember: register \`pub mod $REPO;\` in model_arch/mod.rs and dispatch in models/<family>/model.rs"
