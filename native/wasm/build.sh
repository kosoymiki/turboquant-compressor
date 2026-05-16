#!/bin/bash
# Build turboquant-wasm for Node.js
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export PATH="$HOME/.cargo/bin:$PATH"

echo "Building turboquant-wasm (SIMD128)..."
RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown --release

echo "Generating JS bindings..."
mkdir -p pkg
wasm-bindgen \
  target/wasm32-unknown-unknown/release/turboquant_wasm.wasm \
  --out-dir pkg \
  --target nodejs \
  --omit-default-module-path

echo "Optimizing with wasm-opt..."
wasm-opt --enable-simd --enable-bulk-memory -O3 pkg/turboquant_wasm_bg.wasm -o pkg/turboquant_wasm_bg.wasm

echo "Done. Output:"
ls -lh pkg/*.wasm pkg/*.js pkg/*.d.ts
