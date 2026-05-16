#!/bin/bash
# Build turboquant-wasm for Node.js
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building turboquant-wasm..."
cargo build --target wasm32-unknown-unknown --release

echo "Generating JS bindings..."
mkdir -p pkg
wasm-bindgen \
  target/wasm32-unknown-unknown/release/turboquant_wasm.wasm \
  --out-dir pkg \
  --target nodejs \
  --omit-default-module-path

echo "Optimizing with wasm-opt..."
wasm-opt -O3 pkg/turboquant_wasm_bg.wasm -o pkg/turboquant_wasm_bg.wasm 2>/dev/null || true

echo "Done. Output: pkg/"
ls -lh pkg/
