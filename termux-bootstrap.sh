#!/usr/bin/env bash
# Termux bootstrap script for turboquant-compressor

set -e

HOME_DIR="${HOME:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"
TQ_ROOT="${TURBOQUANT_ROOT:-$HOME_DIR/turboquant-compressor}"

pkg update -y
pkg install -y nodejs-lts

npm install --prefix "$TQ_ROOT"

cd "$TQ_ROOT"
npm run build

echo "Build complete. Run 'node dist/server.js' to start the MCP server."
