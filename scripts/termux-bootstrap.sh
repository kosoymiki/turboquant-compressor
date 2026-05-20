#!/usr/bin/env bash
# Termux bootstrap script for turboquant-compressor

set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="${TURBOQUANT_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)}"

pkg update -y
pkg install -y nodejs-lts

npm install --prefix "$REPO_ROOT"

cd "$REPO_ROOT"
npm run build

echo "Build complete. Run 'node dist/server.js' to start the MCP server."
