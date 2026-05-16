#!/data/data/com.termux/files/usr/bin/bash
# Termux bootstrap script for turboquant-compressor

set -e

pkg update -y
pkg install -y nodejs-lts

npm install --prefix /data/data/com.termux/files/home/turboquant-compressor

cd /data/data/com.termux/files/home/turboquant-compressor
npm run build

echo "Build complete. Run 'node dist/server.js' to start the MCP server."