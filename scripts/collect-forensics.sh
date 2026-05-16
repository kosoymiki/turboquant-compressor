#!/usr/bin/env bash
# Collect forensic environment snapshot for turboquant-compressor debugging.
# Output is written to a directory, then redacted before sharing.
# Usage: bash scripts/collect-forensics.sh [output-dir]

set -euo pipefail

OUT="${1:-/tmp/turboquant-forensics-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

echo "[forensics] collecting to $OUT"

# --- System ---
uname -a > "$OUT/uname.txt" 2>&1 || true
id > "$OUT/id.txt" 2>&1 || true
pwd > "$OUT/cwd.txt" 2>&1 || true
df -h > "$OUT/df.txt" 2>&1 || true
cat /proc/meminfo > "$OUT/meminfo.txt" 2>&1 || true
ulimit -a > "$OUT/ulimit.txt" 2>&1 || true

# --- Runtime ---
node --version > "$OUT/node_version.txt" 2>&1 || true
npm --version > "$OUT/npm_version.txt" 2>&1 || true
python3 --version > "$OUT/python3_version.txt" 2>&1 || true

# --- Termux ---
termux-info > "$OUT/termux_info.txt" 2>&1 || echo "termux-info not available" > "$OUT/termux_info.txt"

# --- Android ---
getprop ro.build.version.release > "$OUT/android_version.txt" 2>&1 || true
getprop ro.product.cpu.abi > "$OUT/cpu_abi.txt" 2>&1 || true
getprop ro.product.model > "$OUT/device_model.txt" 2>&1 || true

# --- ADB ---
adb devices -l > "$OUT/adb_devices.txt" 2>&1 || echo "adb not available" > "$OUT/adb_devices.txt"
adb shell getprop > "$OUT/adb_getprop.txt" 2>&1 || true
adb logcat -d -t 1000 > "$OUT/adb_logcat.txt" 2>&1 || true

# --- Project ---
npm run build > "$OUT/build.txt" 2>&1 || true
npm test -- --runInBand > "$OUT/test.txt" 2>&1 || true
node smoke-stdio.mjs > "$OUT/smoke_stdio.txt" 2>&1 || true
node scripts/mcp-transcript.mjs > "$OUT/mcp_transcript.txt" 2>&1 || true
node scripts/verify-scientific-claims.mjs > "$OUT/verify_scientific.txt" 2>&1 || true
node scripts/verify-format-contract.mjs > "$OUT/verify_format.txt" 2>&1 || true

# --- Redact ---
echo "[forensics] redacting secrets..."
node scripts/redact-forensics.mjs "$OUT"

echo "[forensics] done: $OUT"
ls "$OUT"
