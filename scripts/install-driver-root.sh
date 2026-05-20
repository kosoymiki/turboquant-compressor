#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STACK_DIR="$ROOT_DIR/native/opencl"
DRIVER_ROOT="${TQ_DRIVER_ROOT_DIR:-$STACK_DIR/driver-root}"
ARCHIVE_PATH="${TQ_DRIVER_ARCHIVE_PATH:-$STACK_DIR/driver-pack/tq-driver-pack-adreno-a7xx-a8xx.tar.zst}"
FORCE="${TQ_DRIVER_INSTALL_FORCE:-0}"

log() {
  echo "[install-driver-root] $*"
}

have_env_sh() {
  [ -f "$DRIVER_ROOT/env.sh" ]
}

extract_archive() {
  mkdir -p "$DRIVER_ROOT"
  find "$DRIVER_ROOT" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  if tar --help 2>/dev/null | grep -q -- '--zstd'; then
    tar --zstd -xf "$ARCHIVE_PATH" -C "$DRIVER_ROOT"
    return
  fi
  if command -v zstd >/dev/null 2>&1; then
    zstd -dc "$ARCHIVE_PATH" | tar -xf - -C "$DRIVER_ROOT"
    return
  fi
  log "missing zstd-capable tar or zstd binary; cannot unpack $ARCHIVE_PATH"
  return 1
}

if [ ! -f "$ARCHIVE_PATH" ]; then
  log "archive not found: $ARCHIVE_PATH"
  exit 0
fi

if [ "$FORCE" != "1" ] && have_env_sh; then
  log "driver-root already installed at $DRIVER_ROOT"
  exit 0
fi

extract_archive
log "installed driver-root at $DRIVER_ROOT"
