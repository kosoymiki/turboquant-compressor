#!/usr/bin/env bash
# Verify TurboQuant OpenCL self-tests through source and SPIR-V paths.
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALLER="$ROOT_DIR/scripts/install-driver-root.sh"
CLI_BIN="${TQ_OPENCL_CLI:-${TQ_OPENCL_BUILD_DIR:-$HOME/.cache/turboquant/native-opencl-build}/tq_opencl_cli}"
OUT="$ROOT_DIR/forensics/spirv-parity-report.json"
TMP_DIR="$(mktemp -d)"
LOCK_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/turboquant/opencl-serial.lock"
trap 'rm -rf "$TMP_DIR"' EXIT

acquire_lock() {
  mkdir -p "$(dirname "$LOCK_DIR")"
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    sleep 1
  done
  trap 'rm -rf "$TMP_DIR"; rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT INT TERM
}

resolve_driver_root() {
  if [ -n "${TQ_DRIVER_ROOT:-}" ] && [ -f "${TQ_DRIVER_ROOT}/env.sh" ]; then
    printf '%s\n' "${TQ_DRIVER_ROOT}"
    return
  fi
  local repo_driver="${ROOT_DIR}/native/opencl/driver-root"
  if [ -f "${repo_driver}/env.sh" ]; then
    printf '%s\n' "${repo_driver}"
    return
  fi
  return 1
}

ensure_driver_root() {
  if DRIVER_ROOT="$(resolve_driver_root || true)" && [ -n "${DRIVER_ROOT}" ]; then
    printf '%s\n' "${DRIVER_ROOT}"
    return 0
  fi
  if [ -f "$INSTALLER" ]; then
    bash "$INSTALLER" >&2 || true
  fi
  DRIVER_ROOT="$(resolve_driver_root || true)"
  [ -n "${DRIVER_ROOT}" ] || return 1
  printf '%s\n' "${DRIVER_ROOT}"
}

filter_known_stderr() {
  local file="$1"
  grep -v -E 'failed to open /dev/dri/renderD128: Permission denied|Unsupported SPIR-V capability: SpvCapabilityGenericPointer|^[[:space:]]*[0-9]+ bytes into the SPIR-V binary$|^SPIR-V WARNING:$|^[[:space:]]*In file .*spirv_to_nir\.c:5557$' "$file" >&2 || true
}

run_mode() {
  local cmd="$1"
  local mode="$2"
  local out_json="$3"
  local err_log="$4"
  local env_sh="$5"
  local mode_flag="--${mode}"
  local inner="source '$env_sh' >/dev/null 2>&1 && exec '$CLI_BIN' '$cmd' --self-test '$mode_flag'"
  if command -v timeout >/dev/null 2>&1; then
    env -i HOME="$HOME" USER="${USER:-u0_a452}" SHELL="${SHELL:-/data/data/com.termux/files/usr/bin/bash}" \
      PREFIX="${PREFIX:-/data/data/com.termux/files/usr}" TMPDIR="${TMPDIR:-/data/data/com.termux/files/usr/tmp}" \
      PATH="${PREFIX:-/data/data/com.termux/files/usr}/bin:/system/bin" \
      timeout -k 5s 90s bash -lc "$inner" >"$out_json" 2>"$err_log"
  else
    env -i HOME="$HOME" USER="${USER:-u0_a452}" SHELL="${SHELL:-/data/data/com.termux/files/usr/bin/bash}" \
      PREFIX="${PREFIX:-/data/data/com.termux/files/usr}" TMPDIR="${TMPDIR:-/data/data/com.termux/files/usr/tmp}" \
      PATH="${PREFIX:-/data/data/com.termux/files/usr}/bin:/system/bin" \
      bash -lc "$inner" >"$out_json" 2>"$err_log"
  fi
  filter_known_stderr "$err_log"
}

if [ ! -x "$CLI_BIN" ]; then
  echo "missing tq_opencl_cli: $CLI_BIN" >&2
  exit 1
fi

DRIVER_ROOT="$(ensure_driver_root)"
ENV_SH="$DRIVER_ROOT/env.sh"
commands=(mse-score qjl-score value-dequant fused-attention)

acquire_lock

printf '{\n  "commands": [\n' >"$OUT"
all_ok=true
for i in "${!commands[@]}"; do
  cmd="${commands[$i]}"
  source_json="$TMP_DIR/${cmd}-source.json"
  spirv_json="$TMP_DIR/${cmd}-spirv.json"
  source_err="$TMP_DIR/${cmd}-source.err"
  spirv_err="$TMP_DIR/${cmd}-spirv.err"

  run_mode "$cmd" source "$source_json" "$source_err" "$ENV_SH"
  run_mode "$cmd" spirv "$spirv_json" "$spirv_err" "$ENV_SH"

  parity_equal=false
  if cmp -s "$source_json" "$spirv_json"; then
    parity_equal=true
  else
    all_ok=false
  fi

  printf '    {"command":"%s","source":' "$cmd" >>"$OUT"
  cat "$source_json" >>"$OUT"
  printf ',"spirv":' >>"$OUT"
  cat "$spirv_json" >>"$OUT"
  printf ',"identical":%s}' "$parity_equal" >>"$OUT"
  if [ "$i" -lt "$((${#commands[@]} - 1))" ]; then
    printf ',\n' >>"$OUT"
  else
    printf '\n' >>"$OUT"
  fi
done
printf '  ],\n  "all_pass": %s\n}\n' "$all_ok" >>"$OUT"

if [ "$all_ok" != true ]; then
  echo "FAIL [verify-spirv-parity]: source and SPIR-V outputs diverged" >&2
  exit 1
fi

echo "[OK] verify-spirv-parity: $OUT"
