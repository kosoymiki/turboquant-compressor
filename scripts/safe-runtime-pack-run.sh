#!/usr/bin/env bash
set -euo pipefail

# Safe one-shot runner for a built TurboQuant driver root.
# Purpose:
# - keep tests short-lived
# - isolate environment from ambient shell state
# - avoid broad foreground loops that can destabilize Termux responsiveness
#
# Usage:
#   scripts/safe-runtime-pack-run.sh probe
#   scripts/safe-runtime-pack-run.sh benchmark --warmup 1 --iters 1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALLER="$ROOT_DIR/scripts/install-driver-root.sh"
LOCK_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/turboquant/opencl-serial.lock"
LOCK_PID_FILE="$LOCK_DIR/pid"
LOCK_WAIT_SECONDS="${TQ_OPENCL_LOCK_WAIT_SECONDS:-20}"
LOCK_STALE_SECONDS="${TQ_OPENCL_LOCK_STALE_SECONDS:-120}"
RUNTIME_TIMEOUT_SECONDS="${TQ_OPENCL_TIMEOUT_SECONDS:-45}"

acquire_lock() {
  mkdir -p "$(dirname "$LOCK_DIR")"
  local start_ts now_ts lock_pid lock_age
  start_ts="$(date +%s)"
  while ! mkdir "$LOCK_DIR" 2>/dev/null; do
    if [ -f "$LOCK_PID_FILE" ]; then
      lock_pid="$(cat "$LOCK_PID_FILE" 2>/dev/null || true)"
      if [ -n "$lock_pid" ] && ! kill -0 "$lock_pid" 2>/dev/null; then
        rm -rf "$LOCK_DIR" 2>/dev/null || true
        continue
      fi
    fi
    now_ts="$(date +%s)"
    lock_age=$((now_ts - start_ts))
    if [ "$lock_age" -ge "$LOCK_STALE_SECONDS" ]; then
      rm -rf "$LOCK_DIR" 2>/dev/null || true
      continue
    fi
    if [ "$lock_age" -ge "$LOCK_WAIT_SECONDS" ]; then
      echo "[safe-runtime-pack-run] timed out waiting for runtime-pack lock: $LOCK_DIR" >&2
      exit 124
    fi
    sleep 1
  done
  printf '%s\n' "$$" > "$LOCK_PID_FILE"
  trap 'rm -rf "$LOCK_DIR" 2>/dev/null || true' EXIT INT TERM
}
resolve_cli() {
  if [ -n "${TQ_OPENCL_CLI:-}" ] && [ -x "${TQ_OPENCL_CLI}" ]; then
    printf '%s\n' "${TQ_OPENCL_CLI}"
    return
  fi
  local cache_home="${XDG_CACHE_HOME:-$HOME/.cache}"
  local cache_cli="${cache_home}/turboquant/native-opencl-build/tq_opencl_cli"
  if [ -x "${cache_cli}" ]; then
    printf '%s\n' "${cache_cli}"
    return
  fi
  local repo_cli="${ROOT_DIR}/native/opencl/build/tq_opencl_cli"
  if [ "${TQ_ALLOW_REPO_LOCAL_RUNTIME_PACK:-0}" = "1" ] && [ -x "${repo_cli}" ]; then
    printf '%s\n' "${repo_cli}"
    return
  fi
  return 1
}
CLI="$(resolve_cli || true)"

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

DRIVER_ROOT="$(ensure_driver_root || true)"
ENV_SH="${DRIVER_ROOT:+$DRIVER_ROOT/env.sh}"

if [ -z "${CLI}" ] || [ ! -x "$CLI" ]; then
  echo "[safe-runtime-pack-run] missing CLI; set TQ_OPENCL_CLI or build native sidecar" >&2
  exit 1
fi

if [ -z "${DRIVER_ROOT}" ] || [ ! -f "$ENV_SH" ]; then
  echo "[safe-runtime-pack-run] missing driver stack env.sh; expected ${ROOT_DIR}/native/opencl/driver-root or set TQ_DRIVER_ROOT" >&2
  exit 1
fi

cmd="${1:-}"
shift || true

case "$cmd" in
  probe|frontier-smoke|system-svm-smoke|async-build-smoke|runtime-scheduler-smoke|inference-runtime-smoke|paged-kv-prefix-cache-smoke|benchmark|mse-score|qjl-score|value-dequant|fused-attention)
    ;;
  *)
    echo "[safe-runtime-pack-run] unsupported command: $cmd" >&2
    exit 2
    ;;
esac

acquire_lock

shell_quote() {
  printf "%q " "$@"
}

ARGS_Q="$(shell_quote "$@")"
INNER_CMD="source '$ENV_SH' >/dev/null 2>&1 && exec '$CLI' '$cmd'"
if [ -n "${ARGS_Q// }" ]; then
  INNER_CMD+=" ${ARGS_Q}"
fi

ENV_RUNNER=(
  env -i
  HOME="$HOME"
  USER="${USER:-u0_a452}"
  SHELL="${SHELL:-/data/data/com.termux/files/usr/bin/bash}"
  PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
  TMPDIR="${TMPDIR:-/data/data/com.termux/files/usr/tmp}"
  PATH="${PREFIX:-/data/data/com.termux/files/usr}/bin:/system/bin"
  RUSTICL_DEBUG="${RUSTICL_DEBUG:-}"
  RUSTICL_FEATURES="${RUSTICL_FEATURES:-}"
  TQ_OPENCL_TRACE="${TQ_OPENCL_TRACE:-}"
  TQ_OPENCL_INTERCEPT_LIB="${TQ_OPENCL_INTERCEPT_LIB:-}"
  TQ_OPENCL_INTERCEPT_CONFIG="${TQ_OPENCL_INTERCEPT_CONFIG:-}"
  OCL_INTERCEPT_CONFIG_FILE="${OCL_INTERCEPT_CONFIG_FILE:-}"
  CLI_OpenCLFileName="${CLI_OpenCLFileName:-}"
  LD_PRELOAD="${LD_PRELOAD:-}"
)

if command -v timeout >/dev/null 2>&1; then
  if [ "${TQ_OPENCL_SILENCE_STDERR:-1}" = "1" ]; then
    TMP_ERR="$(mktemp "${TMPDIR:-/tmp}/tq-runtime-stderr.XXXXXX")"
    set +e
    "${ENV_RUNNER[@]}" timeout "${RUNTIME_TIMEOUT_SECONDS}s" bash -lc "$INNER_CMD" 2>"$TMP_ERR"
    rc=$?
    set -e
    grep -v -E 'failed to open /dev/dri/renderD128: Permission denied|Unsupported SPIR-V capability: SpvCapabilityGenericPointer|^[[:space:]]*[0-9]+ bytes into the SPIR-V binary$|^SPIR-V WARNING:$|^[[:space:]]*In file .*spirv_to_nir\.c:5557$' "$TMP_ERR" >&2 || true
    rm -f "$TMP_ERR"
    exit "$rc"
  else
    "${ENV_RUNNER[@]}" timeout "${RUNTIME_TIMEOUT_SECONDS}s" bash -lc "$INNER_CMD"
  fi
else
  if [ "${TQ_OPENCL_SILENCE_STDERR:-1}" = "1" ]; then
    TMP_ERR="$(mktemp "${TMPDIR:-/tmp}/tq-runtime-stderr.XXXXXX")"
    set +e
    "${ENV_RUNNER[@]}" bash -lc "$INNER_CMD" 2>"$TMP_ERR"
    rc=$?
    set -e
    grep -v -E 'failed to open /dev/dri/renderD128: Permission denied|Unsupported SPIR-V capability: SpvCapabilityGenericPointer|^[[:space:]]*[0-9]+ bytes into the SPIR-V binary$|^SPIR-V WARNING:$|^[[:space:]]*In file .*spirv_to_nir\.c:5557$' "$TMP_ERR" >&2 || true
    rm -f "$TMP_ERR"
    exit "$rc"
  else
    "${ENV_RUNNER[@]}" bash -lc "$INNER_CMD"
  fi
fi
