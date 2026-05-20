#!/usr/bin/env bash
set -euo pipefail

# Safe one-shot runner for TurboQuant runtime-pack tests.
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
CLI="$ROOT_DIR/native/opencl/build/tq_opencl_cli"
RUNTIME_PACK="${TQ_RUNTIME_PACK_DIR:-$ROOT_DIR/native/opencl/runtime-pack}"
ENV_SH="$RUNTIME_PACK/env.sh"

if [ ! -x "$CLI" ]; then
  echo "[safe-runtime-pack-run] missing CLI: $CLI" >&2
  exit 1
fi

if [ ! -f "$ENV_SH" ]; then
  echo "[safe-runtime-pack-run] missing runtime env: $ENV_SH" >&2
  exit 1
fi

cmd="${1:-}"
shift || true

case "$cmd" in
  probe|benchmark|mse-score|qjl-score|value-dequant|fused-attention)
    ;;
  *)
    echo "[safe-runtime-pack-run] unsupported command: $cmd" >&2
    exit 2
    ;;
esac

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
)

if command -v timeout >/dev/null 2>&1; then
  if [ "${TQ_OPENCL_SILENCE_STDERR:-1}" = "1" ]; then
    TMP_ERR="$(mktemp "${TMPDIR:-/tmp}/tq-runtime-stderr.XXXXXX")"
    set +e
    "${ENV_RUNNER[@]}" timeout 45s bash -lc "$INNER_CMD" 2>"$TMP_ERR"
    rc=$?
    set -e
    grep -v -E 'failed to open /dev/dri/renderD128: Permission denied|Unsupported SPIR-V capability: SpvCapabilityGenericPointer|^[[:space:]]*[0-9]+ bytes into the SPIR-V binary$|^SPIR-V WARNING:$|^[[:space:]]*In file \.\./src/compiler/spirv/spirv_to_nir\.c:5557$' "$TMP_ERR" >&2 || true
    rm -f "$TMP_ERR"
    exit "$rc"
  else
    "${ENV_RUNNER[@]}" timeout 45s bash -lc "$INNER_CMD"
  fi
else
  if [ "${TQ_OPENCL_SILENCE_STDERR:-1}" = "1" ]; then
    TMP_ERR="$(mktemp "${TMPDIR:-/tmp}/tq-runtime-stderr.XXXXXX")"
    set +e
    "${ENV_RUNNER[@]}" bash -lc "$INNER_CMD" 2>"$TMP_ERR"
    rc=$?
    set -e
    grep -v -E 'failed to open /dev/dri/renderD128: Permission denied|Unsupported SPIR-V capability: SpvCapabilityGenericPointer|^[[:space:]]*[0-9]+ bytes into the SPIR-V binary$|^SPIR-V WARNING:$|^[[:space:]]*In file \.\./src/compiler/spirv/spirv_to_nir\.c:5557$' "$TMP_ERR" >&2 || true
    rm -f "$TMP_ERR"
    exit "$rc"
  else
    "${ENV_RUNNER[@]}" bash -lc "$INNER_CMD"
  fi
fi
