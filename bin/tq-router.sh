#!/usr/bin/env bash
# TQ Agent Router — Intercepts ALL agent calls and routes through TQ by default.
# Install: source this file OR add to ~/.bashrc / ~/.zshrc
#
# Usage:
#   source tq-router.sh   # activate routing
#   claude "your prompt"  # runs through TQ automatically
#   NO_TQ=1 claude "..."  # bypass TQ for this call

set -euo pipefail

TQ_ROOT="${TQ_ROOT:-/data/data/com.termux/files/home/tmp_turboquant}"
CORPUS_ROOT="${CORPUS_ROOT:-/data/data/com.termux/files/home/corpus-control-plane}"
PROXY_SCRIPT="$TQ_ROOT/bin/agent-proxy.py"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log_tq() {
    echo -e "${CYAN}[TQ-ROUTE]${NC} $1" >&2
}

# Check if TQ is bypassed for this call
check_bypass() {
    [[ "${NO_TQ:-0}" == "1" ]] || [[ "${NO_TQ,,}" == "true" ]]
}

# Run through TQ proxy
run_via_tq() {
    local agent="$1"
    shift
    local prompt="$*"

    log_tq "→ $agent (TQ enabled)"

    python3 "$PROXY_SCRIPT" "$agent" --prompt "$prompt"
    return $?
}

# Run directly (bypass TQ)
run_direct() {
    local cmd="$1"
    shift

    log_tq "→ $cmd (TQ bypassed)"
    command "$cmd" "$@"
    return $?
}

# =============================================================================
# AGENT COMMANDS — override with TQ routing
# =============================================================================

claude() {
    if check_bypass; then
        command claude "$@"
    else
        local prompt="$*"
        run_via_tq "claude_code" "$prompt"
    fi
}

claude-code() {
    claude "$@"
}

codex() {
    if check_bypass; then
        command codex "$@"
    else
        local prompt="$*"
        run_via_tq "codex_cli" "$prompt"
    fi
}

gemini() {
    if check_bypass; then
        command gemini "$@"
    else
        local prompt="$*"
        run_via_tq "gemini_cli" "$prompt"
    fi
}

opencode() {
    if check_bypass; then
        command opencode "$@"
    else
        local prompt="$*"
        run_via_tq "opencode" "$prompt"
    fi
}

aider() {
    if check_bypass; then
        command aider "$@"
    else
        local prompt="$*"
        run_via_tq "aider" "$prompt"
    fi
}

antgravy() {
    if check_bypass; then
        command antgravy "$@"
    else
        local prompt="$*"
        run_via_tq "antgravy" "$prompt"
    fi
}

cursor() {
    if check_bypass; then
        command cursor "$@"
    else
        local prompt="$*"
        run_via_tq "cursor" "$prompt"
    fi
}

windsurf() {
    if check_bypass; then
        command windsurf "$@"
    else
        local prompt="$*"
        run_via_tq "windsurf" "$prompt"
    fi
}

continue_() {
    if check_bypass; then
        command continue "$@"
    else
        local prompt="$*"
        run_via_tq "continue_" "$prompt"
    fi
}

codellama() {
    if check_bypass; then
        command codellama "$@"
    else
        local prompt="$*"
        run_via_tq "codellama" "$prompt"
    fi
}

phind() {
    if check_bypass; then
        command phind "$@"
    else
        local prompt="$*"
        run_via_tq "phind" "$prompt"
    fi
}

youcode() {
    if check_bypass; then
        command youcode "$@"
    else
        local prompt="$*"
        run_via_tq "youcode" "$prompt"
    fi
}

# =============================================================================
# UTILITY COMMANDS
# =============================================================================

tq-status() {
    echo "=== TQ Router Status ==="
    echo "TQ_ROOT: $TQ_ROOT"
    echo "CORPUS_ROOT: $CORPUS_ROOT"
    echo "PROXY: $PROXY_SCRIPT"
    echo ""
    echo "TQ enabled by default. Use NO_TQ=1 to bypass."
    echo ""
    python3 "$PROXY_SCRIPT" audit
}

tq-list() {
    python3 "$PROXY_SCRIPT" list
}

tq-search() {
    python3 "$TQ_ROOT/bin/turboquant_pipeline.py" --action search --query "$*"
}

tq-audit() {
    python3 "$TQ_ROOT/bin/turboquant_pipeline.py" --action audit --audit-type full
}

tq-pipeline() {
    python3 "$TQ_ROOT/bin/turboquant_pipeline.py" "$@"
}

# =============================================================================
# ACTIVATION MESSAGE
# =============================================================================

log_tq "TurboQuant Router ACTIVE"
log_tq "All agents route through TQ by default"
log_tq "Use: source $0  (in ~/.bashrc or ~/.zshrc)"
log_tq ""
log_tq "Commands:"
log_tq "  claude 'prompt'     # Claude Code via TQ"
log_tq "  codex 'prompt'      # Codex via TQ"
log_tq "  gemini 'prompt'     # Gemini via TQ"
log_tq "  tq-status           # Router status"
log_tq "  tq-list             # All agents"
log_tq "  tq-search 'query'   # Corpus search"
log_tq "  NO_TQ=1 claude ...  # Bypass TQ"