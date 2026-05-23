#!/usr/bin/env python3
"""
TQ Universal Entry Point — single entry for ALL agents via TQ routing.
Replaces individual agent binaries with TQ-aware routing.

Usage:
  ./tq-entry.py <agent> <prompt> [args...]   # direct call
  ./claude "prompt"                           # via symlink (claude_code)
  ./codex "prompt"                            # via symlink (codex_cli)

Environment:
  NO_TQ=1  — bypass TQ, run native binary
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HOME = Path(os.environ.get("HOME", "/data/data/com.termux/files/home")).expanduser()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def find_tq_root() -> Path:
    search = [
        HOME / "corpus-control-plane" / "mcp" / "turboquant",
        HOME / "tmp_turboquant",
        HOME / "turboquant",
    ]
    for p in search:
        if p.exists() and (p / "bin" / "turboquant_agent_preflight.py").exists():
            return p
    return HOME / "tmp_turboquant"


def find_corpus_root() -> Path:
    search = [HOME / "corpus-control-plane", Path("/data/data/com.termux/files/home/corpus-control-plane")]
    for p in search:
        if p.exists() and (p / "bin" / "corpus_gateway.py").exists():
            return p
    return search[0]


TQ_ROOT = find_tq_root()
CORPUS_ROOT = find_corpus_root()


# =============================================================================
# AGENT MAPPING
# =============================================================================

# When called via symlink (claude, codex, gemini...) → map to TQ agent
SYMLINK_MAP = {
    "claude": "claude_code",
    "codex": "codex_cli",
    "gemini": "gemini_cli",
    "opencode": "opencode",
    "aider": "aider",
    "antgravy": "antgravy",
    "cursor": "cursor",
    "windsurf": "windsurf",
    "continue": "continue_",
    "codellama": "codellama",
    "phind": "phind",
    "youcode": "youcode",
    "roo_code": "roo_code",
}

# TQ internal name → binary name
BINARY_MAP = {
    "claude_code": "claude",
    "codex_cli": "codex",
    "gemini_cli": "gemini",
    "opencode": "opencode",
    "aider": "aider",
    "antgravy": "antgravy",
    "cursor": "cursor",
    "windsurf": "windsurf",
    "continue_": "continue",
    "codellama": "codellama",
    "phind": "phind",
    "youcode": "youcode",
    "roo_code": "roo_code",
    "generic": None,
}


def resolve_agent(name: str) -> tuple[str, str]:
    """Resolves agent name to (tq_agent, binary)."""
    # Check if it's a symlink name
    if name in SYMLINK_MAP:
        tq_agent = SYMLINK_MAP[name]
    else:
        tq_agent = name
    binary = BINARY_MAP.get(tq_agent, None)
    return tq_agent, binary


# =============================================================================
# TQ PIPELINE
# =============================================================================

def run_preflight(agent: str, prompt: str) -> dict:
    script = TQ_ROOT / "bin" / "turboquant_agent_preflight.py"
    if not script.exists():
        return {"error": "preflight not found"}
    cmd = ["python3", str(script), "--agent", agent, "--prompt", prompt]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=25, cwd=TQ_ROOT)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        return {"error": str(e)}


def corpus_search(query: str, top_k: int = 20) -> dict:
    script = CORPUS_ROOT / "bin" / "corpus_web_search.py"
    if not script.exists():
        return {"error": "search not found"}
    cmd = ["python3", str(script), "--query", query, "--top-k", str(top_k)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=CORPUS_ROOT)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        return {"error": str(e)}


def tq_context_pack_search(query: str) -> dict:
    script = TQ_ROOT / "dist" / "tools" / "context_pack_search.js"
    if not script.exists():
        return {}
    cmd = ["node", str(script), "--query", query, "--top-k", "10"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=TQ_ROOT)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except:
        return {}


def cache_lint(traces: list) -> dict:
    script = TQ_ROOT / "dist" / "tools" / "prompt_cache_lint.js"
    if not script.exists():
        return {}
    cmd = ["node", str(script), "--blocks-json", json.dumps(traces)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=TQ_ROOT)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except:
        return {}


def cache_plan(traces: list) -> dict:
    script = TQ_ROOT / "dist" / "tools" / "cache_plan.js"
    if not script.exists():
        return {}
    cmd = ["node", str(script), "--blocks-json", json.dumps(traces)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=TQ_ROOT)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except:
        return {}


# =============================================================================
# MAIN
# =============================================================================

def main():
    argv = sys.argv[1:]

    # If called as ./claude (via symlink), argv[0] is "claude" and rest is prompt
    # If called as ./tq-entry.py claude_code "prompt", argv[0] is "claude_code"
    script_name = Path(sys.argv[0]).name

    # Check if called via symlink (claude, codex, etc)
    if script_name in SYMLINK_MAP:
        # Called as: ./claude "prompt" or ./claude --flag "prompt"
        tq_agent = SYMLINK_MAP[script_name]
        binary = BINARY_MAP[tq_agent]
        # Parse remaining args
        parser = argparse.ArgumentParser()
        parser.add_argument("args", nargs=argparse.REMAINDER)
        args = parser.parse_args(argv)
        prompt = " ".join(a for a in args.args if not a.startswith("-"))
    else:
        # Called as: ./tq-entry.py <agent> <prompt> [args...]
        parser = argparse.ArgumentParser(description="TQ Universal Entry")
        parser.add_argument("agent", help="Agent name")
        parser.add_argument("args", nargs=argparse.REMAINDER)
        parsed = parser.parse_args(argv)
        tq_agent, binary = resolve_agent(parsed.agent)
        prompt = " ".join(a for a in parsed.args if not a.startswith("-"))

    # Extract flags from prompt args (for preflight)
    flags = [a for a in (argv if script_name in SYMLINK_MAP else parsed.args if 'parsed' in dir() else []) if a.startswith("-")]

    print(f"[TQ] {tq_agent}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}", file=sys.stderr)

    # Phase 1: TQ preflight
    preflight = run_preflight(tq_agent, prompt)

    # Phase 2: corpus_web_search (MANDATORY)
    corpus = corpus_search(prompt, top_k=20)

    # Phase 3: TQ context pack search
    pack_search = tq_context_pack_search(prompt)

    # Phase 4: Cache analysis
    traces = [
        {"label": "preflight", "text": json.dumps(preflight)[:500] if preflight else ""},
        {"label": "corpus", "text": json.dumps(corpus.get("results", [])[:20])[:500]},
    ]
    lint = cache_lint(traces)
    plan = cache_plan(traces)

    # Build TQ context
    tq_context = {
        "generated_at": now_iso(),
        "agent": script_name if script_name in SYMLINK_MAP else tq_agent,
        "tq_agent": tq_agent,
        "prompt": prompt[:200],
        "preflight": preflight,
        "corpus": {
            "count": corpus.get("count", 0),
            "results": corpus.get("results", [])[:10],
        },
        "pack_search": pack_search,
        "cache": {"lint": lint, "plan": plan},
        "route": preflight.get("route", {}),
        "additional_context": preflight.get("additional_context", ""),
    }

    # Write context file
    agent_home = HOME / f".{tq_agent.replace('_', '-')}"
    context_file = agent_home / ".turboquant-context.json"
    context_file.parent.mkdir(parents=True, exist_ok=True)
    context_file.write_text(json.dumps(tq_context, indent=2, ensure_ascii=False))

    # Phase 5: Run actual agent
    if binary and os.environ.get("NO_TQ") != "1":
        env = os.environ.copy()
        env["TURBOQUANT_CONTEXT_FILE"] = str(context_file)
        env["TURBOQUANT_CONTEXT"] = json.dumps(tq_context, ensure_ascii=False)[:4000]
        env["TURBOQUANT_ROUTE"] = preflight.get("route", {}).get("task_type", "generic") or "generic"
        env["TURBOQUANT_GPU_ROUTE"] = preflight.get("runtime", {}).get("production_route", "unknown") or "unknown"

        try:
            result = subprocess.run(
                [binary] + (argv if script_name not in SYMLINK_MAP else argv),
                env={k: (v if v is not None else "") for k, v in env.items()},
                timeout=120
            )
            return result.returncode
        except FileNotFoundError:
            print(f"[TQ] Binary not found: {binary}", file=sys.stderr)
            # Fall through to output context
        except subprocess.TimeoutExpired:
            return 124

    # Output TQ context
    print(json.dumps(tq_context, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())