#!/usr/bin/env python3
"""
Agent Proxy Framework — TurboQuant integration for ALL agents.
Each agent gets: corpus_web_search → TQ preflight → TQ runtime → cache_plan → lint

Supported agents:
  claude_code, codex_cli, gemini_cli, opencode, aider, antgravy,
  codelLm, continue_, codellama, phind, youcode, cursor, windsurf,
  generic

Usage:
  ./agent-proxy.py <agent> --prompt "..."
  ./agent-proxy.py list              # show all agents
  ./agent-proxy.py audit            # audit all agents
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# =============================================================================
# AGENT REGISTRY — add new agents here
# =============================================================================

AGENTS = {
    "claude_code": {
        "home": "~/.claude",
        "binary": "claude",
        "desc": "Anthropic Claude Code",
        "TQ_AGENT": "claude_code",
    },
    "codex_cli": {
        "home": "~/.codex",
        "binary": "codex",
        "desc": "OpenAI Codex CLI",
        "TQ_AGENT": "codex_cli",
    },
    "gemini_cli": {
        "home": "~/.gemini",
        "binary": "gemini",
        "desc": "Google Gemini CLI",
        "TQ_AGENT": "gemini_cli",
    },
    "opencode": {
        "home": "~/.opencode",
        "binary": "opencode",
        "desc": "OpenCode CLI",
        "TQ_AGENT": "generic",
    },
    "aider": {
        "home": "~/.aider",
        "binary": "aider",
        "desc": "Aider Chat",
        "TQ_AGENT": "generic",
    },
    "antgravy": {
        "home": "~/.antgravy",
        "binary": "antgravy",
        "desc": "AntiGravy Agent",
        "TQ_AGENT": "generic",
    },
    "codelLm": {
        "home": "~/.codelLm",
        "binary": "codelLm",
        "desc": "CodeLLM",
        "TQ_AGENT": "generic",
    },
    "continue_": {
        "home": "~/.continue",
        "binary": "continue",
        "desc": "Continue.dev",
        "TQ_AGENT": "generic",
    },
    "codellama": {
        "home": "~/.codellama",
        "binary": "codellama",
        "desc": "Code Llama CLI",
        "TQ_AGENT": "generic",
    },
    "phind": {
        "home": "~/.phind",
        "binary": "phind",
        "desc": "Phind CLI",
        "TQ_AGENT": "generic",
    },
    "youcode": {
        "home": "~/.youcode",
        "binary": "youcode",
        "desc": "You.com YouCode",
        "TQ_AGENT": "generic",
    },
    "cursor": {
        "home": "~/.cursor",
        "binary": "cursor",
        "desc": "Cursor IDE",
        "TQ_AGENT": "generic",
    },
    "windsurf": {
        "home": "~/.windsurf",
        "binary": "windsurf",
        "desc": "Windsurf IDE",
        "TQ_AGENT": "generic",
    },
    "roo_code": {
        "home": "~/.roo_code",
        "binary": "roo_code",
        "desc": "Roo Code",
        "TQ_AGENT": "generic",
    },
    "generic": {
        "home": "~",
        "binary": None,
        "desc": "Generic agent",
        "TQ_AGENT": "generic",
    },
}

HOME = Path(os.environ.get("HOME", "/data/data/com.termux/files/home")).expanduser()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def find_tq_root() -> Path:
    """Find TQ root dynamically."""
    search = [
        HOME / "corpus-control-plane" / "mcp" / "turboquant",
        HOME / "tmp_turboquant",
        HOME / "turboquant",
        Path("/data/data/com.termux/files/home/corpus-control-plane/mcp/turboquant"),
        Path("/data/data/com.termux/files/home/tmp_turboquant"),
    ]
    for p in search:
        if p.exists() and (p / "bin" / "turboquant_agent_preflight.py").exists():
            return p
    return HOME / "tmp_turboquant"


def find_corpus_root() -> Path:
    """Find corpus root dynamically."""
    search = [
        HOME / "corpus-control-plane",
        Path("/data/data/com.termux/files/home/corpus-control-plane"),
    ]
    for p in search:
        if p.exists() and (p / "bin" / "corpus_gateway.py").exists():
            return p
    return search[0]


TQ_ROOT = find_tq_root()
CORPUS_ROOT = find_corpus_root()
TQ_BIN = TQ_ROOT / "bin"
TQ_DIST = TQ_ROOT / "dist"


def get_agent_home(agent: str) -> Path:
    cfg = AGENTS.get(agent, AGENTS["generic"])
    return Path(cfg["home"].replace("~", str(HOME))).expanduser()


def run_preflight(agent: str, prompt: str, tq_agent: str) -> dict:
    out = get_agent_home(agent) / ".turboquant-preflight.json"
    script = TQ_BIN / "turboquant_agent_preflight.py"
    if not script.exists():
        return {"error": f"preflight not found: {script}"}
    cmd = ["python3", str(script), "--agent", tq_agent, "--prompt", prompt, "--output", str(out)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=25, cwd=TQ_ROOT)
        if result.stdout:
            return json.loads(result.stdout)
        return {"stderr": result.stderr[:200]}
    except Exception as e:
        return {"error": str(e)}


def run_runtime(agent: str, tq_agent: str) -> dict:
    out = get_agent_home(agent) / ".turboquant-runtime.json"
    script = TQ_BIN / "turboquant_agent_runtime.py"
    if not script.exists():
        return {"error": f"runtime not found: {script}"}
    cmd = ["python3", str(script), "--agent", tq_agent, "--output", str(out)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=TQ_ROOT)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        return {"error": str(e)}


def corpus_search(query: str, top_k: int = 20) -> dict:
    """corpus_web_search — MANDATORY per CLAUDE.md."""
    script = CORPUS_ROOT / "bin" / "corpus_web_search.py"
    if not script.exists():
        return {"error": f"search not found: {script}"}
    cmd = ["python3", str(script), "--query", query, "--top-k", str(top_k)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=CORPUS_ROOT)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        return {"error": str(e)}


def tq_context_pack_search(query: str, top_k: int = 10) -> dict:
    script = TQ_DIST / "tools" / "context_pack_search.js"
    if not script.exists():
        return {"error": "context_pack_search not found"}
    cmd = ["node", str(script), "--query", query, "--top-k", str(top_k)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=TQ_ROOT)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        return {"error": str(e)}


def cache_lint(traces: list) -> dict:
    script = TQ_DIST / "tools" / "prompt_cache_lint.js"
    if not script.exists():
        return {"error": "lint not found"}
    cmd = ["node", str(script), "--blocks-json", json.dumps(traces)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=TQ_ROOT)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        return {"error": str(e)}


def cache_plan(traces: list, budget_mb: int = 512) -> dict:
    script = TQ_DIST / "tools" / "cache_plan.js"
    if not script.exists():
        return {"error": "cache_plan not found"}
    cmd = ["node", str(script), "--blocks-json", json.dumps(traces), "--budget-mb", str(budget_mb)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=TQ_ROOT)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        return {"error": str(e)}


def run_agent(agent: str, prompt: str, args) -> dict:
    cfg = AGENTS.get(agent, AGENTS["generic"])
    tq_agent = cfg["TQ_AGENT"]
    agent_home = get_agent_home(agent)

    # 1. TQ runtime
    runtime = run_runtime(agent, tq_agent)

    # 2. TQ preflight (includes corpus gateway + web_search)
    preflight = run_preflight(agent, prompt, tq_agent)

    # 3. Explicit corpus_web_search (mandatory)
    corpus = corpus_search(prompt, top_k=20)

    # 4. TQ context pack search
    pack_search = tq_context_pack_search(prompt, top_k=10)

    # 5. Cache lint + plan
    traces = [
        {"label": "preflight", "text": json.dumps(preflight)[:500] if preflight else ""},
        {"label": "corpus_results", "text": json.dumps(corpus.get("results", [])[:20])[:500]},
        {"label": "pack_search", "text": json.dumps(pack_search)[:300] if pack_search else ""},
    ]
    lint = cache_lint(traces)
    plan = cache_plan(traces, budget_mb=args.budget_mb)

    # Build packet
    packet = {
        "generated_at": now_iso(),
        "agent": agent,
        "tq_agent": tq_agent,
        "prompt": (prompt or "")[:200],
        "paths": {
            "tq_root": str(TQ_ROOT),
            "corpus_root": str(CORPUS_ROOT),
            "agent_home": str(agent_home),
        },
        "runtime": runtime,
        "preflight": preflight,
        "corpus": {
            "query": (prompt or "")[:100],
            "count": corpus.get("count", 0),
            "top_sources": [r.get("source_type", "?") + ":" + r.get("title", "?")[:40]
                           for r in corpus.get("results", [])[:5]],
        },
        "pack_search": pack_search,
        "cache": {"lint": lint, "plan": plan},
    }

    # Write to agent home
    proxy_out = agent_home / ".turboquant-proxy.json"
    proxy_out.parent.mkdir(parents=True, exist_ok=True)
    proxy_out.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return packet


def cmd_list():
    print("=== SUPPORTED AGENTS ===")
    print(f"{'Agent':<15} {'TQ_AGENT':<15} {'Home':<30} {'Desc'}")
    print("-" * 80)
    for name, cfg in AGENTS.items():
        print(f"{name:<15} {cfg['TQ_AGENT']:<15} {cfg['home']:<30} {cfg['desc']}")


def cmd_audit(args):
    print("=== TQ AGENT PROXY AUDIT ===")
    print(f"TQ_ROOT: {TQ_ROOT}")
    print(f"CORPUS_ROOT: {CORPUS_ROOT}")
    print()

    # Check TQ scripts
    scripts = ["turboquant_agent_preflight.py", "turboquant_agent_runtime.py",
               "turboquant_universal_proxy.py", "turboquant_pipeline.py"]
    print("Scripts:")
    for s in scripts:
        exists = (TQ_BIN / s).exists()
        print(f"  {'✓' if exists else '✗'} {s}")

    # Check TQ tools
    tools = ["context_pack_search.js", "context_pack_build.js", "cache_plan.js",
             "prompt_cache_lint.js", "compress.js"]
    print("\nTools:")
    for t in tools:
        exists = (TQ_DIST / "tools" / t).exists()
        print(f"  {'✓' if exists else '✗'} {t}")

    # Check corpus
    print("\nCorpus:")
    corpus_parts = [
        "bin/corpus_web_search.py",
        "bin/corpus_gateway.py",
        "index/fts.sqlite",
        "artifacts/source_registry.jsonl",
    ]
    for p in corpus_parts:
        exists = (CORPUS_ROOT / p).exists()
        print(f"  {'✓' if exists else '✗'} {p}")

    # Check agent homes
    print("\nAgent Homes:")
    for name, cfg in AGENTS.items():
        home = Path(cfg["home"].replace("~", str(HOME))).expanduser()
        proxy = home / ".turboquant-proxy.json"
        status = "✓ proxy exists" if proxy.exists() else ""
        print(f"  {name:<15} {status}")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Agent Proxy Framework")
    parser.add_argument("agent", nargs="?", help="Agent name (or 'list' or 'audit')")
    parser.add_argument("--prompt", help="Prompt to process")
    parser.add_argument("--argv", nargs=argparse.REMAINDER)
    parser.add_argument("--budget-mb", type=int, default=512)

    args = parser.parse_args()

    if args.agent == "list":
        cmd_list()
        return 0

    if args.agent == "audit":
        cmd_audit(args)
        return 0

    if not args.agent or not args.prompt:
        print("Usage: agent-proxy.py <agent> --prompt '...'")
        print("  agent-proxy.py list  # show all agents")
        print("  agent-proxy.py audit # audit setup")
        return 1

    agent = args.agent.lower()
    if agent not in AGENTS:
        print(f"Unknown agent: {agent}")
        print("Run: agent-proxy.py list")
        return 1

    prompt = args.prompt or " ".join(a for a in (args.argv or []) if not a.startswith("-"))
    result = run_agent(agent, prompt, args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())