#!/usr/bin/env python3
"""
TurboQuant Universal Agent Proxy — All agents route through TQ web_search.
Finds corpus dynamically, no hardcoded paths.

Usage: python turboquant_universal_proxy.py <agent> --prompt "..."
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HOME = Path(os.environ.get("HOME", str(Path.home()))).expanduser()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def find_corpus_root() -> Path:
    """Find corpus-control-plane dynamically."""
    search = [
        HOME / "corpus-control-plane",
        Path("/data/data/com.termux/files/home/corpus-control-plane"),
    ]
    for p in search:
        if p.exists() and (p / "bin" / "corpus_gateway.py").exists():
            return p
    # Fallback: env var
    env_root = os.environ.get("CORPUS_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p
    return search[0]


def find_tq_root() -> Path:
    """Find TurboQuant root dynamically."""
    search = [
        HOME / "corpus-control-plane" / "mcp" / "turboquant",
        HOME / "tmp_turboquant",
        HOME / "turboquant",
    ]
    for p in search:
        if p.exists() and (p / "bin" / "turboquant_agent_preflight.py").exists():
            return p
    return search[0]


def get_agent_home(agent: str) -> Path:
    mapping = {
        "claude_code": HOME / ".claude",
        "codex_cli": HOME / ".codex",
        "gemini_cli": HOME / ".gemini",
    }
    return mapping.get(agent, HOME / f".{agent}")


def run_preflight(agent: str, prompt: str | None, tq_root: Path, agent_home: Path) -> dict | None:
    if not prompt:
        return None
    out = agent_home / ".turboquant-preflight.json"
    preflight_script = tq_root / "bin" / "turboquant_agent_preflight.py"
    if not preflight_script.exists():
        return {"error": f"preflight script not found: {preflight_script}"}
    cmd = ["python3", str(preflight_script), "--agent", agent, "--prompt", prompt, "--output", str(out)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=25, cwd=tq_root)
        if result.stdout:
            return json.loads(result.stdout)
        return {"stderr": result.stderr[:200]}
    except Exception as e:
        return {"error": str(e)}


def run_runtime(agent: str, tq_root: Path, agent_home: Path) -> dict:
    out = agent_home / ".turboquant-runtime.json"
    runtime_script = tq_root / "bin" / "turboquant_agent_runtime.py"
    if not runtime_script.exists():
        return {"error": f"runtime script not found: {runtime_script}"}
    cmd = ["python3", str(runtime_script), "--agent", agent, "--output", str(out)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=tq_root)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        return {"error": str(e)}


def corpus_search(query: str, top_k: int = 20) -> dict:
    """Search corpus via web_search CLI."""
    corpus_root = find_corpus_root()
    search_script = corpus_root / "bin" / "corpus_web_search.py"
    if not search_script.exists():
        return {"error": f"search script not found: {search_script}"}
    cmd = ["python3", str(search_script), "--query", query, "--top-k", str(top_k)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=corpus_root)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        return {"error": str(e)}


def cache_lint(traces: list, tq_root: Path) -> dict:
    """Lint traces for cache-busting patterns."""
    lint_script = tq_root / "dist" / "tools" / "prompt_cache_lint.js"
    if not lint_script.exists():
        return {"error": "lint tool not found"}
    cmd = ["node", str(lint_script), "--blocks-json", json.dumps(traces)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=tq_root)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        return {"error": str(e)}


def cache_plan(traces: list, tq_root: Path, budget_mb: int = 512) -> dict:
    """Generate cache plan."""
    plan_script = tq_root / "dist" / "tools" / "cache_plan.js"
    if not plan_script.exists():
        return {"error": "cache_plan tool not found"}
    cmd = ["node", str(plan_script), "--blocks-json", json.dumps(traces), "--budget-mb", str(budget_mb)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=tq_root)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        return {"error": str(e)}


def context_pack_search(query: str, tq_root: Path, top_k: int = 10) -> dict:
    """TQ compressed context pack search."""
    search_script = tq_root / "dist" / "tools" / "context_pack_search.js"
    if not search_script.exists():
        return {"error": "context_pack_search not found"}
    cmd = ["node", str(search_script), "--query", query, "--top-k", str(top_k)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=tq_root)
        if result.stdout:
            return json.loads(result.stdout)
        return {}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Universal Proxy")
    parser.add_argument("agent", help="Agent: claude_code, codex_cli, gemini_cli, generic")
    parser.add_argument("--prompt", help="Prompt to classify")
    parser.add_argument("--argv", nargs=argparse.REMAINDER)
    parser.add_argument("--cache-budget-mb", type=int, default=512)
    args = parser.parse_args()

    agent = args.agent
    prompt = args.prompt or " ".join(a for a in (args.argv or []) if not a.startswith("-"))

    # Find roots dynamically
    tq_root = find_tq_root()
    corpus_root = find_corpus_root()
    agent_home = get_agent_home(agent)

    # 1. TQ runtime state
    runtime = run_runtime(agent, tq_root, agent_home)

    # 2. TQ preflight (includes corpus gateway, web_search, cache_plan)
    preflight = run_preflight(agent, prompt, tq_root, agent_home)

    # 3. Corpus web_search (mandatory per CLAUDE.md)
    corpus = corpus_search(prompt, top_k=20)

    # 4. TQ context pack search
    pack_search = context_pack_search(prompt, tq_root, top_k=10)

    # 5. Cache lint + plan
    traces = [
        {"label": "preflight", "text": json.dumps(preflight)[:500] if preflight else ""},
        {"label": "corpus_results", "text": json.dumps(corpus.get("results", [])[:20])[:500]},
    ]
    lint = cache_lint(traces, tq_root)
    plan = cache_plan(traces, tq_root, budget_mb=args.cache_budget_mb)

    # Build proxy packet
    packet = {
        "generated_at": now_iso(),
        "agent": agent,
        "prompt": (prompt or "")[:200],
        "paths": {
            "tq_root": str(tq_root),
            "corpus_root": str(corpus_root),
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
        "cache": {
            "lint": lint,
            "plan": plan,
        }
    }

    # Write to agent home
    proxy_out = agent_home / ".turboquant-proxy.json"
    proxy_out.parent.mkdir(parents=True, exist_ok=True)
    proxy_out.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(packet, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())