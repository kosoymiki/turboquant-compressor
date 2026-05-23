#!/usr/bin/env python3
"""
TurboQuant Pipeline — Universal TQ integration point for ALL corpus operations.
Finds roots dynamically, no hardcoded paths.

Every operation: intake → compress → search → agent → output routes through TQ + corpus_web_search.

Usage:
  python3 turboquant_pipeline.py --action intake --pdf <path>
  python3 turboquant_pipeline.py --action search --query <text>
  python3 turboquant_pipeline.py --action agent --prompt <text> --agent claude_code
  python3 turboquant_pipeline.py --action compress --file <path> --bits 4
  python3 turboquant_pipeline.py --action audit --audit-type full
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def log(msg: str):
    print(f"[TQ] {msg}", file=sys.stderr)


def find_tq_root() -> Path:
    """Find TurboQuant root dynamically."""
    home = Path(os.environ.get("HOME", "/data/data/com.termux/files/home"))
    search = [
        home / "corpus-control-plane" / "mcp" / "turboquant",
        home / "tmp_turboquant",
        home / "turboquant",
        home / "turboquant-compressor",
    ]
    for p in search:
        if p.exists() and (p / "bin" / "turboquant_agent_preflight.py").exists():
            return p
    return search[0]


def find_corpus_root() -> Path:
    """Find corpus-control-plane dynamically."""
    home = Path(os.environ.get("HOME", "/data/data/com.termux/files/home"))
    search = [
        home / "corpus-control-plane",
        Path("/data/data/com.termux/files/home/corpus-control-plane"),
    ]
    for p in search:
        if p.exists() and (p / "bin" / "corpus_gateway.py").exists():
            return p
    env_root = os.environ.get("CORPUS_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p
    return search[0]


# Resolve roots once
TQ_ROOT = find_tq_root()
CORPUS_ROOT = find_corpus_root()
TQ_DIST = TQ_ROOT / "dist"
TQ_BIN = TQ_ROOT / "bin"
ARTIFACTS = CORPUS_ROOT / "artifacts"

AGENTS = ["claude_code", "codex_cli", "gemini_cli"]


def run_node(script: Path, *args, timeout=30) -> dict:
    if not script.exists():
        return {"error": f"Script not found: {script}"}
    cmd = ["node", str(script)] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=TQ_ROOT)
        return json.loads(result.stdout) if result.stdout else {}
    except Exception as e:
        return {"error": str(e)}


def run_py(script: Path, *args, timeout=60) -> dict:
    if not script.exists():
        return {"error": f"Script not found: {script}"}
    cmd = ["python3", str(script)] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=script.parent.parent)
        return {"stdout": result.stdout[:500], "returncode": result.returncode}
    except Exception as e:
        return {"error": str(e)}


def run_corpus_search(query: str, top_k: int = 20) -> dict:
    search_script = CORPUS_ROOT / "bin" / "corpus_web_search.py"
    if not search_script.exists():
        return {"error": f"Search script not found: {search_script}"}
    cmd = ["python3", str(search_script), "--query", query, "--top-k", str(top_k)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=CORPUS_ROOT)
        return json.loads(result.stdout) if result.stdout else {}
    except:
        return {"count": 0, "results": []}


def tq_preflight(agent: str, prompt: str) -> dict:
    out = ARTIFACTS / f".tq-preflight-{agent}.json"
    preflight_script = TQ_BIN / "turboquant_agent_preflight.py"
    if not preflight_script.exists():
        return {"error": f"Preflight script not found: {preflight_script}"}
    cmd = ["python3", str(preflight_script), "--agent", agent, "--prompt", prompt, "--output", str(out)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=25, cwd=TQ_ROOT)
        return json.loads(result.stdout) if result.stdout else {}
    except Exception as e:
        return {"error": str(e)}


def tq_runtime(agent: str) -> dict:
    out = ARTIFACTS / f".tq-runtime-{agent}.json"
    runtime_script = TQ_BIN / "turboquant_agent_runtime.py"
    if not runtime_script.exists():
        return {"error": f"Runtime script not found: {runtime_script}"}
    cmd = ["python3", str(runtime_script), "--agent", agent, "--output", str(out)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=TQ_ROOT)
        return json.loads(result.stdout) if result.stdout else {}
    except Exception as e:
        return {"error": str(e)}


def tq_cache_plan(blocks: list, budget_mb: int = 512) -> dict:
    script = TQ_DIST / "tools" / "cache_plan.js"
    if not script.exists():
        return {"error": "cache_plan not found"}
    cmd = ["node", str(script), "--blocks-json", json.dumps(blocks), "--budget-mb", str(budget_mb)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=TQ_ROOT)
        return json.loads(result.stdout) if result.stdout else {}
    except:
        return {}


def tq_cache_lint(blocks: list) -> dict:
    script = TQ_DIST / "tools" / "prompt_cache_lint.js"
    if not script.exists():
        return {"error": "lint tool not found"}
    cmd = ["node", str(script), "--blocks-json", json.dumps(blocks)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=TQ_ROOT)
        return json.loads(result.stdout) if result.stdout else {}
    except:
        return {}


def tq_context_pack_build(files: list, dims: int = 384, bits: int = 4) -> dict:
    script = TQ_DIST / "tools" / "context_pack_build.js"
    if not script.exists():
        return {"error": "context_pack_build not found"}
    cmd = ["node", str(script), "--files", ",".join(files), "--dimensions", str(dims), "--bits", str(bits)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=TQ_ROOT)
        return json.loads(result.stdout) if result.stdout else {}
    except:
        return {}


def tq_context_pack_search(query: str, top_k: int = 20) -> dict:
    script = TQ_DIST / "tools" / "context_pack_search.js"
    if not script.exists():
        return {"error": "context_pack_search not found"}
    cmd = ["node", str(script), "--query", query, "--top-k", str(top_k)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=TQ_ROOT)
        return json.loads(result.stdout) if result.stdout else {}
    except:
        return {}


def tq_compress(file: Path, bits: int = 4) -> dict:
    script = TQ_DIST / "tools" / "compress.js"
    if not script.exists():
        return {"error": "compress tool not found"}
    cmd = ["node", str(script), "--input", str(file), "--bits", str(bits)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=TQ_ROOT)
        return json.loads(result.stdout) if result.stdout else {}
    except:
        return {}


def action_intake(args) -> dict:
    log(f"Intake: {args.pdf}")
    pdf = Path(args.pdf)
    if not pdf.exists():
        return {"error": f"File not found: {pdf}"}

    # 1. PyMuPDF intake via corpus
    log("  → PyMuPDF extract")
    intake_script = CORPUS_ROOT / "bin" / "corpus_pdf_intake.py"
    intake = run_py(intake_script, "ingest", str(pdf), f"--license={args.license}")

    # 2. Find extracted text
    text_file = CORPUS_ROOT / "extracted" / "text" / f"{pdf.stem}.txt"
    if text_file.exists():
        log(f"  → TQ context pack build ({text_file.name})")
        pack = tq_context_pack_build([str(text_file)], dims=384, bits=args.bits)
        log(f"  → TQ compress ({args.bits} bits)")
        compress = tq_compress(text_file, bits=args.bits)
    else:
        pack = {"error": "text extraction failed"}
        compress = {}

    # 3. Lint trace
    blocks = [
        {"label": "intake", "text": json.dumps(intake)[:500]},
        {"label": "pack", "text": json.dumps(pack)[:500]}
    ]
    lint = tq_cache_lint(blocks)

    return {
        "action": "intake",
        "pdf": str(pdf),
        "text_extracted": text_file.exists(),
        "intake_result": intake,
        "pack": pack,
        "compress": compress,
        "lint": lint,
        "paths": {"tq_root": str(TQ_ROOT), "corpus_root": str(CORPUS_ROOT)}
    }


def action_search(args) -> dict:
    log(f"Search: {args.query[:50]}...")

    # 1. Corpus web_search (MANDATORY per CLAUDE.md)
    log("  → corpus_web_search")
    corpus = run_corpus_search(args.query, top_k=args.top_k)

    # 2. TQ context pack search
    log("  → TQ context_pack_search")
    tq_search = tq_context_pack_search(args.query, top_k=args.top_k)

    # 3. Cache plan
    blocks = [
        {"label": "corpus_results", "text": json.dumps(corpus.get("results", [])[:20])[:1000]},
        {"label": "tq_search", "text": json.dumps(tq_search)[:500]}
    ]
    plan = tq_cache_plan(blocks, budget_mb=args.budget_mb)

    return {
        "action": "search",
        "query": args.query[:100],
        "corpus_count": corpus.get("count", 0),
        "tq_search_count": tq_search.get("count", 0) if isinstance(tq_search, dict) else 0,
        "top_sources": [r.get("title", "?")[:40] for r in corpus.get("results", [])[:5]],
        "cache_plan": plan,
        "paths": {"tq_root": str(TQ_ROOT), "corpus_root": str(CORPUS_ROOT)}
    }


def action_agent(args) -> dict:
    log(f"Agent: {args.agent} — {args.prompt[:50]}...")

    # 1. TQ runtime
    log("  → TQ runtime")
    runtime = tq_runtime(args.agent)

    # 2. TQ preflight (includes corpus gateway + web_search)
    log("  → TQ preflight")
    preflight = tq_preflight(args.agent, args.prompt)

    # 3. Corpus web_search (explicit per CLAUDE.md)
    log("  → corpus_web_search")
    corpus = run_corpus_search(args.prompt, top_k=20)

    # 4. Cache plan + lint
    blocks = [
        {"label": "preflight", "text": json.dumps(preflight)[:500] if preflight else ""},
        {"label": "corpus", "text": json.dumps(corpus.get("results", [])[:10])[:500]}
    ]
    plan = tq_cache_plan(blocks, budget_mb=args.budget_mb)
    lint = tq_cache_lint(blocks)

    return {
        "action": "agent",
        "agent": args.agent,
        "prompt": args.prompt[:100],
        "runtime": runtime,
        "preflight": preflight,
        "corpus_count": corpus.get("count", 0),
        "cache_plan": plan,
        "cache_lint": lint,
        "paths": {"tq_root": str(TQ_ROOT), "corpus_root": str(CORPUS_ROOT)}
    }


def action_compress(args) -> dict:
    log(f"Compress: {args.file} ({args.bits} bits)")
    f = Path(args.file)
    if not f.exists():
        return {"error": f"File not found: {f}"}
    result = tq_compress(f, bits=args.bits)
    return {"action": "compress", "file": str(f), "bits": args.bits, "result": result}


def action_audit(args) -> dict:
    log(f"Audit: {args.audit_type}")

    audit_result = {
        "timestamp": now_iso(),
        "audit_type": args.audit_type,
        "tq_root": str(TQ_ROOT),
        "corpus_root": str(CORPUS_ROOT),
        "tq_tools": {},
        "corpus_index": {},
        "artifacts": {}
    }

    # Check TQ tools
    tools = ["compress", "context_pack_build", "context_pack_search", "cache_plan",
             "prompt_cache_lint", "turboquant_backend_probe", "turboquant_opencl_probe", "turboquant_quantize"]
    for tool in tools:
        script = TQ_DIST / "tools" / f"{tool}.js"
        audit_result["tq_tools"][tool] = script.exists()

    # Check corpus index
    audit_result["corpus_index"]["fts_exists"] = (CORPUS_ROOT / "index" / "fts.sqlite").exists()
    audit_result["corpus_index"]["registry_exists"] = (ARTIFACTS / "source_registry.jsonl").exists()
    audit_result["corpus_index"]["corpus_gateway_exists"] = (CORPUS_ROOT / "bin" / "corpus_gateway.py").exists()

    # Check artifacts
    audit_result["artifacts"]["count"] = len(list(ARTIFACTS.glob("*.json")))

    # TQ bin scripts
    scripts = ["turboquant_agent_preflight.py", "turboquant_agent_runtime.py",
               "turboquant_pipeline.py", "turboquant_enhanced_intake.py"]
    audit_result["bin_scripts"] = {s: (TQ_BIN / s).exists() for s in scripts}

    return audit_result


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Pipeline — Universal TQ on-ramp")
    parser.add_argument("--action", required=True, choices=["intake", "search", "agent", "compress", "audit"])

    # Intake
    parser.add_argument("--pdf")
    parser.add_argument("--license", default="open")
    parser.add_argument("--bits", type=int, default=4)

    # Search
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--budget-mb", type=int, default=512)

    # Agent
    parser.add_argument("--prompt")
    parser.add_argument("--agent", default="claude_code")

    # Compress
    parser.add_argument("--file")

    # Audit
    parser.add_argument("--audit-type", default="full")

    args = parser.parse_args()

    dispatch = {
        "intake": lambda: action_intake(args),
        "search": lambda: action_search(args),
        "agent": lambda: action_agent(args),
        "compress": lambda: action_compress(args),
        "audit": lambda: action_audit(args),
    }

    log(f"TQ_ROOT: {TQ_ROOT}")
    log(f"CORPUS_ROOT: {CORPUS_ROOT}")

    result = dispatch[args.action]()
    result["timestamp"] = now_iso()

    output = ARTIFACTS / f"tq_pipeline_{args.action}_{datetime.now().strftime('%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"Result: {output}")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())