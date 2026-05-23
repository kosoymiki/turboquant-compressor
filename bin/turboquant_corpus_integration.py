#!/usr/bin/env python3
"""
TurboQuant Full Corpus Integration
Runs TQ through corpus with web-search at each step.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path("/data/data/com.termux/files/home")
CORPUS_ROOT = ROOT / "corpus-control-plane"
SEARCH_SCRIPT = CORPUS_ROOT / "bin" / "corpus_web_search.py"
TQ_ROOT = ROOT / "tmp_turboquant"

def corpus_search(query: str, top_k: int = 10) -> dict:
    """Run corpus search and return parsed results."""
    try:
        result = subprocess.run(
            ["python3", str(SEARCH_SCRIPT), "--query", query, "--top-k", str(top_k)],
            capture_output=True, text=True, timeout=30
        )
        return json.loads(result.stdout)
    except:
        return {"query": query, "count": 0, "results": []}

def main():
    print("=== TURBOQUANT CORPUS INTEGRATION ===\n")

    # Areas to validate
    areas = [
        # Core TQ concepts
        ("turboquant vector compression", 10),
        ("turboquant codebook quantization", 10),
        ("turboquant context pack", 10),
        ("turboquant cache planning", 10),
        ("turboquant KV analysis", 10),

        # Driver/Runtime
        ("rusticl kgsl mesa opencl", 10),
        ("freedreno turnip android", 10),
        ("adreno opencl loader probe", 10),

        # Integration
        ("turboquant claude codex gemini agent", 10),
        ("turboquant corpus control plane", 10),
        ("turboquant preflight runtime", 10),

        # Performance
        ("turboquant benchmark performance", 10),
        ("turboquant vector search", 10),
    ]

    results = {}
    for query, top_k in areas:
        print(f"Searching: {query[:50]}...")
        r = corpus_search(query, top_k)
        results[query] = {
            "count": r.get("count", 0),
            "top_sources": [res.get("source_type", "?") + ":" + res.get("title", "?")[:30] for res in r.get("results", [])[:3]]
        }
        print(f"  → {r.get('count', 0)} results, top: {results[query]['top_sources'][0] if results[query]['top_sources'] else 'none'}")

    # Summary
    print("\n=== SUMMARY ===")
    for q, r in results.items():
        print(f"{q[:40]}: {r['count']} sources")

    # Write integration report
    report_path = CORPUS_ROOT / "artifacts" / "turboquant_corpus_integration_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": subprocess.run(["date"], capture_output=True, text=True).stdout.strip(),
            "TQ_ROOT": str(TQ_ROOT),
            "TQ_VERSION": "4.1.4",
            "results": results
        }, f, indent=2)
    print(f"\nReport: {report_path}")

if __name__ == "__main__":
    main()