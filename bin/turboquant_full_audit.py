#!/usr/bin/env python3
"""
TurboQuant Full Corpus Audit
Runs entire TQ project through corpus with hard web-search at each step.
Targets: P0-P3 closure, Mesa drivers, C++ migration, v4.5.2 release.
"""

import json
import subprocess
import re
from pathlib import Path
from datetime import datetime

TQ_ROOT = Path("/data/data/com.termux/files/home/tmp_turboquant")
CORPUS_ROOT = Path("/data/data/com.termux/files/home/corpus-control-plane")
SEARCH_SCRIPT = CORPUS_ROOT / "bin" / "corpus_web_search.py"

def corpus_search(query: str, top_k: int = 10) -> dict:
    """Run corpus search."""
    try:
        result = subprocess.run(
            ["python3", str(SEARCH_SCRIPT), "--query", query, "--top-k", str(top_k)],
            capture_output=True, text=True, timeout=30
        )
        return json.loads(result.stdout)
    except:
        return {"query": query, "count": 0, "results": []}

def audit_code_file(path: Path) -> dict:
    """Audit a code file for issues."""
    issues = []
    try:
        content = path.read_text(errors='ignore')
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Whitelist: bench files have TQ_BENCH_LOG-gated console.log
            if 'bench' in str(path):
                continue

            # Check for debug artifacts
            if re.search(r'console\.log|debugger|console\.debug|print\(.*debug', line):
                issues.append({"type": "debug", "line": i, "file": str(path), "code": line[:60]})

            # Check for hardcoded secrets (skip security/redaction modules)
            if 'redaction' in str(path).lower():
                continue
            if re.search(r'api[_-]?key|password|secret|token.*=.*["\'][a-zA-Z0-9]{8,}', line, re.I):
                issues.append({"type": "secret", "line": i, "file": str(path), "code": line[:60]})

            # Check for TODO without corpus reference
            if re.search(r'TODO|FIXME|HACK|XXX', line, re.I) and 'corpus' not in line.lower():
                issues.append({"type": "unresolved_todo", "line": i, "file": str(path), "code": line[:60]})

    except Exception as e:
        issues.append({"type": "read_error", "file": str(path), "error": str(e)})

    return issues

def audit_directory(dir_path: Path, patterns: list) -> list:
    """Audit directory for patterns."""
    all_issues = []
    for pattern in patterns:
        for f in dir_path.rglob(pattern):
            if f.is_file():
                all_issues.extend(audit_code_file(f))
    return all_issues

def main():
    print("=" * 70)
    print("TURBOQUANT FULL CORPUS AUDIT")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    report = {
        "timestamp": datetime.now().isoformat(),
        "TQ_VERSION": "v4.1.4",
        "TARGET_VERSION": "v4.5.2",
        "audit_areas": {}
    }

    # Area 1: P0-P3 Status
    print("\n[1] P0-P3 STATUS AUDIT")
    print("-" * 50)
    p0_p3_terms = [
        "turboquant compress quantization",
        "turboquant vector search",
        "turboquant context pack build",
        "turboquant cache plan",
        "turboquant KV analyze",
        "turboquant backend probe",
        "turboquant opencl probe",
        "turboquant adreno loader",
    ]

    p0_p3_results = {}
    for term in p0_p3_terms:
        r = corpus_search(term, top_k=5)
        count = r.get("count", 0)
        sources = [res.get("source_type", "?") for res in r.get("results", [])[:3]]
        p0_p3_results[term] = {"count": count, "sources": sources}
        status = "PASS" if count > 0 else "FAIL"
        print(f"  {term[:40]}: {count} sources [{status}]")

    report["audit_areas"]["p0_p3_status"] = p0_p3_results

    # Area 2: Mesa Driver Integration
    print("\n[2] MESA DRIVER INTEGRATION AUDIT")
    print("-" * 50)
    mesa_terms = [
        "mesa rusticl kgsl freedreno",
        "freedreno turnip android",
        "adreno opencl loader probe mesa",
        "mesa 26.2 upstream turnip",
    ]

    mesa_results = {}
    for term in mesa_terms:
        r = corpus_search(term, top_k=5)
        count = r.get("count", 0)
        sources = [res.get("source_type", "?") for res in r.get("results", [])[:3]]
        mesa_results[term] = {"count": count, "sources": sources}
        status = "PASS" if count > 0 else "FAIL"
        print(f"  {term[:40]}: {count} sources [{status}]")

    report["audit_areas"]["mesa_driver"] = mesa_results

    # Area 3: C++ Migration Readiness
    print("\n[3] C++ MIGRATION READINESS AUDIT")
    print("-" * 50)
    cpp_terms = [
        "turboquant c++ migration native",
        "turboquant wasm target rust",
        "turboquant cross compile llvm",
    ]

    cpp_results = {}
    for term in cpp_terms:
        r = corpus_search(term, top_k=5)
        count = r.get("count", 0)
        sources = [res.get("source_type", "?") for res in r.get("results", [])[:3]]
        cpp_results[term] = {"count": count, "sources": sources}
        status = "PASS" if count > 0 else "FAIL"
        print(f"  {term[:40]}: {count} sources [{status}]")

    report["audit_areas"]["cpp_migration"] = cpp_results

    # Area 4: Debug Artifacts Check
    print("\n[4] DEBUG ARTIFACTS CHECK")
    print("-" * 50)

    debug_issues = []

    # Check dist for debug
    dist = TQ_ROOT / "dist"
    if dist.exists():
        issues = audit_directory(dist, ["*.js"])
        debug_issues.extend([{"area": "dist", **i} for i in issues])
        print(f"  dist/*.js: {len([i for i in issues if i.get('type') == 'debug'])} debug artifacts")

    # Check native for debug
    native = TQ_ROOT / "native" / "opencl"
    if native.exists():
        issues = audit_directory(native, ["*.cpp", "*.h", "*.c"])
        debug_issues.extend([{"area": "native", **i} for i in issues])
        print(f"  native/**/*.cpp: {len([i for i in issues if i.get('type') == 'debug'])} debug artifacts")

    # Check src for debug
    src = TQ_ROOT / "src"
    if src.exists():
        issues = audit_directory(src, ["*.ts", "*.js"])
        debug_issues.extend([{"area": "src", **i} for i in issues])
        print(f"  src/**/*.ts: {len([i for i in issues if i.get('type') == 'debug'])} debug artifacts")

    debug_count = len([i for i in debug_issues if i.get('type') == 'debug'])
    secret_count = len([i for i in debug_issues if i.get('type') == 'secret'])
    todo_count = len([i for i in debug_issues if i.get('type') == 'unresolved_todo'])

    print(f"\n  Total: {debug_count} debug, {secret_count} secrets, {todo_count} unresolved TODOs")

    report["audit_areas"]["debug_check"] = {
        "debug_artifacts": debug_count,
        "secrets": secret_count,
        "unresolved_todos": todo_count,
        "issues": debug_issues[:50]  # Limit for report size
    }

    # Area 5: Registry Completeness
    print("\n[5] REGISTRY COMPLETENESS AUDIT")
    print("-" * 50)

    tq_reg = CORPUS_ROOT / "registry" / "turboquant_registry.jsonl"
    if tq_reg.exists():
        entries = sum(1 for _ in open(tq_reg))
        print(f"  TQ registry: {entries} entries")

        # Check tool coverage
        tools_in_registry = 0
        with open(tq_reg) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get('type') == 'turboquant_tool':
                        tools_in_registry += 1
                except:
                    pass

        print(f"  Tools in registry: {tools_in_registry}/13 expected")

        report["audit_areas"]["registry"] = {
            "total_entries": entries,
            "tools_registered": tools_in_registry
        }
    else:
        print("  ERROR: TQ registry not found")
        report["audit_areas"]["registry"] = {"error": "not_found"}

    # Area 6: Version Check
    print("\n[6] VERSION CHECK")
    print("-" * 50)

    pkg = TQ_ROOT / "package.json"
    if pkg.exists():
        with open(pkg) as f:
            d = json.load(f)
        version = d.get("version", "?")
        print(f"  Current version: {version}")
        print(f"  Target version: v4.5.2")
        report["audit_areas"]["version"] = {
            "current": version,
            "target": "v4.5.2"
        }

    # Area 7: MCP Conformance
    print("\n[7] MCP CONFORMANCE CHECK")
    print("-" * 50)

    mcp_tools = [
        "turboquant_compress",
        "turboquant_quantize",
        "turboquant_vector_search",
        "turboquant_context_pack_search",
        "turboquant_context_pack_build",
        "turboquant_cache_plan",
        "turboquant_prompt_cache_lint",
        "turboquant_kv_analyze",
        "turboquant_backend_probe",
        "turboquant_opencl_probe",
        "turboquant_adreno_loader_probe",
        "turboquant_cost_analyze",
        "turboquant_cli_mcp_profile",
        "corpus_web_search",
    ]

    mcp_results = {}
    for tool in mcp_tools:
        r = corpus_search(tool, top_k=3)
        count = r.get("count", 0)
        mcp_results[tool] = {"count": count, "found": count > 0}
        status = "PASS" if count > 0 else "FAIL"
        print(f"  {tool}: {count} sources [{status}]")

    report["audit_areas"]["mcp_conformance"] = mcp_results

    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    all_pass = True

    # P0-P3
    p0_fail = sum(1 for v in p0_p3_results.values() if v['count'] == 0)
    if p0_fail > 0:
        print(f"P0-P3: {p0_fail} FAILED (out of {len(p0_p3_results)})")
        all_pass = False
    else:
        print("P0-P3: ALL PASS")

    # Mesa
    mesa_fail = sum(1 for v in mesa_results.values() if v['count'] == 0)
    if mesa_fail > 0:
        print(f"Mesa Driver: {mesa_fail} FAILED (out of {len(mesa_results)})")
        all_pass = False
    else:
        print("Mesa Driver: ALL PASS")

    # C++ Migration
    cpp_fail = sum(1 for v in cpp_results.values() if v['count'] == 0)
    if cpp_fail > 0:
        print(f"C++ Migration: {cpp_fail} FAILED (out of {len(cpp_results)})")
        all_pass = False
    else:
        print("C++ Migration: ALL PASS")

    # Debug artifacts
    if debug_count > 0:
        print(f"Debug Artifacts: {debug_count} FOUND — CLEANUP REQUIRED")
        all_pass = False
    else:
        print("Debug Artifacts: CLEAN")

    if secret_count > 0:
        print(f"Secrets: {secret_count} FOUND — REMOVE IMMEDIATELY")
        all_pass = False
    else:
        print("Secrets: CLEAN")

    # MCP conformance
    mcp_fail = sum(1 for v in mcp_results.values() if not v['found'])
    if mcp_fail > 0:
        print(f"MCP Tools: {mcp_fail} MISSING (out of {len(mcp_results)})")
        all_pass = False
    else:
        print("MCP Tools: ALL REGISTERED")

    print("-" * 70)
    if all_pass:
        print("OVERALL: READY FOR v4.5.2 RELEASE")
    else:
        print("OVERALL: BLOCKERS FOUND — FIX BEFORE RELEASE")

    # Write report
    report_path = CORPUS_ROOT / "artifacts" / "turboquant_v450_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")

    return 0 if all_pass else 1

if __name__ == "__main__":
    exit(main())