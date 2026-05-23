#!/usr/bin/env python3
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_gateway(prompt: str) -> None:
    subprocess.run(
        ["python3", "bin/corpus_gateway.py", prompt],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def compact(items: list[str], limit: int = 4) -> list[str]:
    return [item for item in items[:limit] if item]


def build_additional_context(prompt: str, route: dict, retrieval: dict, corroboration: dict, budget: dict) -> str:
    selected_skills = ", ".join(route.get("selected_skills", [])[:6]) or "n/a"
    selected_sources = ", ".join(retrieval.get("selected_sources", [])[:6]) or "n/a"
    families = ", ".join(corroboration.get("independent_source_families", [])[:4]) or "n/a"
    index_used = retrieval.get("turboquant_registry_index", {}).get("used", False)
    est_reduction = budget.get("estimated_reduction_vs_raw")
    return (
        "TURBOQUANT PREFLIGHT ACTIVE. "
        f"Route={route.get('task_type')} risk={route.get('risk')} needs_web={route.get('needs_web_corroboration')}. "
        f"Skills={selected_skills}. "
        f"RegistryIndexUsed={index_used}. "
        f"SelectedSources={selected_sources}. "
        f"IndependentFamilies={families}. "
        f"Exact identifiers from prompt must survive all compression. "
        f"Estimated local context reduction={est_reduction}%."
    )


def load_runtime_env(agent: str) -> dict:
    runtime_script = ROOT / "bin" / "turboquant_agent_runtime.py"
    result = subprocess.run(
        ["python3", str(runtime_script), "--agent", agent],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return payload.get("meta", {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="generic")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--transcript-path")
    parser.add_argument("--output")
    args = parser.parse_args()

    run_gateway(args.prompt)

    route = load_json(ARTIFACTS / "route_trace.json")
    envelope = load_json(ARTIFACTS / "request_envelope.json")
    retrieval = load_json(ARTIFACTS / "retrieval_trace.json")
    corroboration = load_json(ARTIFACTS / "corroboration_trace.json")
    runtime = load_json(ARTIFACTS / "turboquant_runtime_report.json")
    budget = load_json(ARTIFACTS / "context_budget_report.json")
    cache_plan = load_json(ARTIFACTS / "cache_plan.json")
    runtime_env = load_runtime_env(args.agent)

    payload = {
        "generated_at": now_iso(),
        "agent": args.agent,
        "prompt": args.prompt,
        "transcript_path": args.transcript_path,
        "route": {
            "task_type": route.get("task_type"),
            "risk": route.get("risk"),
            "needs_web_corroboration": route.get("needs_web_corroboration"),
            "selected_skills": compact(route.get("selected_skills", []), 8),
        },
        "request_envelope": {
            "input_mode": envelope.get("input_mode"),
            "exact_identifiers": compact(envelope.get("exact_identifiers", []), 16),
            "compressed_request_ru": envelope.get("compressed_request_ru"),
        },
        "retrieval": {
            "selected_sources": compact(retrieval.get("selected_sources", []), 8),
            "official_seed_available": retrieval.get("official_seed_available"),
            "registry_index": retrieval.get("turboquant_registry_index"),
        },
        "corroboration": {
            "status": corroboration.get("status"),
            "independent_source_families": compact(corroboration.get("independent_source_families", []), 8),
            "live_url_checks": corroboration.get("live_url_checks", []),
        },
        "runtime": {
            "production_route": runtime.get("driver_policy", {}).get("productionRoute"),
            "production_ready": runtime.get("driver_policy", {}).get("productionReady"),
            "blockers": runtime.get("driver_policy", {}).get("blockers", []),
            "loader_state": runtime.get("opencl_probe", {}).get("loaderState"),
            "recommended_backend": runtime.get("opencl_probe", {}).get("recommendedBackend"),
            "gpu_ready": runtime_env.get("gpu_ready"),
            "driver_library": runtime_env.get("driver_library"),
        },
        "context_budget": {
            "raw_total_bytes": budget.get("raw_total_bytes"),
            "cache_kept_bytes": budget.get("cache_kept_bytes"),
            "turboquant_pack_b64_bytes": budget.get("turboquant_pack_b64_bytes"),
            "estimated_reduction_vs_raw": budget.get("estimated_reduction_vs_raw"),
        },
        "agent_runtime_env": runtime_env.get("env", {}),
        "cache_plan": cache_plan,
        "additional_context": build_additional_context(args.prompt, route, retrieval, corroboration, budget),
    }

    output_path = Path(args.output) if args.output else ARTIFACTS / "agent_preflight.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
