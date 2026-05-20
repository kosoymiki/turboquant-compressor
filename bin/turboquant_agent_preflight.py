#!/usr/bin/env python3
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
FORENSICS = ROOT / "forensics"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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

    runtime = load_json(ARTIFACTS / "turboquant_runtime_report.json")
    if not runtime:
        runtime = load_json(FORENSICS / "RELEASE_EVIDENCE_MANIFEST.json")
    runtime_env = load_runtime_env(args.agent)
    driver_policy = runtime.get("driver_policy", {})
    backend_probe = runtime.get("backend_probe", {})
    opencl_probe = runtime.get("opencl_probe", {})

    payload = {
        "generated_at": now_iso(),
        "agent": args.agent,
        "prompt": args.prompt,
        "transcript_path": args.transcript_path,
        "route": {
            "task_type": "standalone_turboquant_preflight",
            "risk": "medium",
            "needs_web_corroboration": False,
            "selected_skills": ["turboquant_runtime", "compressed_mcp", "runtime_policy"],
        },
        "runtime": {
            "production_route": driver_policy.get("productionRoute") or backend_probe.get("recommendedBackend"),
            "runtime_execution_verified": driver_policy.get("runtimeExecutionVerified"),
            "gpu_ready": runtime_env.get("gpu_ready"),
            "driver_library": runtime_env.get("driver_library"),
            "opencl_loader_state": opencl_probe.get("loaderState"),
        },
        "agent_runtime_env": runtime_env.get("env", {}),
        "additional_context": (
            "TURBOQUANT STANDALONE PREFLIGHT ACTIVE. "
            "Use TurboQuant runtime policy, compressed MCP transport, and exact-identifier preservation. "
            "Do not claim provider-side token savings from local compression unless upstream supports cached input or prompt caching."
        ),
    }
    output_path = Path(args.output) if args.output else ARTIFACTS / "agent_preflight.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
