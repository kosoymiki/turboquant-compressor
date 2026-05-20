#!/usr/bin/env python3
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def nonflag_prompt(argv: list[str]) -> str | None:
    tokens = [arg for arg in argv if arg and not arg.startswith("-")]
    return " ".join(tokens).strip() or None


def run_runtime(agent: str, runtime_out: Path) -> dict:
    result = subprocess.run(
        ["python3", str(ROOT / "bin" / "turboquant_agent_runtime.py"), "--agent", agent, "--output", str(runtime_out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(result.stdout)


def run_preflight(agent: str, prompt: str | None, preflight_out: Path, timeout_seconds: int = 25) -> dict | None:
    prompt = (prompt or "").strip()
    if not prompt:
        return None
    result = subprocess.run(
        [
            "python3",
            str(ROOT / "bin" / "turboquant_agent_preflight.py"),
            "--agent",
            agent,
            "--prompt",
            prompt,
            "--output",
            str(preflight_out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    argv = args.argv or []
    prompt = args.prompt or nonflag_prompt(argv)
    packet_root = Path(args.packet_root)
    packet_root.mkdir(parents=True, exist_ok=True)
    runtime_out = packet_root / ".turboquant-runtime.json"
    preflight_out = packet_root / ".turboquant-preflight.json"
    proxy_out = packet_root / ".turboquant-proxy.json"

    runtime = run_runtime(args.agent, runtime_out)
    preflight = run_preflight(args.agent, prompt, preflight_out)
    packet = {
        "generated_at": now_iso(),
        "agent": args.agent,
        "argv": argv,
        "prompt": prompt,
        "runtime": runtime.get("meta", {}),
        "preflight": {"available": preflight is not None, "path": str(preflight_out)},
        "paths": {"runtime": str(runtime_out), "proxy": str(proxy_out)},
    }
    proxy_out.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(packet, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
