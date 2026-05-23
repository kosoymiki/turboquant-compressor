#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("HOME", str(Path.home()))).expanduser().resolve()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_runtime() -> dict:
    result = subprocess.run(
        ["python3", str(ROOT / "bin" / "turboquant_agent_runtime.py"), "--agent", "claude_code", "--output", str(HOME / ".claude" / ".turboquant-runtime.json")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(result.stdout)


def run_preflight(prompt: str | None) -> dict | None:
    prompt = (prompt or "").strip()
    if not prompt:
        return None
    result = subprocess.run(
        [
            "python3",
            str(ROOT / "bin" / "turboquant_agent_preflight.py"),
            "--agent",
            "claude_code",
            "--prompt",
            prompt,
            "--output",
            str(HOME / ".claude" / ".turboquant-preflight.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt")
    parser.add_argument("--argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    runtime = run_runtime()
    preflight = run_preflight(args.prompt)
    packet = {
        "generated_at": now_iso(),
        "argv": args.argv or [],
        "prompt": args.prompt,
        "runtime": runtime.get("meta", {}),
        "preflight": {
            "available": preflight is not None,
            "path": str(HOME / ".claude" / ".turboquant-preflight.json"),
        },
    }
    out = HOME / ".claude" / ".turboquant-proxy.json"
    out.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(packet, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
