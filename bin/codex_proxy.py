#!/usr/bin/env python3
"""Codex CLI TQ Proxy."""
import subprocess, sys
from pathlib import Path
SCRIPT = Path(__file__).resolve().parent / "agent-proxy.py"
sys.argv = [__file__, "codex_cli"] + sys.argv[1:]
sys.exit(subprocess.run(["python3", str(SCRIPT)] + sys.argv[1:]).returncode)
