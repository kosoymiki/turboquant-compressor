#!/usr/bin/env python3
"""
Claude Code TQ Proxy — wrapper script.
Symlinks to agent-proxy.py with agent=claude_code.
"""
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "agent-proxy.py"
sys.path.insert(0, str(SCRIPT.parent))

if __name__ == "__main__":
    import subprocess
    import sys
    sys.argv = [__file__, "claude_code"] + sys.argv[1:]
    sys.exit(subprocess.run(["python3", str(SCRIPT)] + sys.argv[1:]).returncode)