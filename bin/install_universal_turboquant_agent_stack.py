#!/usr/bin/env python3
import json
import os
from pathlib import Path

HOME = Path(os.environ.get("HOME", str(Path.home()))).expanduser().resolve()
ROOT = Path(os.environ.get("TURBOQUANT_ROOT", str(Path(__file__).resolve().parent.parent))).expanduser().resolve()
TQ_SERVER = ROOT / "dist" / "server.js"
POLICY_DIR = HOME / ".config" / "turboquant"
POLICY_PATH = POLICY_DIR / "AGENT_POLICY.md"
LOCAL_BIN = HOME / ".local" / "bin"
TQ_MCP_WRAPPER = LOCAL_BIN / "turboquant-mcp"


def detect_repo_root_for_wrapper(required_rel: str) -> str:
    return "\n".join(
        [
            'detect_repo_root() {',
            '  if [ -n "${TURBOQUANT_ROOT:-}" ] && [ -f "${TURBOQUANT_ROOT}/' + required_rel + '" ]; then',
            '    printf "%s\\n" "$TURBOQUANT_ROOT"',
            '    return 0',
            '  fi',
            '  SEARCH_BASE="${HOME:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"',
            '  find "$SEARCH_BASE" -maxdepth 4 -type f -path "*/' + required_rel + '" 2>/dev/null | head -n 1 | sed "s#/' + required_rel.replace("/", "\\/") + '$##"',
            '}',
        ]
    )


def detect_prefix() -> Path:
    env_prefix = os.environ.get("PREFIX")
    if env_prefix:
        return Path(env_prefix).expanduser().resolve()
    node_path = Path(os.popen("command -v node 2>/dev/null").read().strip()).expanduser()
    if node_path.name == "node" and node_path.parent.name == "bin":
        return node_path.parent.parent.resolve()
    python_bin = Path(os.environ.get("_", "")).expanduser()
    if python_bin.name.startswith("python") and python_bin.parent.name == "bin":
        return python_bin.parent.parent.resolve()
    return Path(os.path.dirname(os.path.dirname(os.__file__))).expanduser().resolve()


PREFIX = detect_prefix()


def write_policy() -> None:
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_PATH.write_text(
        "\n".join(
            [
                "# TurboQuant Agent Policy",
                "",
                "- Treat the local TurboQuant root as the primary acceleration and retrieval system.",
                "- Prefer TurboQuant retrieval, cache planning, context packing, and compressed MCP paths over raw context stuffing.",
                "- Preserve exact identifiers across compression.",
                "- Never claim provider token savings from local compression unless the provider explicitly supports cached input or prompt caching.",
                "- Treat generic OpenCL, vendor OpenCL fallback, CPU fallback, Triton fallback, or Python GPU fallback as non-production routes.",
                "- Only accept verified custom stack routes such as `mesa_rusticl_kgsl` or `turnip_vulkan_kgsl` as production acceleration.",
                "- Use `TURBOQUANT_LLVM_ROOT` or local LLVM auto-detection when compilation is required.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_turboquant_mcp_wrapper() -> None:
    LOCAL_BIN.mkdir(parents=True, exist_ok=True)
    TQ_MCP_WRAPPER.write_text(
        "\n".join(
            [
                "#!/usr/bin/env sh",
                "set -eu",
                detect_repo_root_for_wrapper("dist/server.js"),
                'ROOT_CANDIDATE="$(detect_repo_root)"',
                'if [ -z "$ROOT_CANDIDATE" ] || [ ! -f "$ROOT_CANDIDATE/dist/server.js" ]; then',
                '  echo "turboquant-mcp: dist/server.js not found" >&2',
                '  exit 1',
                'fi',
                'exec node "$ROOT_CANDIDATE/dist/server.js" "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    TQ_MCP_WRAPPER.chmod(0o755)


def patch_settings_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("mcpServers", {})
    data["mcpServers"]["turboquant-compressor"] = {"command": "sh", "args": ["-lc", "\"$HOME/.local/bin/turboquant-mcp\""]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_wrapper(name: str, agent: str, exec_path: str, packet_dir: str, proxy_name: str) -> None:
    wrapper = HOME / ".local" / "bin" / name
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "\n".join(
            [
                "#!/usr/bin/env sh",
                "set -eu",
                'PREFIX_DIR="${PREFIX:-$(CDPATH= cd -- "$(dirname -- "$(command -v node)")/.." && pwd)}"',
                detect_repo_root_for_wrapper("bin/" + proxy_name),
                'TQ_ROOT="$(detect_repo_root)"',
                'HOME_DIR="${HOME:-$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)}"',
                f'PACKET_ROOT="$HOME_DIR/{packet_dir}"',
                'mkdir -p "$PACKET_ROOT"',
                'python3 "$TQ_ROOT/bin/' + proxy_name + '" --prompt "$(printf "%s " "$@" | sed \'s/[[:space:]]*$//\')" --argv "$@" >/dev/null 2>&1 || true',
                exec_path + ' "$@"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)


def main() -> int:
    write_policy()
    write_turboquant_mcp_wrapper()
    patch_settings_json(HOME / ".gemini" / "settings.json")
    write_wrapper("gemini-turboquant", "gemini_cli", '"$PREFIX_DIR/bin/gemini"', ".gemini", "turboquant_gemini_proxy.py")
    write_wrapper("aider-turboquant", "aider", '"$PREFIX_DIR/bin/aider"', ".aider", "turboquant_generic_agent_proxy.py")
    write_wrapper("crush-turboquant", "crush", '"$PREFIX_DIR/bin/crush"', ".config/crush", "turboquant_generic_agent_proxy.py")
    write_wrapper("opencode-turboquant", "opencode", '"$PREFIX_DIR/bin/opencode"', ".opencode", "turboquant_generic_agent_proxy.py")
    print(json.dumps({"ok": True, "root": str(ROOT), "server": str(TQ_SERVER)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
