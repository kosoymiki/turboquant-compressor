#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
HOME = Path(os.environ.get("HOME", str(Path.home()))).expanduser().resolve()
DRIVER_PACK = ROOT / "mcp" / "turboquant" / "native" / "opencl" / "driver-pack"


def detect_prefix() -> Path:
    env_prefix = os.environ.get("PREFIX")
    if env_prefix:
        return Path(env_prefix).expanduser().resolve()
    python_bin = Path(os.environ.get("_", "")).expanduser()
    if python_bin.name.startswith("python") and python_bin.parent.name == "bin":
        return python_bin.parent.parent.resolve()
    return Path(os.path.dirname(os.path.dirname(os.__file__))).expanduser().resolve()


TERMUX_PREFIX = detect_prefix()


def detect_llvm_root() -> Path | None:
    env_root = os.environ.get("TURBOQUANT_LLVM_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([
        HOME / "llvm-current",
        HOME / "llvm",
        HOME / "toolchains" / "llvm-current",
    ])
    candidates.extend(sorted(HOME.glob("llvm-*"), reverse=True))
    toolchain_dir = HOME / "toolchains"
    if toolchain_dir.exists():
        candidates.extend(sorted(toolchain_dir.glob("llvm-*"), reverse=True))
    for candidate in candidates:
        nested = candidate / candidate.name
        if (candidate / "bin" / "clang").exists():
            return candidate
        if nested != candidate and (nested / "bin" / "clang").exists():
            return nested
    return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def detect_driver_library() -> str | None:
    candidates = [
        DRIVER_PACK / "layer1-compute" / "libRusticlOpenCL.so.1",
        TERMUX_PREFIX / "lib" / "libOpenCL.so",
        Path("/system/vendor/lib64/libOpenCL.so"),
        Path("/vendor/lib64/libOpenCL.so"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def build_env(agent: str) -> tuple[dict, dict]:
    runtime = load_json(ARTIFACTS / "turboquant_runtime_report.json")
    offload = load_json(ARTIFACTS / "gpu_offload_boundary.json")

    production = runtime.get("driver_policy", {})
    backend_probe = runtime.get("backend_probe", {})
    opencl_probe = runtime.get("opencl_probe", {})
    cache_dir = HOME / ".cache" / "turboquant-rusticl"
    cache_dir.mkdir(parents=True, exist_ok=True)

    driver_library = detect_driver_library()
    production_route = production.get("productionRoute") or backend_probe.get("recommendedBackend") or "typescript_cpu"
    gpu_ready = bool(
        production.get("productionReady")
        and production.get("runtimeExecutionVerified")
        and production_route in {"mesa_rusticl_kgsl", "turnip_vulkan_kgsl"}
    )

    llvm_root = detect_llvm_root()
    llvm_bin = llvm_root / "bin" if llvm_root else None
    llvm_lib = llvm_root / "lib" if llvm_root else None
    env = {
        "TURBOQUANT_AGENT_RUNTIME_ACTIVE": "1",
        "TURBOQUANT_AGENT": agent,
        "TURBOQUANT_ACCEL_POLICY": offload.get("policy", "max_gpu_offload_without_false_claims"),
        "TURBOQUANT_GPU_ROUTE": production_route,
        "TURBOQUANT_GPU_READY": "1" if gpu_ready else "0",
        "TURBOQUANT_RUNTIME_EXECUTION_VERIFIED": "1" if production.get("runtimeExecutionVerified") else "0",
        "TURBOQUANT_CORPUS_WEBSEARCH_PRIMARY": "1",
        "TURBOQUANT_PREFLIGHT_REQUIRED": "1",
        "TURBOQUANT_COMPRESSED_MCP_PREFERRED": "1",
        "TURBOQUANT_EXACT_IDENTIFIER_PRESERVE": "1",
        "TURBOQUANT_CONTEXT_INDEX_FIRST": "1",
        "TURBOQUANT_PROVIDER_CACHE_ORCHESTRATION": "1",
        "RUSTICL_CACHE_DIR": str(cache_dir),
        "MESA_SHADER_CACHE_DIR": str(cache_dir),
        "MESA_NO_ERROR": "1",
        "FD_MESA_DEBUG": "",
    }

    if llvm_bin and llvm_bin.exists():
        path_parts = [str(llvm_bin)]
        existing_path = os.environ.get("PATH", "")
        if existing_path:
            path_parts.append(existing_path)
        env["PATH"] = ":".join(path_parts)
        env["CC"] = str(llvm_bin / "clang")
        env["CXX"] = str(llvm_bin / "clang++")
        env["AR"] = str(llvm_bin / "llvm-ar")
        env["LD"] = str(llvm_bin / "ld.lld")
        env["LLVM_ROOT"] = str(llvm_root)
        env["LLVM_BIN"] = str(llvm_bin)
        if llvm_lib and llvm_lib.exists():
            env["LDFLAGS"] = f"-L{llvm_lib}"

    if driver_library:
        env["TURBOQUANT_OPENCL_LIBRARY"] = driver_library
        env["LIBOPENCL_PATH"] = driver_library

    if gpu_ready:
        env["TURBOQUANT_GPU_OFFLOAD_PREFERRED"] = "1"
        env["TURBOQUANT_NUMERIC_BACKEND"] = production_route
    else:
        env["TURBOQUANT_GPU_OFFLOAD_PREFERRED"] = "0"
        env["TURBOQUANT_NUMERIC_BACKEND"] = "disabled_unverified_custom_stack"
        env["TURBOQUANT_GPU_ROUTE"] = "disabled_unverified_custom_stack"

    payload = {
        "generated_at": now_iso(),
        "agent": agent,
        "gpu_ready": gpu_ready,
        "production_route": production_route,
        "driver_library": driver_library,
        "llvm_root": str(llvm_root) if llvm_root and llvm_root.exists() else None,
        "runtime_execution_verified": production.get("runtimeExecutionVerified"),
        "opencl_library_exists": opencl_probe.get("libraryExists"),
        "opencl_loader_state": opencl_probe.get("loaderState"),
        "gpu_eligible_paths": offload.get("gpu_eligible_paths", []),
        "cpu_bound_control_paths": offload.get("cpu_bound_control_paths", []),
        "env": env,
    }
    return env, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="generic")
    parser.add_argument("--output")
    args = parser.parse_args()

    env, payload = build_env(args.agent)
    output_path = Path(args.output) if args.output else ARTIFACTS / f"{args.agent}_runtime_env.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"env": env, "meta": payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
