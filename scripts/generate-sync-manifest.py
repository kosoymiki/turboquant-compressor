#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STANDALONE = Path(os.environ.get("TURBOQUANT_ROOT", str(ROOT))).expanduser().resolve()


def detect_mirror_root() -> Path:
    env_root = os.environ.get("TQ_CORPUS_MIRROR_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    for sibling in STANDALONE.parent.iterdir():
        candidate = sibling / "mcp" / "turboquant"
        if candidate == STANDALONE:
            continue
        if (candidate / "package.json").exists() and (candidate / "native" / "opencl").is_dir():
            return candidate.resolve()
    raise SystemExit("Unable to detect corpus mirror root; set TQ_CORPUS_MIRROR_ROOT")


MIRROR = detect_mirror_root()

CRITICAL_FILES = [
    "native/opencl/src/tq_opencl_context.cpp",
    "native/opencl/src/tq_opencl_program.cpp",
    "native/opencl/src/tq_opencl_probe.cpp",
    "native/opencl/src/tq_opencl_cli.cpp",
    "native/opencl/src/tq_opencl_benchmark.cpp",
    "native/opencl/src/tq_opencl_mse_score.cpp",
    "native/opencl/src/tq_opencl_qjl_score.cpp",
    "native/opencl/src/tq_opencl_value_dequant.cpp",
    "native/opencl/src/tq_opencl_fused_attention.cpp",
    "native/opencl/src/tq_opencl_fused_attention_tiled.cpp",
    "native/opencl/src/tq_driver_loader.cpp",
    "native/opencl/include/tq_driver_loader.h",
    "native/opencl/include/tq_opencl.h",
    "native/opencl/driver-pack/build_mesa.sh",
    "native/opencl/driver-pack/pack_driver.sh",
    "native/opencl/driver-pack/manifest.json",
    "scripts/safe-runtime-pack-run.sh",
    "scripts/benchmark-opencl-adreno.mjs",
    "scripts/verify-release.mjs",
    ".gitignore",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = []
all_match = True
for rel in CRITICAL_FILES:
    left = STANDALONE / rel
    right = MIRROR / rel
    left_exists = left.exists()
    right_exists = right.exists()
    left_hash = sha256(left) if left_exists else None
    right_hash = sha256(right) if right_exists else None
    match = left_exists and right_exists and left_hash == right_hash
    all_match = all_match and match
    rows.append({
        "path": rel,
        "standalone_exists": left_exists,
        "mirror_exists": right_exists,
        "standalone_sha256": left_hash,
        "mirror_sha256": right_hash,
        "match": match,
    })

manifest = {
    "format": "turboquant_sync_manifest_v1",
    "standalone_root": str(STANDALONE),
    "mirror_root": str(MIRROR),
    "critical_file_count": len(CRITICAL_FILES),
    "all_match": all_match,
    "files": rows,
}

out = STANDALONE / "artifacts" / "sync_manifest_release_ready_2026-05-20.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(out)
