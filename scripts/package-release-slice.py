#!/usr/bin/env python3
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "artifacts" / "export_manifest_release_ready_2026-05-20.json"

data = json.loads(MANIFEST.read_text(encoding="utf-8"))
include = data["include"]

paths = []
for group in ("source_of_truth", "release_contracts", "evidence"):
    paths.extend(include.get(group, []))

version = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["version"]
out = ROOT.parent / f"{ROOT.name}-{version}-release-slice.tar.gz"

with tarfile.open(out, "w:gz") as tar:
    for rel in sorted(dict.fromkeys(paths)):
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing required export path: {rel}")
        tar.add(path, arcname=f"{ROOT.name}/{rel}")

print(out)
