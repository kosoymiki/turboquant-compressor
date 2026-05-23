#!/usr/bin/env python3
"""
TurboQuant-Enhanced PDF Intake
Wraps corpus_pdf_intake.py with TQ context pack build + compression at every step.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/data/data/com.termux/files/home/corpus-control-plane")
TQ_DIST = ROOT / "mcp" / "turboquant" / "dist"
INTAKE_SCRIPT = ROOT / "bin" / "corpus_pdf_intake.py"
ARTIFACTS = ROOT / "artifacts"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def run_intake(pdf_path: str, license: str) -> dict:
    cmd = ["python3", str(INTAKE_SCRIPT), "ingest", pdf_path, f"--license={license}"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=ROOT)
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except Exception as e:
        return {"error": str(e)}


def build_context_pack(text_file: Path, source_id: str) -> dict:
    if not text_file.exists():
        return {"error": "text file not found"}

    pack_out = ARTIFACTS / f"tq_pack_{source_id}.json"
    cmd = [
        "node", str(TQ_DIST / "tools" / "context_pack_build.js"),
        "--files", str(text_file),
        "--dimensions", "384",
        "--bits", "4",
        "--output", str(pack_out)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=ROOT.parent / "corpus-control-plane" / "mcp" / "turboquant")
        if result.stdout:
            return json.loads(result.stdout)
        return {"pack_path": str(pack_out)}
    except Exception as e:
        return {"error": str(e)}


def compress_pack(pack_file: Path, bits: int = 4) -> dict:
    if not pack_file.exists():
        return {"error": "pack not found"}

    compress_out = pack_file.with_suffix(".compressed.json")
    cmd = [
        "node", str(TQ_DIST / "tools" / "compress.js"),
        "--input", str(pack_file),
        "--bits", str(bits)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return {"compressed": str(compress_out), "raw": result.stdout[:200] if result.stdout else ""}
    except Exception as e:
        return {"error": str(e)}


def lint_intake_trace(intake_result: dict) -> dict:
    blocks = [
        {"label": "intake_result", "text": json.dumps(intake_result)[:1000]}
    ]
    lint_script = TQ_DIST / "tools" / "prompt_cache_lint.js"
    if not lint_script.exists():
        return {"error": "lint tool not found"}
    cmd = ["node", str(lint_script), "--blocks-json", json.dumps(blocks)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return json.loads(result.stdout) if result.stdout else {}
    except:
        return {}


def main():
    parser = argparse.ArgumentParser(description="TurboQuant-Enhanced PDF Intake")
    parser.add_argument("pdf_path")
    parser.add_argument("--license", default="open")
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    print(f"=== TQ-ENHANCED INTAKE: {Path(args.pdf_path).name} ===\n")

    # Step 1: PyMuPDF intake
    print("1. Running PDF intake...")
    intake = run_intake(args.pdf_path, args.license)
    print(f"   returncode: {intake.get('returncode', '?')}")

    # Step 2: Find extracted text
    pdf_stem = Path(args.pdf_path).stem
    text_file = ROOT / "extracted" / "text" / f"{pdf_stem}.txt"
    print(f"   extracted: {text_file.exists()}")

    # Step 3: TQ context pack build
    print("\n2. Building TQ context pack...")
    source_id = Path(args.pdf_path).name.replace(".pdf", "").replace(" ", "_")
    pack = build_context_pack(text_file, source_id)
    print(f"   pack: {pack}")

    # Step 4: Compress pack
    pack_file = ARTIFACTS / f"tq_pack_{source_id}.json"
    if pack_file.exists():
        print(f"\n3. Compressing pack ({args.bits} bits)...")
        compressed = compress_pack(pack_file, args.bits)
        print(f"   result: {compressed}")

    # Step 5: Lint trace
    print("\n4. Linting intake trace...")
    lint = lint_intake_trace(intake)
    print(f"   lint: {lint}")

    # Result summary
    result = {
        "timestamp": now_iso(),
        "pdf": args.pdf_path,
        "license": args.license,
        "intake": intake,
        "text_extracted": text_file.exists(),
        "pack_built": pack_file.exists(),
        "compression_bits": args.bits,
        "lint": lint
    }
    print(f"\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())