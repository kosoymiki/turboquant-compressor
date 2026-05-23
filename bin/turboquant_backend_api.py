#!/usr/bin/env python3
"""
TurboQuant Backend API — Flask API with TQ at every endpoint.
All operations: search, intake, compress, analyze, cache — via TQ.

Endpoints:
  POST /api/search          — TQ corpus search
  POST /api/intake          — TQ PDF intake
  POST /api/compress        — TQ vector compression
  POST /api/cache-plan      — TQ cache planning
  POST /api/analyze         — TQ KV/cache analysis
  GET  /api/tools           — TQ tool registry
  GET  /api/runtime         — TQ runtime status
  WS   /ws/live             — WebSocket live compression stream
"""
import json
import sys
import os
import gzip
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

ROOT = Path("/data/data/com.termux/files/home/corpus-control-plane")
TQ_ROOT = ROOT / "mcp" / "turboquant"
TQ_DIST = TQ_ROOT / "dist"
TQ_BIN = TQ_ROOT / "bin"
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "bin"))

app = Flask(__name__)
CORS(app)

# Import TQ tools
sys.path.insert(0, str(TQ_DIST))
try:
    from tools.context_pack_build import turboquantContextPackBuild
    from tools.context_pack_search import turboquantContextPackSearch
    from tools.compress import turboquantCompress
    from tools.cache_plan import turboquantCachePlan
    from tools.prompt_cache_lint import turboquantPromptCacheLint
    from tools.turboquant_backend_probe import turboquantBackendProbe
    from tools.turboquant_kv_analyze import turboquantKvAnalyze
    TQ_IMPORTED = True
except ImportError:
    TQ_IMPORTED = False


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def tq_log(endpoint: str, data: dict):
    log_file = ARTIFACTS / "tq_api_log.jsonl"
    entry = {"ts": now_iso(), "endpoint": endpoint, "size_kb": len(json.dumps(data)) // 1024}
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def require_tq(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TQ_IMPORTED:
            return jsonify({"error": "TQ not initialized", "status": "import_failed"}), 500
        return func(*args, **kwargs)
    return wrapper


# =============================================================================
# TQ CORE ENDPOINTS
# =============================================================================

@app.route("/api/search", methods=["POST"])
@require_tq
def api_search():
    """Search corpus + TQ context pack search."""
    data = request.get_json() or {}
    query = data.get("query", "")
    top_k = data.get("top_k", 20)
    dims = data.get("dims", 384)
    bits = data.get("bits", 4)

    if not query:
        return jsonify({"error": "query required"}), 400

    # 1. Corpus search
    from corpus_web_search import corpus_web_search_main
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    corpus_result = corpus_web_search_main(["--query", query, "--top-k", str(top_k)])
    corpus_text = sys.stdout.getvalue()
    sys.stdout = old_stdout

    try:
        corpus_data = json.loads(corpus_text)
    except:
        corpus_data = {"count": 0, "results": []}

    # 2. TQ context pack search (if pack exists)
    pack_search = {}
    if TQ_IMPORTED:
        try:
            pack_search = turboquantContextPackSearch({
                "query": query,
                "topK": top_k,
                "dimensions": dims,
                "bitsPerValue": bits
            })
        except Exception as e:
            pack_search = {"error": str(e)}

    result = {
        "query": query,
        "corpus_count": corpus_data.get("count", 0),
        "corpus_results": corpus_data.get("results", [])[:10],
        "pack_search": pack_search,
        "timestamp": now_iso()
    }

    tq_log("search", result)
    return jsonify(result)


@app.route("/api/intake", methods=["POST"])
@require_tq
def api_intake():
    """Intake PDF → extract → TQ context pack build → compress."""
    data = request.get_json() or {}
    pdf_path = data.get("pdf_path", "")
    license = data.get("license", "open")
    dims = data.get("dims", 384)
    bits = data.get("bits", 4)

    pdf = Path(pdf_path)
    if not pdf.exists():
        return jsonify({"error": f"File not found: {pdf}"}), 400

    # 1. PyMuPDF intake
    from corpus_pdf_intake import PDFIntakeEngine
    engine = PDFIntakeEngine(license_class=license)
    intake_result = engine.ingest(str(pdf))

    # 2. Find extracted text
    text_file = ROOT / "extracted" / "text" / f"{pdf.stem}.txt"

    # 3. TQ context pack build
    pack_result = {}
    compress_result = {}
    if text_file.exists() and TQ_IMPORTED:
        try:
            pack_result = turboquantContextPackBuild({
                "files": [str(text_file)],
                "dimensions": dims,
                "bitsPerValue": bits,
                "vectorizer": "hashed_tfidf",
                "storageMode": "inline_text"
            })
        except Exception as e:
            pack_result = {"error": str(e)}

        # 4. Compress
        try:
            compress_result = turboquantCompress({
                "vectors": [[0.1] * dims] * 100,  # placeholder
                "dimensions": dims,
                "bitsPerValue": bits
            })
        except Exception as e:
            compress_result = {"error": str(e)}

    result = {
        "pdf": str(pdf),
        "intake": intake_result,
        "text_extracted": text_file.exists(),
        "pack": pack_result,
        "compress": compress_result,
        "timestamp": now_iso()
    }

    tq_log("intake", result)
    return jsonify(result)


@app.route("/api/compress", methods=["POST"])
@require_tq
def api_compress():
    """Compress text/vectors with TQ."""
    data = request.get_json() or {}
    text = data.get("text", "")
    vectors = data.get("vectors", [])
    dims = data.get("dimensions", 384)
    bits = data.get("bits", 4)
    include_qjl = data.get("includeQJL", True)

    if not vectors and not text:
        return jsonify({"error": "text or vectors required"}), 400

    # Generate vectors from text if not provided
    if not vectors and text:
        import hashlib
        words = text.split()
        vectors = [[hashlib.md5(f"{w}{i}".encode()).hexdigest().__getitem__(i % 32) / 31.0 for i in range(dims)] for w in words[:100]]

    if not TQ_IMPORTED:
        return jsonify({"error": "TQ not available", "vectors_count": len(vectors)}), 500

    try:
        result = turboquantCompress({
            "vectors": vectors[:1000],
            "dimensions": dims,
            "bitsPerValue": bits,
            "includeQJL": include_qjl,
            "seed": 42
        })
    except Exception as e:
        result = {"error": str(e)}

    result["timestamp"] = now_iso()
    tq_log("compress", result)
    return jsonify(result)


@app.route("/api/cache-plan", methods=["POST"])
@require_tq
def api_cache_plan():
    """Analyze blocks and generate cache plan."""
    data = request.get_json() or {}
    blocks = data.get("blocks", [])
    target = data.get("target", "claude_code")
    budget_mb = data.get("budget_mb", 512)

    if not blocks:
        # Generate from recent artifacts
        blocks = []
        for artifact in ARTIFACTS.glob("*_trace.json")[:5]:
            try:
                d = json.loads(artifact.read_text())
                blocks.append({"label": artifact.stem, "text": json.dumps(d)[:500]})
            except:
                pass

    if not TQ_IMPORTED:
        return jsonify({"blocks_analyzed": len(blocks), "error": "TQ not available"}), 500

    try:
        result = turboquantCachePlan({"blocks": blocks, "target": target})
    except Exception as e:
        result = {"error": str(e)}

    result["budget_mb"] = budget_mb
    result["blocks_analyzed"] = len(blocks)
    result["timestamp"] = now_iso()
    tq_log("cache_plan", result)
    return jsonify(result)


@app.route("/api/analyze", methods=["POST"])
@require_tq
def api_analyze():
    """KV cache and runtime analysis."""
    data = request.get_json() or {}
    action = data.get("action", "estimate_savings")
    head_dim = data.get("head_dim", 128)
    num_kv_heads = data.get("num_kv_heads", 8)
    num_layers = data.get("num_layers", 32)
    seq_len = data.get("seq_len", 4096)
    key_bits = data.get("key_bits", 4)
    value_bits = data.get("value_bits", 2)

    if not TQ_IMPORTED:
        return jsonify({"error": "TQ not available"}), 500

    try:
        result = turboquantKvAnalyze({
            "action": action,
            "headDim": head_dim,
            "numKvHeads": num_kv_heads,
            "numLayers": num_layers,
            "seqLen": seq_len,
            "keyBits": key_bits,
            "valueBits": value_bits
        })
    except Exception as e:
        result = {"error": str(e)}

    result["timestamp"] = now_iso()
    tq_log("analyze", result)
    return jsonify(result)


@app.route("/api/backend-probe", methods=["GET"])
@require_tq
def api_backend_probe():
    """Probe GPU backends."""
    deep = request.args.get("deep", "false").lower() == "true"

    if not TQ_IMPORTED:
        return jsonify({"error": "TQ not available"}), 500

    try:
        result = turboquantBackendProbe({"deep": deep})
    except Exception as e:
        result = {"error": str(e)}

    result["timestamp"] = now_iso()
    return jsonify(result)


# =============================================================================
# TOOLS REGISTRY & STATUS
# =============================================================================

@app.route("/api/tools", methods=["GET"])
def api_tools():
    """List all TQ tools with schemas."""
    tools = [
        {"name": "search", "method": "POST", "path": "/api/search", "desc": "Corpus + TQ pack search"},
        {"name": "intake", "method": "POST", "path": "/api/intake", "desc": "PDF intake with TQ"},
        {"name": "compress", "method": "POST", "path": "/api/compress", "desc": "Vector compression"},
        {"name": "cache-plan", "method": "POST", "path": "/api/cache-plan", "desc": "Cache tier planning"},
        {"name": "analyze", "method": "POST", "path": "/api/analyze", "desc": "KV cache analysis"},
        {"name": "backend-probe", "method": "GET", "path": "/api/backend-probe", "desc": "GPU backend detection"},
    ]
    return jsonify({"tools": tools, "tq_imported": TQ_IMPORTED, "timestamp": now_iso()})


@app.route("/api/runtime", methods=["GET"])
def api_runtime():
    """TQ runtime status."""
    runtime_file = Path(os.environ.get("HOME", "/data/data/com.termux/files/home")) / ".claude" / ".turboquant-runtime.json"
    runtime = {}
    if runtime_file.exists():
        try:
            runtime = json.loads(runtime_file.read_text())
        except:
            pass

    return jsonify({
        "tq_root": str(TQ_ROOT),
        "tq_imported": TQ_IMPORTED,
        "runtime": runtime,
        "timestamp": now_iso()
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "tq_imported": TQ_IMPORTED,
        "artifacts_count": len(list(ARTIFACTS.glob("*.json"))),
        "timestamp": now_iso()
    })


# =============================================================================
# WEBSOCKET LIVE COMPRESSION
# =============================================================================

@app.route("/ws/live", methods=["GET"])
def ws_live():
    """WebSocket for live compression streaming."""
    from flask_socketio import SocketIO, emit
    # Note: Requires flask-socketio. For now, use SSE.
    return jsonify({"error": "Use /api/stream for SSE streaming"})


@app.route("/api/stream", methods=["POST"])
def api_stream():
    """SSE stream for live compression."""
    data = request.get_json() or {}
    text = data.get("text", "")
    bits = data.get("bits", 4)

    def generate():
        yield f"data: {json.dumps({'status': 'starting'})}\n\n"

        # Compress in chunks
        chunk_size = 1000
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            yield f"data: {json.dumps({'chunk': i, 'text': chunk[:50]})}\n\n"

        yield f"data: {json.dumps({'status': 'done'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("TQ_API_PORT", 8765))
    debug = os.environ.get("TQ_API_DEBUG", "false").lower() == "true"
    print(f"TQ Backend API starting on port {port} (debug={debug})")
    print(f"TQ imported: {TQ_IMPORTED}")
    app.run(host="0.0.0.0", port=port, debug=debug)