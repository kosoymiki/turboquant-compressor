#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "mcp" / "turboquant" / "dist" / "server.js"


def encode_transport(payload: dict) -> dict:
    script = f"""
import {{ encodeTransportEnvelope }} from './dist/core/transport_codec.js';
console.log(JSON.stringify(encodeTransportEnvelope({json.dumps(payload, ensure_ascii=False)})));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=SERVER.parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def decode_transport(envelope: dict) -> dict:
    script = f"""
import {{ decodeTransportEnvelope }} from './dist/core/transport_codec.js';
console.log(JSON.stringify(decodeTransportEnvelope({json.dumps(envelope, ensure_ascii=False)})));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=SERVER.parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def call_tool(tool: str, payload: dict, response_compressed: bool) -> dict:
    envelope = encode_transport(payload)
    reqs = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "turboquant-mcp-call", "version": "1.0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool,
                "arguments": {
                    "turboquant_transport_envelope_b64": envelope["payload_b64"],
                    "original_bytes": envelope["original_bytes"],
                    "compressed_bytes": envelope["compressed_bytes"],
                    "_turboquant_response_compressed": response_compressed,
                },
            },
        },
    ]
    completed = subprocess.run(
        ["node", str(SERVER)],
        input="\n".join(json.dumps(item, ensure_ascii=False) for item in reqs) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    response = next(item for item in lines if item.get("id") == 2)
    text = response["result"]["content"][0]["text"]
    if response.get("result", {}).get("isError"):
        raise RuntimeError(text)
    payload = json.loads(text)
    if response_compressed and "turboquant_transport_response" in payload:
        return decode_transport(payload["turboquant_transport_response"])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True)
    parser.add_argument("--payload-json", required=True)
    parser.add_argument("--response-compressed", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.payload_json)
    result = call_tool(args.tool, payload, args.response_compressed)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
