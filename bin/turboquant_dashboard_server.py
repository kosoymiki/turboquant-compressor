#!/usr/bin/env python3
"""
TQ Frontend Server — Serves dashboard + proxies to TQ pipeline.
Single entry point for browser dashboard.
"""
import http.server
import json
import os
import socketserver
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
API_SCRIPT = ROOT / "bin" / "turboquant_backend_api.py"

PORT = int(os.environ.get("TQ_DASHBOARD_PORT", 8080))
API_PORT = 8765


class TQDashHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND), **kwargs)

    def do_POST(self):
        """Proxy POST to TQ API."""
        import urllib.request

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length else b''

        api_url = f"http://localhost:{API_PORT}/api{urlparse(self.path).path}"
        req = urllib.request.Request(api_url, data=body, method='POST',
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', len(data))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.URLError as e:
            error = json.dumps({"error": str(e)}).encode()
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(error)

    def do_GET(self):
        """Proxy GET to TQ API."""
        import urllib.request

        api_path = urlparse(self.path).path
        if api_path.startswith('/api'):
            api_url = f"http://localhost:{API_PORT}{api_path}"
            if self.path.strip():
                parsed = urlparse(self.path)
                if parsed.query:
                    api_url += '?' + parsed.query

            req = urllib.request.Request(api_url, method='GET')
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                    self.send_response(resp.status)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', len(data))
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(data)
            except urllib.error.URLError as e:
                error = json.dumps({"error": str(e)}).encode()
                self.send_response(502)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(error)
        else:
            super().do_GET()

    def log_message(self, fmt, *args):
        print(f"[TQ-DASH] {args[0]}", file=__import__('sys').stderr)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True


def main():
    print(f"TQ Dashboard starting on http://localhost:{PORT}")
    print(f"API proxy to http://localhost:{API_PORT}")
    print(f"Frontend: {FRONTEND}")

    with ThreadedHTTPServer(("", PORT), TQDashHandler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()