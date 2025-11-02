#!/usr/bin/env python3
"""
Simple static file server for the Four-in-a-Roll tree viewer.

Usage:
    python serve_tree_viewer.py [PORT] [--host 0.0.0.0]

Defaults to port 8000 and serves files relative to the repository root.
"""

import http.server
import json
import os
import socketserver
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import argparse


def serve(port: int = 8000, host: str = "", directory: Optional[Path] = None) -> None:
    repo_root = directory or Path(__file__).resolve().parent
    handler_class = http.server.SimpleHTTPRequestHandler

    def latest_output_json() -> Optional[str]:
        outputs_dir = repo_root / "outputs"
        if not outputs_dir.exists():
            return None
        json_files = sorted(
            outputs_dir.glob("*.json"),
            key=lambda file_path: file_path.stat().st_mtime,
            reverse=True,
        )
        if not json_files:
            return None
        return f"outputs/{json_files[0].name}"

    class Handler(handler_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(repo_root), **kwargs)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/outputs/latest.json":
                latest_path = latest_output_json()
                if not latest_path:
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error":"no outputs available"}')
                    return
                payload = json.dumps({"path": latest_path}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                # Explicitly instruct browsers not to cache this pointer
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True  # rebind immediately after shutdown

    with ThreadingHTTPServer((host, port), Handler) as httpd:
        host_for_print = host or "localhost"
        print(f"Serving {repo_root} at http://{host_for_print}:{port}/tree_viewer.html")
        print("Press Ctrl+C to stop.")

        # Make Ctrl+C responsive even if no requests are active
        httpd.timeout = 0.5

        try:
            # Loop so KeyboardInterrupt is reliably caught between timeouts
            while True:
                httpd.handle_request()
        except KeyboardInterrupt:
            # Clean shutdown without a traceback
            pass
        finally:
            httpd.server_close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static server for tree viewer")
    parser.add_argument("port", nargs="?", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--host", default=os.environ.get("HOST", ""), help="Host to bind (default: '')")
    args = parser.parse_args()
    serve(port=args.port, host=args.host)
