"""Run the local TARC experiment viewer using Python's standard library."""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import webbrowser
from urllib.parse import urlsplit


class DemoHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if urlsplit(self.path).path in {"/api/local-texts", "/local-texts.json"}:
            body = json.dumps(self.server.local_texts, ensure_ascii=False).encode(
                "utf-8"
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--local-texts",
        type=Path,
        help="Optional owner-local transcript JSON, outside the uploadable repository",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "demo"
    if not (root / "data.json").is_file():
        parser.error(
            "demo/data.json is missing. Extract the complete release archive first."
        )
    local_texts = {"datasets": {}}
    if args.local_texts:
        local_texts = json.loads(args.local_texts.read_text(encoding="utf-8"))
        if not isinstance(local_texts.get("datasets"), dict):
            parser.error("--local-texts must contain a datasets mapping")
    server = ThreadingHTTPServer(
        (args.host, args.port), partial(DemoHandler, directory=str(root))
    )
    server.local_texts = local_texts
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"TARC local demo: {url}", flush=True)
    print(
        "Saved final-model validation outputs. No GPU or model installation required.",
        flush=True,
    )
    print("Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
