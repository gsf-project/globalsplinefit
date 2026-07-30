#!/usr/bin/env python3
"""Dev server for GSF Explorer: http.server + ``Cache-Control: no-store``.

Plain ``python -m http.server`` sends no Cache-Control header, so browsers
heuristically cache the JS modules / Python files and fresh edits look like
they "don't work" until a hard reload. This serves the webapp directory with
caching disabled — a normal reload always picks up the current files.

Usage: python serve.py [port]      (default 8123)
"""

import functools
import http.server
import pathlib
import sys

WEBAPP = pathlib.Path(__file__).resolve().parent


class NoStoreHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    handler = functools.partial(NoStoreHandler, directory=str(WEBAPP))
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as srv:
        print(f"serving {WEBAPP} at http://127.0.0.1:{port}/ (no-store)")
        srv.serve_forever()
