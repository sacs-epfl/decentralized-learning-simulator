"""Read-only, loopback-first dashboard. Only explicitly allowlisted files are served."""
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class LeaderboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, output, **kwargs):
        self.output = Path(output)
        super().__init__(*args, **kwargs)

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        assets = Path(__file__).with_name("leaderboard_assets")
        allowed = {"/": (assets / "index.html", "text/html; charset=utf-8"),
                   "/app.js": (assets / "app.js", "text/javascript; charset=utf-8"),
                   "/leaderboard.json": (self.output / "leaderboard.json", "application/json"),
                   "/leaderboard.csv": (self.output / "leaderboard.csv", "text/csv; charset=utf-8")}
        if route not in allowed:
            self.send_error(404)
            return
        path, content_type = allowed[route]
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(503, "Waiting for benchmark results")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


def create_server(output, host="127.0.0.1", port=8765):
    return ThreadingHTTPServer((host, port), partial(LeaderboardHandler, output=output))
