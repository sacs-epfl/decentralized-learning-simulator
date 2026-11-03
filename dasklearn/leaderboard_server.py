"""Read-only, loopback-first dashboard. Only explicitly allowlisted files are served."""
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit


def load_settings(output=None, catalog=None):
    """Resolve a fixed allowlist of result directories, including pending settings."""
    if (output is None) == (catalog is None):
        raise ValueError("Specify either an output directory or a settings catalog")
    if catalog is None:
        output = Path(output).resolve()
        return "default", [dict(id="default", label=output.name, output=output)]
    path = Path(catalog).expanduser().resolve()
    document = json.loads(path.read_text())
    if (not isinstance(document, dict) or set(document) != {"version", "default", "settings"}
            or type(document["version"]) is not int or document["version"] != 1):
        raise ValueError("A settings catalog requires version: 1, default, and settings")
    entries, seen = [], set()
    if not isinstance(document["settings"], list) or not document["settings"]:
        raise ValueError("A settings catalog requires nonempty settings")
    for entry in document["settings"]:
        if not isinstance(entry, dict) or set(entry) != {"id", "label", "output"}:
            raise ValueError("Each setting requires id, label, and output")
        identifier = entry["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", identifier) or identifier in seen:
            raise ValueError("Setting IDs must be unique lowercase letters/digits with hyphens/underscores")
        if any(not isinstance(entry[key], str) or not entry[key].strip() for key in ("label", "output")):
            raise ValueError("Setting label/output must be nonempty strings")
        seen.add(identifier)
        entries.append(dict(entry, output=(path.parent / Path(entry["output"]).expanduser()).resolve()))
    if not isinstance(document["default"], str) or document["default"] not in seen:
        raise ValueError("The default setting must exist in settings")
    return document["default"], entries


class LeaderboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, default, settings, **kwargs):
        self.default = default
        self.settings = settings
        super().__init__(*args, **kwargs)

    def do_GET(self):
        url = urlsplit(self.path)
        route = url.path
        assets = Path(__file__).with_name("leaderboard_assets")
        if route == "/settings.json":
            data = json.dumps(dict(version=1, default=self.default,
                settings=[{key: entry[key] for key in ("id", "label")} for entry in self.settings]), ensure_ascii=False).encode()
            self.send_data(data, "application/json")
            return
        allowed = {"/": (assets / "index.html", "text/html; charset=utf-8"),
                   "/app.js": (assets / "app.js", "text/javascript; charset=utf-8")}
        if route in {"/leaderboard.json", "/leaderboard.csv"}:
            identifiers = parse_qs(url.query).get("setting", [self.default])
            entry = next((entry for entry in self.settings if len(identifiers) == 1 and entry["id"] == identifiers[0]), None)
            if entry is None:
                self.send_error(404, "Unknown leaderboard setting")
                return
            allowed[route] = (entry["output"] / route[1:], "application/json" if route.endswith(".json") else "text/csv; charset=utf-8")
        if route not in allowed:
            self.send_error(404)
            return
        path, content_type = allowed[route]
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(503, "Waiting for benchmark results")
            return
        self.send_data(data, content_type)

    def send_data(self, data, content_type):
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


def create_server(output=None, host="127.0.0.1", port=8765, *, catalog=None):
    default, settings = load_settings(output, catalog)
    return ThreadingHTTPServer((host, port), partial(LeaderboardHandler, default=default, settings=settings))
