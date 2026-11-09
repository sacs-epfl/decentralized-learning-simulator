"""Read-only, loopback-first dashboard. Only explicitly allowlisted files are served."""
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from dasklearn.leaderboard_reference import centralized_reference, reference_spec
from dasklearn.leaderboard_details import (algorithm_document, algorithm_entry, annotate_leaderboard,
    combined_leaderboard, load_algorithms, pending_csv, pending_leaderboard, public_algorithms)


def load_settings(output=None, catalog=None):
    """Resolve a fixed allowlist of result directories, including pending settings."""
    if (output is None) == (catalog is None):
        raise ValueError("Specify either an output directory or a settings catalog")
    if catalog is None:
        output = Path(output).resolve()
        return "default", [dict(id="default", label=output.name, output=output)]
    path = Path(catalog).expanduser().resolve()
    document = json.loads(path.read_text())
    if (not isinstance(document, dict) or not {"version", "default", "settings"} <= set(document)
            or set(document) - {"version", "default", "settings", "reference", "algorithms", "workloads", "default_workload"}
            or type(document["version"]) is not int or document["version"] != 1):
        raise ValueError("A settings catalog requires version: 1, default, and settings")
    reference_spec(path)  # Validate optional reference before starting the server.
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


def load_workloads(catalog):
    """Resolve trusted catalogs independently; never combine different workloads."""
    if catalog is None:
        return None, None
    path = Path(catalog).resolve()
    document = json.loads(path.read_text())
    if 'workloads' not in document and 'default_workload' not in document:
        return None, None
    declared = document.get('workloads')
    if not isinstance(declared, list) or not declared:
        raise ValueError('workloads requires a nonempty catalog list')
    workloads = {}
    for entry in declared:
        if (not isinstance(entry, dict) or set(entry) != {'id', 'label', 'catalog'}
                or not isinstance(entry['id'], str)
                or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', entry['id'])
                or entry['id'] in workloads
                or any(not isinstance(entry[key], str) or not entry[key].strip() for key in ('label', 'catalog'))):
            raise ValueError('Workloads require unique valid IDs, labels and catalog paths')
        target = (path.parent / entry['catalog']).resolve()
        default, settings = load_settings(catalog=target)
        algorithms = load_algorithms(target)
        if algorithms and set(algorithms['scenarios']) != {item['id'] for item in settings}:
            raise ValueError('Algorithm scenarios must match each workload catalog')
        workloads[entry['id']] = dict(id=entry['id'], label=entry['label'], default=default,
                                     settings=settings, algorithms=algorithms, reference=reference_spec(target))
    default = document.get('default_workload')
    if not isinstance(default, str) or default not in workloads:
        raise ValueError('default_workload must exist in workloads')
    return default, workloads


class LeaderboardHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, default, settings, reference=None, algorithms=None,
                 default_workload=None, workloads=None, **kwargs):
        self.default = default
        self.settings = settings
        self.reference = reference
        self.algorithms = algorithms
        self.default_workload, self.workloads = default_workload, workloads
        super().__init__(*args, **kwargs)

    def do_GET(self):
        url = urlsplit(self.path)
        route = url.path
        assets = Path(__file__).with_name("leaderboard_assets")
        query = parse_qs(url.query)
        if self.workloads:
            requested = query.get('workload', [self.default_workload])
            workload = self.workloads.get(requested[0]) if len(requested) == 1 else None
            if workload is None:
                self.send_error(404, 'Unknown dataset/model combination')
                return
            self.default, self.settings = workload['default'], workload['settings']
            self.algorithms, self.reference = workload['algorithms'], workload['reference']
        elif 'workload' in query:
            self.send_error(404, 'Unknown dataset/model combination')
            return
        if route == "/settings.json":
            document = dict(version=1, default=self.default,
                settings=[{key: entry[key] for key in ("id", "label")} for entry in self.settings])
            if self.algorithms:
                document['algorithms'] = public_algorithms(self.algorithms)
            if self.workloads:
                document.update(default_workload=self.default_workload, workload=workload['id'],
                                workloads=[dict(id=item['id'], label=item['label']) for item in self.workloads.values()])
            data = json.dumps(document, ensure_ascii=False).encode()
            self.send_data(data, "application/json")
            return
        allowed = {"/": (assets / "index.html", "text/html; charset=utf-8"),
                   "/app.js": (assets / "app.js", "text/javascript; charset=utf-8"),
                   "/style.css": (assets / "style.css", "text/css; charset=utf-8"),
                   "/algorithm.js": (assets / "algorithm.js", "text/javascript; charset=utf-8")}
        if route.startswith('/algorithm/') or route in {'/algorithm.json', '/algorithm-config.json', '/implementation'}:
            identifiers = [route.removeprefix('/algorithm/')] if route.startswith('/algorithm/') else query.get('algorithm', [])
            algorithm = algorithm_entry(self.algorithms, identifiers[0]) if len(identifiers) == 1 else None
            settings = query.get('setting', [self.default])
            if algorithm is None or len(settings) != 1 or settings[0] not in self.algorithms['scenarios']:
                self.send_error(404, 'Unknown algorithm or setting')
                return
            if route == '/implementation':
                # Only source files explicitly registered in trusted metadata.
                files = query.get('file', ['0'])
                if len(files) != 1 or files[0] not in {str(i) for i in range(len(algorithm['implementation_files']))}:
                    self.send_error(404)
                    return
                path = Path(__file__).resolve().parents[1] / algorithm['implementation_files'][int(files[0])]
                self.send_data(path.read_bytes(), 'text/plain; charset=utf-8')
                return
            if route in {'/algorithm.json', '/algorithm-config.json'}:
                try:
                    document = algorithm_document(algorithm, self.algorithms, settings[0])
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.send_error(503, f'Cannot read algorithm evidence: {exc}')
                    return
                if route == '/algorithm-config.json':
                    document = document['configuration']
                self.send_data(json.dumps(document, ensure_ascii=False, allow_nan=False).encode(), 'application/json')
                return
            allowed[route] = (assets / 'algorithm.html', 'text/html; charset=utf-8')
        if route in {"/leaderboard.json", "/leaderboard.csv"}:
            identifiers = query.get("setting", [self.default])
            entry = next((entry for entry in self.settings if len(identifiers) == 1 and entry["id"] == identifiers[0]), None)
            if entry is None:
                self.send_error(404, "Unknown leaderboard setting")
                return
            if self.algorithms:
                from dasklearn.benchmark import leaderboard_csv
                try:
                    document = combined_leaderboard(self.algorithms, identifiers[0])
                    if self.reference is not None:
                        reference = centralized_reference(document, self.reference)
                        if reference is not None:
                            document['reference'] = reference
                    data = (json.dumps(document, ensure_ascii=False, allow_nan=False).encode()
                            if route.endswith('.json') else leaderboard_csv(document).encode())
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.send_error(503, f'Cannot read registered comparison: {exc}')
                    return
                self.send_data(data, 'application/json' if route.endswith('.json') else 'text/csv; charset=utf-8')
                return
            allowed[route] = (entry["output"] / route[1:], "application/json" if route.endswith(".json") else "text/csv; charset=utf-8")
        if route not in allowed:
            self.send_error(404)
            return
        path, content_type = allowed[route]
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            if self.algorithms and route in {'/leaderboard.json', '/leaderboard.csv'}:
                document = pending_leaderboard(self.algorithms, identifiers[0])
                data = (json.dumps(annotate_leaderboard(document, self.algorithms), ensure_ascii=False, allow_nan=False).encode()
                        if route.endswith('.json') else pending_csv(document))
                self.send_data(data, content_type)
                return
            self.send_error(503, "Waiting for benchmark results")
            return
        if route in {'/leaderboard.json', '/leaderboard.csv'}:
            from dasklearn.benchmark import checkpoint_budget_view, leaderboard, leaderboard_csv
            # Recompute saved endpoint-era tables from recorded curves without
            # changing experiment files or rerunning training. Preserve pipeline metadata.
            document = json.loads(path.with_name('leaderboard.json').read_text())
            if document.get('runs') and 'tables' in document:
                seeds = sorted({run['seed'] for run in document['runs']})
                refreshed = leaderboard(document | dict(seeds=seeds))
                document.update(tables=refreshed['tables'], checkpoint_policy=refreshed['checkpoint_policy'])
                if document.get('evaluation'):
                    document['evaluation'] = document['evaluation'] | dict(policy=refreshed['checkpoint_policy'])
                manifest_path = path.with_name('manifest.json')
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    document = checkpoint_budget_view(document, 600, times=manifest.get('times', []), seeds=seeds)
                data = (json.dumps(document, ensure_ascii=False, allow_nan=False).encode()
                        if route.endswith('.json') else leaderboard_csv(document).encode())
        if route == '/leaderboard.json' and self.algorithms:
            data = json.dumps(annotate_leaderboard(json.loads(data), self.algorithms), ensure_ascii=False, allow_nan=False).encode()
        if route == "/leaderboard.json" and self.reference is not None:
            document = json.loads(data)
            reference = centralized_reference(document, self.reference)
            if reference is not None:
                document["reference"] = reference
                data = json.dumps(document, ensure_ascii=False, allow_nan=False).encode()
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
    algorithms = load_algorithms(catalog)
    if algorithms and set(algorithms['scenarios']) != {entry['id'] for entry in settings}:
        raise ValueError('Algorithm scenarios must match the settings catalog')
    default_workload, workloads = load_workloads(catalog)
    return ThreadingHTTPServer((host, port), partial(LeaderboardHandler, default=default, settings=settings,
                                                    reference=reference_spec(catalog), algorithms=algorithms,
                                                    default_workload=default_workload, workloads=workloads))
