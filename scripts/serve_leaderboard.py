"""Serve live or completed benchmark results locally, without extra dependencies."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from dasklearn.leaderboard_server import create_server
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", help="Single benchmark output directory")
    parser.add_argument("--catalog", type=Path, help="JSON catalog of selectable leaderboard settings")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        server = create_server(args.output, args.host, args.port, catalog=args.catalog)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    with server:
        print(f"Leaderboard: http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
