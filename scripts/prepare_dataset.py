"""Save any Hugging Face dataset locally for offline simulator runs."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Hub dataset ID or local data builder")
    parser.add_argument("--config")
    parser.add_argument("--revision", help="Pin a dataset revision for reproducible runs")
    parser.add_argument("--data-files", type=json.loads)
    parser.add_argument("--data", required=True, type=Path, help="Output directory")
    args = parser.parse_args()
    data = load_dataset(args.dataset, name=args.config, revision=args.revision, data_files=args.data_files)
    data.save_to_disk(str(args.data))
    (args.data / "source.json").write_text(json.dumps({
        "dataset": args.dataset, "config": args.config, "revision": args.revision,
        "fingerprints": {name: split._fingerprint for name, split in data.items()},
        "splits": {name: len(split) for name, split in data.items()},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
