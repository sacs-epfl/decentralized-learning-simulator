"""Reproduce the fixed 2x2 V2 scheduling/placement ablation against Moshpit.

Run once per network profile, on separate GPUs if available. Each case delegates
to the resumable CIFAR runner and preserves its source/configuration manifest.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


CASES = {
    "reference": ("conflux-v2", 8, False),
    "immediate": ("conflux-v2", 8, True),
    "wide": ("conflux-v2", 16, False),
    "immediate-wide": ("conflux-v2", 16, True),
    "moshpit": ("moshpit", 8, False),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-base-path", required=True)
    parser.add_argument("--profile", choices=["uniform", "heterogeneous"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    folder = args.output.resolve() / args.profile
    folder.mkdir(parents=True, exist_ok=True)
    commands = []
    for case in args.cases:
        variant, chunks, eager = CASES[case]
        commands.append([sys.executable, str(Path(__file__).with_name("run_cifar.py")),
            "--output", str(folder / case), "--dataset-base-path", args.dataset_base_path,
            "--variants", variant, "--profiles", args.profile,
            "--partitioners", "iid", "dirichlet", "--seeds", *map(str, args.seeds),
            "--peers", "64", "--sample-size", "16", "--chunks", str(chunks),
            "--relay-count", str(chunks), "--group-size", "8", "--averaging-steps", "1",
            "--duration", "900", "--test-period", "150", "--bandwidth", "100000",
            "--step-ms", "20", "--latency-ms", "20", "--learning-rate", ".01",
            "--local-steps", "5", "--batch-size", "32", "--alpha", ".1",
            "--momentum", "0", "--weight-decay", "0", "--model-bytes", "346251",
            "--evaluation-split", "test", "--torch-threads", "4", "--device", args.device,
            "--immediate-uploads" if eager else "--no-immediate-uploads"])
    dispatch = {"profile": args.profile, "cases": args.cases, "seeds": args.seeds,
                "commands": commands, "preset": "peer-baselines64 common configuration",
                "hypotheses": ["Earlier uploads preserve every per-round aggregate.",
                    "Sixteen reducers reduce contention relative to eight reducers."],
                "readout": "All-node checkpoints; fresh V2 aggregate separately; simulated time separate from execution time.",
                "selection": "Fixed factorial comparison; report all cases, no test-set hyperparameter selection."}
    dispatch_path = folder / "dispatch.json"
    if dispatch_path.exists() and json.loads(dispatch_path.read_text()) != dispatch:
        raise ValueError("Existing dispatch differs; use a new output directory")
    dispatch_path.write_text(json.dumps(dispatch, indent=2) + "\n")
    for case, command in zip(args.cases, commands):
        print(json.dumps({"case": case, "profile": args.profile, "command": command}), flush=True)
        subprocess.run(command, check=True)
    (folder / "COMPLETE").write_text("All requested ablation cases completed\n")


if __name__ == "__main__":
    main()
