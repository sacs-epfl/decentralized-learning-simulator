"""Summarize all seeds and retain unreached targets explicitly."""
import argparse
import json
from pathlib import Path
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", type=Path, required=True)
args = p.parse_args()
rows = pd.read_csv(args.input / "summary.csv")
curves = pd.read_csv(args.input / "curves.csv")
summary = []
for (profile, partitioner, variant), frame in rows.groupby(["profile", "partitioner", "variant"]):
    record = {"profile": profile, "partitioner": partitioner, "variant": variant,
              "seeds": frame.seed.tolist(), "accuracy_mean": float(frame.final_accuracy.mean()),
              "accuracy_std": float(frame.final_accuracy.std()), "final_accuracies": frame.final_accuracy.tolist(),
              "bytes_mean": float(frame.bytes_sent.mean()), "training_tasks_mean": float(frame.training_tasks.mean()),
              "actual_execution_s_mean": float(frame.actual_execution_s.mean())}
    for target in [30, 40, 50, 60, 70]:
        times = {}
        for seed in frame.seed:
            data = curves[(curves.profile == profile) & (curves.partitioner == partitioner) &
                          (curves.variant == variant) & (curves.seed == seed)]
            reached = data.loc[data.accuracy >= target, "time"]
            times[int(seed)] = None if reached.empty else float(reached.min())
        record[f"tta_{target}"] = times
    summary.append(record)
(args.input / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
lines = ["| Network | Data | Protocol | Final accuracy (mean ± SD) | Mean bytes |", "|---|---|---|---:|---:|"]
for row in summary:
    lines.append(f"| {row['profile']} | {row['partitioner']} | {row['variant']} | {row['accuracy_mean']:.2f} ± {row['accuracy_std']:.2f}% | {row['bytes_mean'] / 1e9:.3f} GB |")
(args.input / "table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
