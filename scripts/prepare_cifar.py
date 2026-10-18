"""Fetch checksum-verified CIFAR-10 from the canonical HF mirror if Toronto is slow.

Requires pyarrow only during preparation. No account or remote Python code needed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import requests

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--data", type=Path, required=True)
args = p.parse_args()
args.data.mkdir(parents=True, exist_ok=True)
hashes = {"train": "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883",
          "test": "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de"}
arrays = {}
for split, expected in hashes.items():
    url = f"https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"
    path = args.data / f"{split}.parquet"
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with path.open("wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    out.write(chunk)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch for {split}")
    frame = pd.read_parquet(path)
    images = np.stack([np.asarray(Image.open(io.BytesIO(row["bytes"])).convert("RGB")) for row in frame["img"]])
    labels = frame["label"].to_numpy(dtype=np.int64)
    assert images.shape == ((50000 if split == "train" else 10000), 32, 32, 3)
    assert set(labels) == set(range(10))
    arrays[f"{split}_images"], arrays[f"{split}_labels"] = images, labels
    print(f"Verified {split}: {images.shape}", flush=True)
np.savez(args.data / "cifar10-hf.npz", **arrays)
(args.data / "cifar10-hf-manifest.json").write_text(json.dumps({"source": "uoft-cs/cifar10", "sha256": hashes}, indent=2) + "\n")
