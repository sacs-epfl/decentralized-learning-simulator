"""Prepare checksum-verified Tiny ImageNet without executing dataset repository code."""
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
files = {"train": ("train-00000-of-00001-1359597a978bc4fa.parquet", "220fde5ae5e6ad267b513b5cb1442aa97d69bf535da8f4d16910c59a70388cd3", 100000),
         "valid": ("valid-00000-of-00001-70d52db3c749a935.parquet", "fc85b521b6ee9506df978a65ba46cf869133ae27a3f63e029ffabd78ce268640", 10000)}
for split, (name, digest, count) in files.items():
    path = args.data / name
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        url = f"https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main/data/{name}"
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with path.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    frame = pd.read_parquet(path)
    images = np.lib.format.open_memmap(args.data / f"{split}_images.npy", mode="w+", dtype=np.uint8, shape=(count, 64, 64, 3))
    assert len(frame) == count
    for idx, row in enumerate(frame["image"]):
        images[idx] = np.asarray(Image.open(io.BytesIO(row["bytes"])).convert("RGB"))
    images.flush()
    labels = frame["label"].to_numpy(dtype=np.int64)
    assert set(labels) == set(range(200))
    np.save(args.data / f"{split}_labels.npy", labels)
    print(f"Verified {split}: {count} RGB images", flush=True)
(args.data / "tiny-manifest.json").write_text(json.dumps({"source": "zh-plus/tiny-imagenet", "files": files}, indent=2) + "\n")
