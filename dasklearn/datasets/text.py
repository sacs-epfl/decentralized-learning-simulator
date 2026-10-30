"""Tokenized text with disjoint client, validation and final test partitions."""
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F


@lru_cache(maxsize=2)
def tokenizer_for(base_model, tokenizer=None, revision=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer or base_model, revision=revision, trust_remote_code=False)
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise ValueError("Tokenizer needs a pad token or EOS token")
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


class TextPartition(Dataset):
    def __init__(self, tokens, indices):
        self.tokens = tokens
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return {name: values[int(self.indices[index])] for name, values in self.tokens.items()}

    def evaluation_subset(self, limit):
        return TextPartition(self.tokens, self.indices[:limit])


def toy_records(count, offset=0):
    # Deliberately easy smoke workload. Unique IDs keep records disjoint.
    return [{"text": f"{'good happy nice' if i % 2 else 'bad sad awful'} movie example {i}",
             "label": i % 2} for i in range(offset, offset + count)]


class TextDataset:
    def __init__(self, settings):
        if not settings.lora:
            raise ValueError("--dataset text requires --lora and --base-model")
        if settings.partitioner not in {"iid", "dirichlet"}:
            raise ValueError("Text supports iid and dirichlet partitioning")
        source = settings.text_dataset
        if source == "toy-text":
            splits = {"train": toy_records(max(256, settings.participants * 32)),
                      "validation": toy_records(64, 100000), "test": toy_records(64, 200000)}
        elif Path(source).is_dir():
            splits = {}
            for split in ("train", "validation", "test"):
                path = Path(source) / (split + ".jsonl")
                if path.exists():
                    splits[split] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            from datasets import load_dataset
            loaded = load_dataset(source, settings.text_dataset_config)
            splits = {name: loaded[name] for name in ("train", "validation", "test") if name in loaded}
        if "train" not in splits or "test" not in splits or not len(splits["test"]):
            raise ValueError("Text datasets require nonempty train and test splits")
        rng = np.random.default_rng(settings.seed)
        records = list(splits["train"])
        if "validation" not in splits:
            fraction = settings.validation_set_fraction or 0.1
            if not 0 < fraction < 1:
                raise ValueError("validation_set_fraction must be in (0,1)")
            order = rng.permutation(len(records))
            count = max(1, int(len(records) * fraction))
            splits["validation"] = [records[i] for i in order[:count]]
            records = [records[i] for i in order[count:]]
        if not len(splits["validation"]) or len(records) < settings.participants:
            raise ValueError("Need nonempty validation data and at least one training example per peer")
        tok = tokenizer_for(settings.base_model, settings.tokenizer, settings.model_revision)

        def tokenize(rows):
            rows = list(rows)
            encoded = tok([row[settings.text_column] for row in rows], truncation=True,
                          padding="max_length", max_length=settings.max_sequence_length,
                          return_tensors="pt")
            if settings.lora_task == "causal-lm":
                labels = encoded["input_ids"].clone()
                labels[encoded["attention_mask"] == 0] = -100
                if not (labels[:, 1:] != -100).any(dim=1).all():
                    raise ValueError("Every causal LM example needs at least two tokens")
                encoded["labels"] = labels
            else:
                labels = torch.tensor([row[settings.label_column] for row in rows], dtype=torch.long)
                if labels.min() < 0 or labels.max() >= settings.num_labels:
                    raise ValueError("Classification labels must be integers in [0, num_labels)")
                encoded["labels"] = labels
            return dict(encoded)

        tokens = tokenize(records)
        if settings.partitioner == "iid":
            partitions = np.array_split(rng.permutation(len(records)), settings.participants)
        else:
            if settings.lora_task == "causal-lm":
                raise ValueError("Dirichlet text partitioning needs classification labels; use iid for causal-lm")
            if settings.alpha <= 0:
                raise ValueError("Dirichlet alpha must be positive")
            partitions = [[] for _ in range(settings.participants)]
            for label in range(settings.num_labels):
                indices = rng.permutation(np.where(tokens["labels"].numpy() == label)[0])
                counts = rng.multinomial(len(indices), rng.dirichlet(np.full(settings.participants, settings.alpha)))
                for peer, chunk in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
                    partitions[peer].extend(chunk.tolist())
            for indices in partitions:
                if not len(indices):
                    indices.append(max(partitions, key=len).pop())
        self.partitions = [TextPartition(tokens, indices) for indices in partitions]
        self.validation = TextPartition(tokenize(splits["validation"]), np.arange(len(splits["validation"])))
        self.test = TextPartition(tokenize(splits["test"]), np.arange(len(splits["test"])))

    def load_partition(self, peer, split="train"):
        return self.partitions[peer]

    def load_split(self, split):
        if split not in {"test", "validation"}:
            raise ValueError("Text evaluation split must be test or validation")
        return getattr(self, split)


def text_forward(model, batch, device):
    inputs = {name: value.to(device) for name, value in batch.items()}
    # Use explicit losses so classification and token normalization agree across
    # training, validation, accuracy CSVs, and gradient-based algorithms.
    labels = inputs.pop("labels")
    logits = model(**inputs).logits
    if model.spec.task == "causal-lm":
        logits, labels = logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return loss, logits, labels


def evaluate_text(model, partition, settings, device):
    if not len(partition):
        raise ValueError("Cannot evaluate an empty text partition")
    total_loss = correct = count = 0
    model.to(device).eval()
    try:
        with torch.no_grad():
            for batch in DataLoader(partition, batch_size=settings.learning.batch_size):
                loss, logits, labels = text_forward(model, batch, device)
                mask = labels != -100
                size = mask.sum().item()
                total_loss += loss.item() * size
                correct += ((logits.argmax(-1) == labels) & mask).sum().item()
                count += size
        return 100 * correct / count, total_loss / count
    finally:
        model.cpu()
