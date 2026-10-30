"""Tokenized text with disjoint client, validation and final test partitions."""
from functools import lru_cache

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
        from dasklearn.datasets.huggingface import load_splits, allocate_indices
        splits = load_splits(settings)
        records = splits["train"]
        if settings.lora_task == "causal-lm" and settings.partitioner in {"dirichlet", "shards"} and not settings.partition_column:
            raise ValueError("Label-based causal LM partitioning needs --partition-column; use iid otherwise")
        partitions = allocate_indices(records, settings, label_column=settings.label_column)
        tok = tokenizer_for(settings.base_model, settings.tokenizer, settings.model_revision)

        def tokenize(rows):
            encoded = tok(list(rows[settings.text_column]), truncation=True,
                          padding="max_length", max_length=settings.max_sequence_length,
                          return_tensors="pt")
            if settings.lora_task == "causal-lm":
                labels = encoded["input_ids"].clone()
                labels[encoded["attention_mask"] == 0] = -100
                if not (labels[:, 1:] != -100).any(dim=1).all():
                    raise ValueError("Every causal LM example needs at least two tokens")
                encoded["labels"] = labels
            else:
                labels = torch.tensor(list(rows[settings.label_column]), dtype=torch.long)
                if labels.min() < 0 or labels.max() >= settings.num_labels:
                    raise ValueError("Classification labels must be integers in [0, num_labels)")
                encoded["labels"] = labels
            return dict(encoded)

        tokens = tokenize(records)
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
