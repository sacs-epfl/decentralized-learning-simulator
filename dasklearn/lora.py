"""Adapter-only DAG values backed by one lazily loaded PEFT model per process.

The base is deliberately NOT a child module or an attribute of LoRAModel: copies,
state_dict, chunking and broker pickle messages can only contain trainable state.
Functional calls temporarily substitute a client's parameters without copying the
base. A lock serializes calls on the shared module (including device/mode changes).
"""
from dataclasses import dataclass
from functools import lru_cache
import threading

import torch
from torch import nn


@dataclass(frozen=True)
class LoRASpec:
    base_model: str
    revision: str | None = None
    rank: int = 8
    alpha: float = 16
    dropout: float = 0.0
    targets: str | None = None
    task: str = "sequence-classification"
    num_labels: int = 2
    seed: int = 42
    tokenizer: str | None = None

    @classmethod
    def from_settings(cls, settings):
        spec = cls(base_model=settings.base_model, revision=settings.model_revision,
                   rank=settings.lora_rank, alpha=settings.lora_alpha, dropout=settings.lora_dropout,
                   targets=settings.lora_target_modules, task=settings.lora_task,
                   num_labels=settings.num_labels, seed=settings.seed, tokenizer=settings.tokenizer)
        if not spec.base_model:
            raise ValueError("--lora requires --base-model (Hugging Face ID or local checkpoint)")
        expected_dataset = "vision" if spec.task == "image-classification" else "text"
        if settings.dataset != expected_dataset:
            raise ValueError(f"LoRA task {spec.task!r} requires --dataset {expected_dataset}")
        if (spec.rank < 1 or not 0 <= spec.dropout < 1
                or not 0 < spec.alpha < float('inf') or spec.num_labels < 2):
            raise ValueError("LoRA needs rank >= 1, finite alpha > 0, dropout in [0,1), num_labels >= 2")
        if spec.task not in {"sequence-classification", "causal-lm", "image-classification"}:
            raise ValueError("Unknown LoRA task")
        if spec.task != "image-classification" and settings.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be >= 2")
        return spec


class _Runtime:
    def __init__(self, spec):
        try:
            from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForImageClassification
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("Install LoRA dependencies: pip install -r requirements-lora.txt") from exc
        self.lock = threading.RLock()
        # Missing classification heads and adapter initialization must agree on
        # every broker, independent of when this process first loads the base.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed)
            factory = {"causal-lm": AutoModelForCausalLM, "sequence-classification": AutoModelForSequenceClassification,
                       "image-classification": AutoModelForImageClassification}[spec.task]
            options = {"revision": spec.revision, "trust_remote_code": False}
            if spec.task != "causal-lm":
                options["num_labels"] = spec.num_labels
                options["ignore_mismatched_sizes"] = True
            base = factory.from_pretrained(spec.base_model, **options)
            if spec.task != "image-classification" and base.config.pad_token_id is None:
                from dasklearn.datasets.text import tokenizer_for
                base.config.pad_token_id = tokenizer_for(spec.base_model, spec.tokenizer, spec.revision).pad_token_id
            if hasattr(base.config, "use_cache"):
                base.config.use_cache = False
            targets = spec.targets
            if not targets and spec.task == "image-classification":
                # Transformers versions expose either separate query/value or
                # q_proj/v_proj projections for ViT checkpoints.
                names = {name.rsplit(".", 1)[-1] for name, module in base.named_modules()
                         if isinstance(module, nn.Linear)}
                for pair in ("query,value", "q_proj,v_proj", "qkv"):
                    if set(pair.split(",")) <= names:
                        targets = pair
                        break
                if not targets:
                    raise ValueError("Set --lora-target-modules for this vision architecture")
            if targets and targets != "all-linear":
                targets = [name.strip() for name in targets.split(",") if name.strip()]
            self.model = get_peft_model(base, LoraConfig(
                r=spec.rank, lora_alpha=spec.alpha, lora_dropout=spec.dropout,
                target_modules=targets, bias="none",
                modules_to_save=["classifier"] if spec.task == "image-classification" else None,
                task_type={"causal-lm": "CAUSAL_LM", "sequence-classification": "SEQ_CLS"}.get(spec.task)))
        self.names = tuple(name for name, p in self.model.named_parameters() if p.requires_grad)
        # Keep the runtime itself gradient-free; functional_call supplies the
        # client's trainable Parameters for every adapter/head entry.
        self.model.requires_grad_(False)


# A process normally runs one experiment. A bounded cache avoids retaining bases
# indefinitely when an interactive process runs different configurations.
@lru_cache(maxsize=1)
def shared_runtime(spec):
    return _Runtime(spec)


class LoRAModel(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        runtime = shared_runtime(spec)
        with runtime.lock:
            parameters = dict(runtime.model.named_parameters())
            self.adapter = nn.ParameterList([
                nn.Parameter(parameters[name].detach().cpu().clone()) for name in runtime.names])
            self.parameter_names = runtime.names

    def forward(self, **inputs):
        runtime = shared_runtime(self.spec)
        with runtime.lock:
            runtime.model.to(self.adapter[0].device)
            runtime.model.train(self.training)
            return torch.func.functional_call(runtime.model,
                dict(zip(self.parameter_names, self.adapter)), (), inputs, strict=False)


def add_lora_arguments(parser):
    parser.add_argument("--lora", action="store_true", help="Fine-tune adapters on a shared frozen transformer")
    parser.add_argument("--base-model", help="Hugging Face model ID or local pretrained checkpoint")
    parser.add_argument("--model-revision", default=None, help="Pin a model/tokenizer revision for reproducibility")
    parser.add_argument("--tokenizer", default=None, help="Defaults to --base-model")
    parser.add_argument("--lora-rank", "--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default=None, help="PEFT defaults, all-linear, or comma-separated module names")
    parser.add_argument("--lora-task", choices=["sequence-classification", "causal-lm", "image-classification"], default="sequence-classification")
    parser.add_argument("--image-processor", default=None, help="Defaults to the pretrained vision checkpoint's processor")
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--text-dataset", default="toy-text", help="Alias for --dataset-name for text (default: toy-text)")
    parser.add_argument("--text-dataset-config", default=None)
    parser.add_argument("--text-column", default="text")


def lora_options(args):
    return {name: getattr(args, name) for name in (
        "lora", "base_model", "model_revision", "tokenizer", "lora_rank", "lora_alpha",
        "lora_dropout", "lora_target_modules", "lora_task", "num_labels", "max_sequence_length",
        "text_dataset", "text_dataset_config", "text_column", "label_column", "image_processor")}


def lora_forward(model, batch, device):
    """Explicit classification/token losses shared by all LoRA modalities."""
    from torch.nn import functional as F
    inputs = {name: value.to(device) for name, value in batch.items()}
    labels = inputs.pop("labels")
    logits = model(**inputs).logits
    if model.spec.task == "causal-lm":
        logits, labels = logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    return loss, logits, labels


def evaluate_lora(model, partition, settings, device):
    from torch.utils.data import DataLoader
    if not len(partition):
        raise ValueError("Cannot evaluate an empty LoRA partition")
    total_loss = correct = count = 0
    model.to(device).eval()
    try:
        with torch.no_grad():
            for batch in DataLoader(partition, batch_size=settings.learning.batch_size):
                loss, logits, labels = lora_forward(model, batch, device)
                mask = labels != -100
                size = mask.sum().item()
                total_loss += loss.item() * size
                correct += ((logits.argmax(-1) == labels) & mask).sum().item()
                count += size
        return 100 * correct / count, total_loss / count
    finally:
        model.cpu()
