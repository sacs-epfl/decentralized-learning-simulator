"""Vision LoRA inputs prepared with the pretrained model's own image processor."""
from functools import lru_cache

import torch

from dasklearn.datasets.huggingface import HuggingFaceDataset


@lru_cache(maxsize=2)
def image_processor_for(base_model, processor=None, revision=None):
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(processor or base_model, revision=revision, trust_remote_code=False)


class VisionDataset(HuggingFaceDataset):
    def __init__(self, settings):
        if not settings.lora or settings.lora_task != "image-classification":
            raise ValueError("--dataset vision requires --lora --lora-task image-classification")
        if not settings.dataset_name and not settings.dataset_base_path:
            raise ValueError("--dataset vision requires --dataset-name or --dataset-path")
        super().__init__(settings)
        self.processor = image_processor_for(settings.base_model, settings.image_processor, settings.model_revision)
        self.num_labels = settings.num_labels

    def transform(self, batch):
        pixels = self.processor(images=[image.convert("RGB") for image in batch["image"]], return_tensors="pt")
        labels = torch.tensor(batch["label"], dtype=torch.long)
        if labels.min() < 0 or labels.max() >= self.num_labels:
            raise ValueError("Classification labels must be integers in [0, num_labels)")
        return {"pixel_values": pixels["pixel_values"], "labels": labels}

    def load_partition(self, peer, split="train"):
        return super().load_partition(peer, split).with_transform(self.transform, columns=["image", "label"])

    def load_split(self, split):
        return super().load_split(split).with_transform(self.transform, columns=["image", "label"])
