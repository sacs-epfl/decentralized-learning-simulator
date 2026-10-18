from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

if TYPE_CHECKING:
    from flwr_datasets import FederatedDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dasklearn.session_settings import SessionSettings


class ModelEvaluator:
    """
    Contains the logic to evaluate the accuracy of a given model on a test dataset.
    """

    def __init__(self, dataset: FederatedDataset, settings: SessionSettings):
        self.dataset = dataset
        self.settings: SessionSettings = settings
        self.partition: Optional[Dataset] = None

    def evaluate_accuracy(self, model, device_name: str = "cpu", evaluation_examples=None):
        if not self.partition:
            partition_name = "valid" if self.settings.dataset == "tiny_imagenet" else "test"
            if self.settings.evaluation_split != "test":
                partition_name = self.settings.evaluation_split
            self.partition = self.dataset.load_split(partition_name)
            if self.settings.dataset == "cifar10":
                from dasklearn.datasets.transforms import apply_transforms_cifar10, apply_transforms_cifar10_resnet
                transforms = apply_transforms_cifar10_resnet if self.settings.model in ["resnet8", "resnet18", "mobilenet_v3_large"] else apply_transforms_cifar10
                self.partition = self.partition.with_transform(transforms)
            elif self.settings.dataset == "femnist":
                from dasklearn.datasets.transforms import apply_transforms_femnist
                self.partition = self.partition.with_transform(apply_transforms_femnist)
            elif self.settings.dataset == "google_speech":
                from dasklearn.datasets.transforms import preprocess_audio_test as transforms
                # filter removes the silent samples from testing/training as they don't really have a label
                self.partition = self.partition.filter(lambda x : x["speaker_id"] is not None).with_transform(transforms)
            elif self.settings.dataset == "tiny_imagenet":
                from dasklearn.datasets.transforms import apply_transforms_tiny_imagenet_resnet
                self.partition = self.partition.with_transform(apply_transforms_tiny_imagenet_resnet)
            elif self.settings.dataset in ["synthetic", "digits"]:
                pass
            else:
                raise RuntimeError("Unknown dataset %s for partitioning!" % self.settings.dataset)

        limit = self.settings.evaluation_examples if evaluation_examples is None else evaluation_examples
        partition = self.partition.evaluation_subset(limit) if limit else self.partition
        test_loader = DataLoader(partition, batch_size=256)
        device = torch.device(device_name)

        correct = example_number = total_loss = num_batches = 0
        model.to(device)
        model.eval()

        if self.settings.dataset == "femnist":
            feature_column_name = "x"
        elif self.settings.dataset == "tiny_imagenet":
            feature_column_name = "image"
        else:
            feature_column_name = "img"

        label_column_name = "y" if self.settings.dataset == "femnist" else "label"

        if self.settings.dataset == "cifar10":
            lossf = nn.CrossEntropyLoss() if self.settings.model in ["resnet8", "resnet18", "mobilenet_v3_large"] else nn.NLLLoss()
        else:
            lossf = nn.CrossEntropyLoss()
    
        with torch.no_grad():
            for batch in iter(test_loader):
                data, target = batch[feature_column_name], batch[label_column_name]
                data, target = Variable(data.to(device)), Variable(target.to(device))
                output = model(data)
                loss = lossf(output, target)
                total_loss += loss.item() * len(target)

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                example_number += target.size(0)
                num_batches += 1

                del data, target, output, loss, pred

        # We move the model back to the CPU to avoid memory leaks
        model.to("cpu")
        if self.settings.release_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        accuracy = float(correct) / float(example_number) * 100.0
        average_loss = total_loss / example_number
        return accuracy, average_loss
