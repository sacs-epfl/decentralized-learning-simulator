from typing import Optional

from datasets import Dataset

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

    def evaluate_accuracy(self, model, device_name: str = "cpu"):
        if not self.partition:
            self.partition = self.dataset.load_split("test")
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
            else:
                raise RuntimeError("Unknown dataset %s for partitioning!" % self.settings.dataset)

        test_loader = DataLoader(self.partition, batch_size=512)
        device = torch.device(device_name)

        correct = example_number = total_loss = num_batches = 0
        model.to(device)
        model.eval()

        ce_loss = nn.CrossEntropyLoss()
        feature_column_name = "x" if self.settings.dataset == "femnist" else "img"
        label_column_name = "y" if self.settings.dataset == "femnist" else "label"
        with torch.no_grad():
            for batch in iter(test_loader):
                data, target = batch[feature_column_name], batch[label_column_name]
                data, target = Variable(data.to(device)), Variable(target.to(device))
                output = model.forward(data)
                if "ResNet" in model.__class__.__name__:
                    total_loss += ce_loss(output, target)
                else:
                    total_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                example_number += target.size(0)
                num_batches += 1

        accuracy = float(correct) / float(example_number) * 100.0
        loss = total_loss / float(example_number)
        return accuracy, loss
