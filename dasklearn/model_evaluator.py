from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

if TYPE_CHECKING:
    from dasklearn.datasets.huggingface import FederatedData

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dasklearn.session_settings import SessionSettings


class ModelEvaluator:
    """
    Contains the logic to evaluate the accuracy of a given model on a test dataset.
    """

    def __init__(self, dataset: FederatedData, settings: SessionSettings):
        self.dataset = dataset
        self.settings: SessionSettings = settings
        self.partition: Optional[Dataset] = None

    def evaluate_accuracy(self, model, device_name: str = "cpu", evaluation_examples=None):
        if self.partition is None:
            from dasklearn.datasets.transforms import prepare_partition
            self.partition = prepare_partition(
                self.dataset.load_split(self.settings.evaluation_split), self.settings, training=False)

        limit = self.settings.evaluation_examples if evaluation_examples is None else evaluation_examples
        partition = self.partition
        if limit:
            if limit < 0:
                raise ValueError("evaluation_examples must be nonnegative")
            partition = (partition.evaluation_subset(limit) if hasattr(partition, "evaluation_subset")
                         else partition.select(range(min(limit, len(partition)))))
        if not len(partition):
            raise ValueError("Cannot evaluate an empty dataset split")
        if self.settings.dataset == "text":
            from dasklearn.datasets.text import evaluate_text
            return evaluate_text(model, partition, self.settings, device_name)
        test_loader = DataLoader(partition, batch_size=256)
        device = torch.device(device_name)

        correct = example_number = total_loss = num_batches = 0
        model.to(device)
        model.eval()

        from dasklearn.datasets.huggingface import column_names
        feature_column_name, label_column_name = column_names(self.settings)

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
