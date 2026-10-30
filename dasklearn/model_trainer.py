from __future__ import annotations

import logging
from typing import Optional, Dict

import torch
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.utils.data import DataLoader

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dasklearn.datasets.huggingface import FederatedData

from dasklearn.optimizer.sgd import SGDOptimizer
from dasklearn.session_settings import SessionSettings

if TYPE_CHECKING:
    from datasets import Dataset


class ModelTrainer:
    """
    Manager to train a particular model.
    """

    def __init__(self, dataset: FederatedData, settings: SessionSettings, participant_index: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset: FederatedData = dataset
        self.partition: Optional[Dataset] = None
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.is_training: bool = False

    def get_validation_loss(self, model) -> float:
        from dataclasses import replace
        from dasklearn.model_evaluator import ModelEvaluator
        settings = replace(self.settings, evaluation_split="validation", evaluation_examples=0)
        return ModelEvaluator(self.dataset, settings).evaluate_accuracy(model, settings.torch_device_name)[1]

    def prepare_partition(self):
        """Load exactly the same partition/transforms for all execution backends."""
        if self.partition is None:
            from dasklearn.datasets.transforms import prepare_partition
            self.partition = prepare_partition(
                self.dataset.load_partition(self.participant_index, "train"), self.settings, training=True)

    def make_train_loader(self, seed=None):
        self.prepare_partition()
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        return DataLoader(self.partition, batch_size=self.settings.learning.batch_size, shuffle=True, generator=generator)

    def train(self, model, local_steps: int, gradient_only: bool = False, device_name: str = "cpu", seed=None,
              keep_on_device=False) -> Dict:
        """Train a private model; accelerator executors may retain its workspace."""
        self.is_training = True
        self.prepare_partition()

        validation_loss_global_model: Optional[float] = None
        if self.settings.compute_validation_loss_global_model:
            validation_loss_global_model = self.get_validation_loss(model)

        device = torch.device(device_name)
        model = model.to(device)

        train_loader = self.make_train_loader(seed)

        optimizer = SGDOptimizer(model, self.settings.learning.learning_rate, self.settings.learning.momentum, self.settings.learning.weight_decay)
        self.logger.debug("Will perform %d local steps of training on device %s (batch size: %d, lr: %f, wd: %f)",
                          local_steps, device_name, self.settings.learning.batch_size,
                          self.settings.learning.learning_rate, self.settings.learning.weight_decay)

        samples_trained_on = 0

        from dasklearn.datasets.huggingface import column_names
        feature_column_name, label_column_name = column_names(self.settings)
        batches = iter(train_loader)
        for local_step in range(local_steps):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(train_loader)
                batch = next(batches)
            if self.settings.dataset == "text":
                from dasklearn.datasets.text import text_forward
                model.train()
                optimizer.optimizer.zero_grad()
                loss, output, target = text_forward(model, batch, device)
                samples_trained_on += len(target)
                loss.backward()
                if gradient_only:
                    model.gradient = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                                      for p in model.parameters()]
                    break
                optimizer.optimizer.step()
                continue
            data, target = batch[feature_column_name], batch[label_column_name]

            model.train()
            data, target = data.to(device), target.to(device)
            samples_trained_on += len(data)

            optimizer.optimizer.zero_grad()
            self.logger.debug('d-sgd.next node forward propagation (step %d/%d)', local_step, local_steps)
            output = model.forward(data)

            if self.settings.dataset == "cifar10":
                if self.settings.model in ["resnet8", "resnet18", "mobilenet_v3_large"]:
                    lossf = CrossEntropyLoss()
                else:
                    lossf = NLLLoss()
            else:
                lossf = CrossEntropyLoss()

            loss = lossf(output, target)
            self.logger.debug('d-sgd.next node backward propagation (step %d/%d)', local_step, local_steps)
            loss.backward()
            if gradient_only:
                model.gradient = [param.grad.detach() for param in model.parameters()]
                break
            optimizer.optimizer.step()

            del data, target, output, loss

        self.is_training = False
        if not keep_on_device:
            model.to("cpu")
        if self.settings.release_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"samples": samples_trained_on, "validation_loss_global": validation_loss_global_model}

    def gradient_update(self, model, gradient_model):
        """
        Manually set the gradient and perform a single training step
        """
        optimizer = SGDOptimizer(model, self.settings.learning.learning_rate, self.settings.learning.momentum,
                                 self.settings.learning.weight_decay)

        for param, grad in zip(model.parameters(), gradient_model.gradient):
            param.grad = grad

        optimizer.optimizer.step()
