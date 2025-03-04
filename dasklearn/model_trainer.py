import logging
from typing import Optional, Dict

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss
from torch.utils.data import DataLoader

from flwr_datasets import FederatedDataset

from dasklearn.optimizer.sgd import SGDOptimizer
from dasklearn.session_settings import SessionSettings

from datasets import Dataset


class ModelTrainer:
    """
    Manager to train a particular model.
    """

    def __init__(self, dataset: FederatedDataset, settings: SessionSettings, participant_index: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset: FederatedDataset = dataset
        self.partition: Optional[Dataset] = None
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.is_training: bool = False

        self.optimizer: Optional[SGDOptimizer] = None

    def get_validation_loss(self, model) -> float:
        validation_set = self.dataset.get_validationset()
        total_loss = 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()  # set model to evaluation mode

        with torch.no_grad():
            for data, target in validation_set:
                data, target = data.to(device), target.to(device)
                output = model(data)

                if self.settings.dataset == "movielens":
                    loss_func = MSELoss()
                elif self.settings.dataset == "cifar10":
                    if self.settings.model == "resnet8":
                        loss_func = CrossEntropyLoss()
                    else:
                        loss_func = NLLLoss()
                else:
                    loss_func = CrossEntropyLoss()

                loss = loss_func(output, target)
                total_loss += loss.item()

        avg_loss = total_loss / len(validation_set)
        return float(avg_loss)

    def train(self, model, local_steps: int, gradient_only: bool = False, device_name: str = "cpu") -> Dict:
        """
        Train the model on a batch. Return an integer that indicates how many local steps we have done.
        """
        self.is_training = True

        # Load the partition if it's not loaded yet
        if not self.partition:
            self.partition = self.dataset.load_partition(self.participant_index, "train")
            if(self.settings.dataset == "cifar10"):
                from dasklearn.datasets.transforms import apply_transforms_cifar10, apply_transforms_cifar10_resnet
                transforms = apply_transforms_cifar10_resnet if self.settings.model in ["resnet8", "resnet18", "mobilenet_v3_large"] else apply_transforms_cifar10
                self.partition = self.partition.with_transform(transforms)
            elif(self.settings.dataset == "google_speech"):
                # filter removes the silent samples from testing/training as they don't really have a label
                from dasklearn.datasets.transforms import preprocess_audio_train as transforms
                self.partition = self.partition.filter(lambda x : x["speaker_id"] is not None).with_transform(transforms)
            else:
                raise RuntimeError("Unknown dataset %s for partitioning!" % self.settings.dataset)

        validation_loss_global_model: Optional[float] = None
        if self.settings.compute_validation_loss_global_model and len(self.dataset.validationset) > 0:
            validation_loss_global_model = self.get_validation_loss(model)

        device = torch.device(device_name)
        model = model.to(device)

        train_loader = DataLoader(self.partition, batch_size=self.settings.learning.batch_size, shuffle=True)

        optimizer = SGDOptimizer(model, self.settings.learning.learning_rate, self.settings.learning.momentum, self.settings.learning.weight_decay)
        if self.optimizer is not None:
            optimizer.optimizer.load_state_dict(self.optimizer.optimizer.state_dict())
        self.optimizer = optimizer

        self.logger.debug("Will perform %d local steps of training on device %s (batch size: %d, lr: %f, wd: %f)",
                          local_steps, device_name, self.settings.learning.batch_size,
                          self.settings.learning.learning_rate, self.settings.learning.weight_decay)

        samples_trained_on = 0
        for local_step, batch in enumerate(train_loader):
            data, target = batch["img"], batch["label"]  # TODO hard-coded, not generic enough for different datasets
            if local_step >= local_steps:
                break

            model.train()
            data, target = Variable(data.to(device)), Variable(target.to(device))
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

        self.is_training = False
        model.to("cpu")

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
