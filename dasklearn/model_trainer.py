import logging
import os
from typing import Optional

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss

from dasklearn.datasets import create_dataset, Dataset
from dasklearn.optimizer.sgd import SGDOptimizer
from dasklearn.session_settings import SessionSettings

AUGMENTATION_FACTOR_SIM = 3.0


class ModelTrainer:
    """
    Manager to train a particular model.
    Runs in a separate process.
    """

    def __init__(self, data_dir, settings: SessionSettings, participant_index: int):
        """
        :param simulated_speed: compute speed of the simulated device, in ms/sample.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.simulated_speed: Optional[float] = None
        self.total_training_time: float = 0
        self.is_training: bool = False

        if settings.dataset in ["cifar10", "mnist", "movielens", "spambase"]:
            self.train_dir = data_dir
        else:
            self.train_dir = os.path.join(data_dir, "per_user_data", "train")
        self.dataset: Optional[Dataset] = None

    def train(self, model, device_name: str = "cpu") -> int:
        """
        Train the model on a batch. Return an integer that indicates how many local steps we have done.
        """
        self.is_training = True

        if not self.dataset:
            self.dataset = create_dataset(self.settings, participant_index=self.participant_index, train_dir=self.train_dir)

        local_steps: int = self.settings.learning.local_steps
        device = torch.device(device_name)
        model = model.to(device)
        optimizer = SGDOptimizer(model, self.settings.learning.learning_rate, self.settings.learning.momentum, self.settings.learning.weight_decay)

        self.logger.info("Will perform %d local steps of training on device %s (batch size: %d, lr: %f, wd: %f)",
                         local_steps, device_name, self.settings.learning.batch_size,
                         self.settings.learning.learning_rate, self.settings.learning.weight_decay)

        samples_trained_on = 0
        model = model.to(device)  # just to make sure...
        for local_step in range(local_steps):
            train_set = self.dataset.get_trainset(batch_size=self.settings.learning.batch_size, shuffle=True)
            train_set_it = iter(train_set)

            data, target = next(train_set_it)
            model.train()
            data, target = Variable(data.to(device)), Variable(target.to(device))
            samples_trained_on += len(data)

            optimizer.optimizer.zero_grad()
            self.logger.debug('d-sgd.next node forward propagation (step %d/%d)', local_step, local_steps)
            output = model.forward(data)

            if self.settings.dataset == "movielens":
                lossf = MSELoss()
            elif self.settings.dataset == "cifar10":
                if self.settings.model == "resnet8":
                    lossf = CrossEntropyLoss()
                else:
                    lossf = NLLLoss()
            else:
                lossf = CrossEntropyLoss()

            loss = lossf(output, target)
            self.logger.debug('d-sgd.next node backward propagation (step %d/%d)', local_step, local_steps)
            loss.backward()
            optimizer.optimizer.step()

        self.is_training = False

        return samples_trained_on
