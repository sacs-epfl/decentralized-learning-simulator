import pickle
from typing import Optional

import torch

from dasklearn.models.Model import Model
from dasklearn.models.lora import LORALayer


def serialize_model(model: torch.nn.Module) -> bytes:
    return pickle.dumps(model.state_dict())


def unserialize_model(serialized_model: bytes, dataset: str, architecture: Optional[str] = None) -> torch.nn.Module:
    model = create_model(dataset, architecture=architecture)
    model.load_state_dict(pickle.loads(serialized_model))
    return model


def unserialize_adapter(serialized_adapter: bytes, adapted_layer) -> LORALayer:
    lora_layer = LORALayer(adapted_layer)
    lora_layer.load_state_dict(pickle.loads(serialized_adapter))
    return lora_layer


def create_model(dataset: str, architecture: Optional[str] = None, pretrained: bool = False) -> Model:
    if dataset == "cifar10":
        if not architecture or architecture == "lenet":
            from dasklearn.models.cifar10 import LeNet
            return LeNet()
        elif architecture == "gnlenet":
            from dasklearn.models.cifar10 import GNLeNet
            return GNLeNet(input_channel=3, output=10, model_input=(32, 32))
        elif architecture == "resnet18":
            import torchvision.models as tormodels
            if pretrained:
                from torchvision.models.resnet import ResNet18_Weights
                model = tormodels.resnet18(weights=ResNet18_Weights.DEFAULT)
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = torch.nn.Linear(model.fc.in_features, 10)
                return model
            else:
                return tormodels.resnet18(num_classes=10)
        else:
            raise RuntimeError("Unknown model architecture for CIFAR10: %s" % architecture)
    elif dataset == "femnist":
        from dasklearn.models.femnist import CNN
        return CNN()
    elif dataset == "movielens":
        from dasklearn.models.MovieLens import MatrixFactorization
        return MatrixFactorization()
    else:
        raise RuntimeError("Unknown dataset %s" % dataset)
