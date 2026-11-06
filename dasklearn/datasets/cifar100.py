"""Native CIFAR-100 tensors; fine labels and the fixed stratified holdout."""
from dasklearn.datasets.cifar10 import Cifar10Dataset


class Cifar100Dataset(Cifar10Dataset):
    models = ('resnet18_gn',)
    num_classes = 100
    mean = (.5071, .4867, .4408)
    std = (.2675, .2565, .2761)
