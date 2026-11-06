"""ResNet-18 for native 32x32 images, without running-statistic buffers."""
from torch import nn
from torchvision.models import resnet18


def resnet18_gn(num_classes=100):
    model = resnet18(weights=None, num_classes=num_classes,
                     norm_layer=lambda channels: nn.GroupNorm(32, channels))
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    model.maxpool = nn.Identity()
    return model
