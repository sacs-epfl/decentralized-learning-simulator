from torchvision.transforms import ToTensor


transforms = ToTensor()

def apply_transforms_cifar10(batch):
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch
