"""Dataset/model presets with configurable Hugging Face sources."""
from dasklearn.datasets.huggingface import FederatedData
from dasklearn.session_settings import SessionSettings


def create_dataset(settings: SessionSettings) -> FederatedData:
    if settings.dataset in {"synthetic", "digits"} and (
            settings.dataset_name or settings.dataset_base_path or settings.dataset_data_files):
        raise ValueError("Offline synthetic/digits workloads do not accept custom dataset sources; choose a model/preprocessing preset")
    if settings.dataset == "text":
        from dasklearn.datasets.text import TextDataset
        return TextDataset(settings)
    if settings.dataset == "tiny_imagenet" and settings.dataset_base_path:
        from dasklearn.datasets.tiny_imagenet import TinyImagenetDataset
        return TinyImagenetDataset(settings)
    if settings.dataset == "cifar10" and settings.dataset_base_path:
        from dasklearn.datasets.cifar10 import Cifar10Dataset
        return Cifar10Dataset(settings)
    if settings.dataset == "digits":
        from dasklearn.datasets.digits import DigitsDataset
        return DigitsDataset(settings)
    if settings.dataset == "synthetic":
        from dasklearn.datasets.synthetic import SyntheticDataset
        return SyntheticDataset(settings)
    from dasklearn.datasets.huggingface import HuggingFaceDataset
    return HuggingFaceDataset(settings)
