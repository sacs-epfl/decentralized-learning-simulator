import torch
import torchvision.transforms as T

cifar10_normalization_vectors = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
tiny_imagenet_normalization_vectors = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)

transforms_tens_resnet = T.Compose([
    T.Resize((224, 224)),  # Ensure input size is large enough for ResNet
    T.ToTensor(),
    T.Normalize(*cifar10_normalization_vectors),
])

tiny_imagenet_transforms_tens_resnet = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize((224, 224)),  # Ensure input size is large enough for ResNet
    T.ToTensor(),
    T.Normalize(*tiny_imagenet_normalization_vectors),
])

transforms_tens = T.Compose([
    T.ToTensor(),
    T.Normalize(*cifar10_normalization_vectors),
])

transforms_femnist = T.Compose([
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])

def apply_transforms_cifar10(batch):
    batch["img"] = [transforms_tens(img) for img in batch["img"]]
    return batch

def apply_transforms_cifar10_resnet(batch):
    batch["img"] = [transforms_tens_resnet(img) for img in batch["img"]]
    return batch

def apply_transforms_tiny_imagenet_resnet(batch):
    batch["image"] = [tiny_imagenet_transforms_tens_resnet(img) for img in batch["image"]]
    return batch

def apply_transforms_femnist(batch):
    batch["x"] = [transforms_femnist(img) for img in batch["x"]]
    return batch


def preprocess_audio_train(batch):
    from dasklearn.datasets.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, ToMelSpectrogramFromSTFT, DeleteSTFT
    from dasklearn.datasets.transforms_wav import ChangeAmplitude, ChangeSpeedAndPitchAudio, FixAudioLength, ToTensor
    data_aug_transform = T.Compose(
            [ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(),
             TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    train_feature_transform = T.Compose([ToMelSpectrogramFromSTFT(
            n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])

    audio = [{"samples" : x["array"], "sample_rate" : x["sampling_rate"]} for x in batch["audio"]]
    batch["label"] = torch.tensor(batch["label"]).long()

    audio = [data_aug_transform(a) for a in audio]
    audio = [train_feature_transform(a) for a in audio]
    batch['img'] = [torch.from_numpy(x["samples"]).unsqueeze(0).float() for x in audio]

    del batch['audio']

    return batch


def preprocess_audio_test(batch):
    from dasklearn.datasets.transforms_wav import FixAudioLength, ToMelSpectrogram, ToTensor
    test_transform = T.Compose(
            [FixAudioLength(), ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])

    audio = [{"samples" : x["array"], "sample_rate" : x["sampling_rate"]} for x in batch["audio"]]

    audio = [test_transform(a) for a in audio]
    batch['img'] = [torch.from_numpy(x["samples"]).unsqueeze(0).float() for x in audio]

    batch["label"] = torch.tensor(batch["label"]).long()

    del batch['audio']
    batch.pop('speaker_id', None)
    return batch


def prepare_partition(partition, settings, *, training):
    """One preprocessing decision shared by training and evaluation."""
    if settings.dataset in {"synthetic", "digits", "text", "vision"}:
        return partition
    if settings.dataset == "cifar10":
        transform = (apply_transforms_cifar10_resnet if settings.model in
                     {"resnet8", "resnet18", "mobilenet_v3_large"} else apply_transforms_cifar10)
    elif settings.dataset == "tiny_imagenet":
        transform = apply_transforms_tiny_imagenet_resnet
    elif settings.dataset == "femnist":
        transform = apply_transforms_femnist
    elif settings.dataset == "google_speech":
        transform = preprocess_audio_train if training else preprocess_audio_test
    else:
        raise ValueError(f"Unknown preprocessing preset {settings.dataset!r}")
    feature = "x" if settings.dataset == "femnist" else "image" if settings.dataset == "tiny_imagenet" else "img"
    label = "y" if settings.dataset == "femnist" else "label"

    def model_inputs(batch):
        transformed = transform(batch)
        return {feature: transformed[feature], label: transformed[label]}

    return partition.with_transform(model_inputs)
