import torch
import torchvision.transforms as T

from dasklearn.datasets.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, \
    AddBackgroundNoiseOnSTFT, ToMelSpectrogramFromSTFT, DeleteSTFT
from dasklearn.datasets.transforms_wav import ChangeAmplitude, ChangeSpeedAndPitchAudio, FixAudioLength, ToTensor, \
    ToMelSpectrogram, LoadAudio

transforms_tens = T.ToTensor()

transforms_tens_resnet = T.Compose([
    T.Resize((224, 224)),  # Ensure input size is large enough for ResNet
    T.ToTensor()
])

def apply_transforms_cifar10(batch):
    batch["img"] = [transforms_tens(img) for img in batch["img"]]
    return batch

def apply_transforms_cifar10_resnet(batch):
    batch["img"] = [transforms_tens_resnet(img) for img in batch["img"]]
    return batch


def preprocess_audio_train(batch):
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
    test_transform = T.Compose(
            [FixAudioLength(), ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])

    audio = [{"samples" : x["array"], "sample_rate" : x["sampling_rate"]} for x in batch["audio"]]

    audio = [test_transform(a) for a in audio]
    batch['img'] = [torch.from_numpy(x["samples"]).unsqueeze(0).float() for x in audio]

    batch["label"] = torch.tensor(batch["label"]).long()

    del batch['audio']
    del batch['speaker_id']
    return batch
