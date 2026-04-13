import os
import json
import wave
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

try:
    import clip
except ModuleNotFoundError:
    clip = None
from pytorchcv.model_provider import get_model as ptcv_get_model
from models.ast_backbone import build_ast_backbone

DATASET_ROOTS = {
    "imagenet_train": "YOUR_PATH/CLS-LOC/train/",
    "imagenet_val": "YOUR_PATH/ImageNet_val/",
    "cub_train":"data/CUB/train",
    "cub_val":"data/CUB/test"
}

LABEL_FILES = {"places365":"archive/original_paper/data/categories_places365_clean.txt",
               "imagenet":"archive/original_paper/data/imagenet_classes.txt",
               "cifar10":"archive/original_paper/data/cifar10_classes.txt",
               "cifar100":"archive/original_paper/data/cifar100_classes.txt",
               "cub":"archive/original_paper/data/cub_classes.txt",
               "esc50":"data/esc50_classes.txt",
               "audioset":"data/audioset_classes.txt"}

AUDIO_DEFAULTS = {
    "esc50": {
        "sample_rate": 16000,
        "duration_sec": 5.0,
        "manifests_dir": "data/esc50/manifests",
    },
    "audioset": {
        "sample_rate": 16000,
        "duration_sec": 10.0,
        "manifests_dir": "data/audioset/manifests",
    },
}

AUDIO_CLASS_FILES = {
    "esc50": "data/esc50_classes.txt",
    "audioset": "data/audioset_classes.txt",
}

AUDIO_MAPPING_FILES = {
    "esc50": {
        "label_to_idx": "data/esc50/label_to_idx.json",
        "idx_to_label": "data/esc50/idx_to_label.json",
    },
    "audioset": {
        "mid_to_idx": "data/audioset/mid_to_idx.json",
        "idx_to_mid": "data/audioset/idx_to_mid.json",
        "idx_to_display_name": "data/audioset/idx_to_display_name.json",
    },
}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_wav_audio(audio_path, target_sample_rate=16000, mono=True):
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        raw = wav_file.readframes(n_frames)

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Unsupported WAV sample width for {}: {}".format(audio_path, sample_width))

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
        if mono:
            audio = np.mean(audio, axis=1)
        else:
            audio = audio.T

    if n_channels == 1 or mono:
        audio_t = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
    else:
        audio_t = torch.from_numpy(np.ascontiguousarray(audio)).float()

    if target_sample_rate is not None and sample_rate != target_sample_rate:
        new_len = int(round(audio_t.shape[-1] * float(target_sample_rate) / float(sample_rate)))
        if new_len <= 0:
            raise ValueError("Invalid resampled length for {}".format(audio_path))
        audio_t = F.interpolate(audio_t.unsqueeze(0), size=new_len, mode="linear", align_corners=False).squeeze(0)
        sample_rate = target_sample_rate

    return audio_t, sample_rate


def _pad_or_truncate(audio, sample_rate, duration_sec=None):
    if duration_sec is None:
        return audio
    target_len = int(round(float(sample_rate) * float(duration_sec)))
    if target_len <= 0:
        return audio
    if audio.shape[-1] > target_len:
        return audio[..., :target_len]
    if audio.shape[-1] < target_len:
        return F.pad(audio, (0, target_len - audio.shape[-1]))
    return audio


class AudioManifestDataset(Dataset):
    """Generic audio dataset backed by JSONL manifests."""

    def __init__(self, dataset_name, manifest_path, sample_rate=16000, mono=True, duration_sec=None):
        self.dataset_name = dataset_name
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError("Manifest not found: {}".format(self.manifest_path))

        self.sample_rate = sample_rate
        self.mono = mono
        self.duration_sec = duration_sec
        self.samples = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        if len(self.samples) == 0:
            raise ValueError("Manifest has no samples: {}".format(self.manifest_path))

        if self.dataset_name == "audioset":
            max_idx = -1
            for sample in self.samples:
                labels = sample.get("label_idx", [])
                if len(labels) > 0:
                    max_idx = max(max_idx, int(max(labels)))
            if max_idx < 0:
                raise ValueError("AudioSet manifest has no valid label_idx values")
            self.num_classes = max_idx + 1
        else:
            self.num_classes = len(get_dataset_classes(self.dataset_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        audio_path = Path(sample["audio_path"])
        if not audio_path.exists():
            raise FileNotFoundError("Audio file missing for sample {}: {}".format(sample.get("id", index), audio_path))

        audio, sr = _load_wav_audio(audio_path, target_sample_rate=self.sample_rate, mono=self.mono)
        audio = _pad_or_truncate(audio, sr, self.duration_sec)

        if self.dataset_name == "audioset":
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            for idx in sample.get("label_idx", []):
                target[int(idx)] = 1.0
        else:
            target = int(sample["label_idx"])

        return {
            "id": sample.get("id", str(index)),
            "audio": audio,
            "sr": int(sr),
            "target": target,
            "path": str(audio_path),
            "dataset": self.dataset_name,
        }


def collate_audio_batch(batch):
    audios = torch.stack([item["audio"] for item in batch], dim=0)
    if isinstance(batch[0]["target"], int):
        targets = torch.LongTensor([item["target"] for item in batch])
    else:
        targets = torch.stack([item["target"] for item in batch], dim=0)

    return {
        "id": [item["id"] for item in batch],
        "audio": audios,
        "sr": torch.LongTensor([item["sr"] for item in batch]),
        "target": targets,
        "path": [item["path"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
    }


def get_dataset_classes(dataset_name):
    if dataset_name in LABEL_FILES:
        with open(LABEL_FILES[dataset_name], "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    if dataset_name in AUDIO_CLASS_FILES:
        class_path = AUDIO_CLASS_FILES[dataset_name]
        if not os.path.exists(class_path):
            raise FileNotFoundError("Class file not found for {}: {}".format(dataset_name, class_path))
        with open(class_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    raise ValueError("Unknown dataset for class loading: {}".format(dataset_name))


def get_audio_manifest_path(dataset_name, split):
    if dataset_name not in AUDIO_DEFAULTS:
        raise ValueError("Unknown audio dataset: {}".format(dataset_name))
    manifest_dir = AUDIO_DEFAULTS[dataset_name]["manifests_dir"]
    manifest_path = os.path.join(manifest_dir, "{}.jsonl".format(split))
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            "Manifest not found for dataset='{}', split='{}': {}".format(dataset_name, split, manifest_path)
        )
    return manifest_path


def get_audio_dataset(dataset_name, split, manifest_path=None, sample_rate=None, mono=True, duration_sec=None):
    if dataset_name not in AUDIO_DEFAULTS:
        raise ValueError("Unknown audio dataset: {}".format(dataset_name))

    if manifest_path is None:
        manifest_path = get_audio_manifest_path(dataset_name, split)

    if sample_rate is None:
        sample_rate = AUDIO_DEFAULTS[dataset_name]["sample_rate"]
    if duration_sec is None:
        duration_sec = AUDIO_DEFAULTS[dataset_name]["duration_sec"]

    return AudioManifestDataset(
        dataset_name=dataset_name,
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        mono=mono,
        duration_sec=duration_sec,
    )


def get_audio_dataloader(dataset_name, split, batch_size=8, shuffle=False, num_workers=0,
                         manifest_path=None, sample_rate=None, mono=True, duration_sec=None):
    dataset = get_audio_dataset(
        dataset_name=dataset_name,
        split=split,
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        mono=mono,
        duration_sec=duration_sec,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
    )


def get_audio_label_mappings(dataset_name):
    if dataset_name not in AUDIO_MAPPING_FILES:
        raise ValueError("Unknown audio dataset mapping request: {}".format(dataset_name))

    out = {}
    for key, path in AUDIO_MAPPING_FILES[dataset_name].items():
        if os.path.exists(path):
            out[key] = _load_json(path)
    return out

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess


def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)
        
    elif dataset_name == "places365_train":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=True,
                                       transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=False,
                                   transform=preprocess)
            
    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True,
                                   transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False,
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])
    return data

def get_targets_only(dataset_name):
    pil_data = get_data(dataset_name)
    return pil_data.targets

def get_target_model(target_name, device):
    if target_name.startswith("ast_"):
        target_model = build_ast_backbone(target_name=target_name, device=device)
        preprocess = None
    
    elif target_name.startswith("clip_"):
        if clip is None:
            raise ModuleNotFoundError("CLIP dependencies are missing. Install requirements.txt to use clip_* backbones.")
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()
    
    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
        
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess