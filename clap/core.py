import math
import wave
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, ClapModel


DEFAULT_CLAP_MODEL = "laion/clap-htsat-unfused"
DEFAULT_CLAP_SAMPLE_RATE = 48000


def _extract_embedding_tensor(model_output, preferred_keys):
    if isinstance(model_output, torch.Tensor):
        return model_output

    for key in preferred_keys:
        if hasattr(model_output, key):
            value = getattr(model_output, key)
            if isinstance(value, torch.Tensor):
                return value

    if isinstance(model_output, (tuple, list)) and len(model_output) > 0:
        first = model_output[0]
        if isinstance(first, torch.Tensor):
            return first

    raise TypeError("Could not extract embedding tensor from CLAP model output")


def load_clap_model(model_name: str = DEFAULT_CLAP_MODEL, device: str = "cuda"):
    """Load CLAP model and processor on a target device."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return {"model": model, "processor": processor, "device": device, "model_name": model_name}


def _load_wav(path: str) -> Tuple[torch.Tensor, int]:
    with wave.open(path, "rb") as wav_file:
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
        raise ValueError("Unsupported WAV sample width for {}: {}".format(path, sample_width))

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return torch.from_numpy(np.ascontiguousarray(audio)).float(), sample_rate


def _resample_if_needed(audio: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    if source_sr == target_sr:
        return audio
    new_len = int(round(audio.shape[-1] * float(target_sr) / float(source_sr)))
    if new_len <= 0:
        raise ValueError("Invalid resampled audio length")
    audio = F.interpolate(audio[None, None, :], size=new_len, mode="linear", align_corners=False)
    return audio.squeeze(0).squeeze(0)


def encode_audio(
    audio_or_paths: Sequence,
    clap_bundle,
    sample_rates: Optional[Sequence[int]] = None,
    batch_size: int = 32,
    target_sample_rate: int = DEFAULT_CLAP_SAMPLE_RATE,
    normalize: bool = True,
) -> torch.Tensor:
    """Encode audio waveforms or wav paths into CLAP audio embeddings.

    `audio_or_paths` may be a list of strings (wav paths) or 1D/2D tensors.
    """
    model = clap_bundle["model"]
    processor = clap_bundle["processor"]
    device = clap_bundle["device"]

    waveforms: List[np.ndarray] = []

    if len(audio_or_paths) == 0:
        raise ValueError("encode_audio received an empty batch")

    if isinstance(audio_or_paths[0], str):
        for path in audio_or_paths:
            wav, sr = _load_wav(path)
            wav = _resample_if_needed(wav, sr, target_sample_rate)
            waveforms.append(wav.cpu().numpy())
    else:
        if sample_rates is None:
            raise ValueError("sample_rates is required when encoding waveform tensors")
        if len(sample_rates) != len(audio_or_paths):
            raise ValueError("sample_rates length must match audio batch length")

        for wav, sr in zip(audio_or_paths, sample_rates):
            if isinstance(wav, torch.Tensor):
                if wav.dim() == 2:
                    wav = wav.mean(dim=0)
                elif wav.dim() != 1:
                    raise ValueError("Expected audio tensors with shape [T] or [C,T]")
                wav = wav.float().cpu()
            else:
                wav = torch.tensor(wav, dtype=torch.float32)
            wav = _resample_if_needed(wav, int(sr), target_sample_rate)
            waveforms.append(wav.numpy())

    audio_embeddings = []
    with torch.no_grad():
        for i in range(math.ceil(len(waveforms) / batch_size)):
            curr = waveforms[i * batch_size : (i + 1) * batch_size]
            try:
                inputs = processor(audio=curr, sampling_rate=target_sample_rate, return_tensors="pt", padding=True)
            except (TypeError, ValueError):
                inputs = processor(audios=curr, sampling_rate=target_sample_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb_out = model.get_audio_features(**inputs)
            emb = _extract_embedding_tensor(emb_out, preferred_keys=["audio_embeds", "pooler_output"])
            audio_embeddings.append(emb.detach().cpu())

    audio_embeddings = torch.cat(audio_embeddings, dim=0)
    if normalize:
        audio_embeddings = F.normalize(audio_embeddings, dim=1)
    return audio_embeddings


def encode_text(
    concepts: Sequence[str],
    clap_bundle,
    batch_size: int = 64,
    normalize: bool = True,
) -> torch.Tensor:
    """Encode concept strings into CLAP text embeddings."""
    if len(concepts) == 0:
        raise ValueError("encode_text received an empty concept list")

    model = clap_bundle["model"]
    processor = clap_bundle["processor"]
    device = clap_bundle["device"]

    text_embeddings = []
    with torch.no_grad():
        for i in range(math.ceil(len(concepts) / batch_size)):
            curr = list(concepts[i * batch_size : (i + 1) * batch_size])
            inputs = processor(text=curr, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb_out = model.get_text_features(**inputs)
            emb = _extract_embedding_tensor(emb_out, preferred_keys=["text_embeds", "pooler_output"])
            text_embeddings.append(emb.detach().cpu())

    text_embeddings = torch.cat(text_embeddings, dim=0)
    if normalize:
        text_embeddings = F.normalize(text_embeddings, dim=1)
    return text_embeddings


def compute_audio_text_similarity(audio_embs: torch.Tensor, text_embs: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Compute batched CLAP audio-text similarity matrix P = E_audio @ E_text^T."""
    if audio_embs.dim() != 2 or text_embs.dim() != 2:
        raise ValueError("Expected 2D embeddings, got {} and {}".format(tuple(audio_embs.shape), tuple(text_embs.shape)))
    if audio_embs.shape[1] != text_embs.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch: audio {} vs text {}".format(audio_embs.shape[1], text_embs.shape[1])
        )

    if normalize:
        audio_embs = F.normalize(audio_embs, dim=1)
        text_embs = F.normalize(text_embs, dim=1)

    return audio_embs @ text_embs.T
