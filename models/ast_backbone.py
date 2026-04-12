import numpy as np
import torch

from transformers import ASTModel, AutoFeatureExtractor


AST_MODEL_ALIASES = {
    "ast_audioset": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "ast_mit_audioset": "MIT/ast-finetuned-audioset-10-10-0.4593",
}


def _resolve_model_id(target_name):
    if target_name in AST_MODEL_ALIASES:
        return AST_MODEL_ALIASES[target_name]

    # custom hf id support: ast_hf__org__repo -> org/repo
    if target_name.startswith("ast_hf__"):
        return target_name[len("ast_hf__"):].replace("__", "/")

    return AST_MODEL_ALIASES["ast_audioset"]


class ASTAudioBackbone(torch.nn.Module):
    def __init__(self, model_id, device):
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = ASTModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.hidden_size = int(self.model.config.hidden_size)
        self.default_sample_rate = int(getattr(self.feature_extractor, "sampling_rate", 16000))
        self.expects_sample_rates = True

    def _to_waveform_list(self, audio_batch):
        # expected inputs: [B, T] or [B, 1, T]
        if isinstance(audio_batch, torch.Tensor):
            if audio_batch.dim() == 1:
                audio_batch = audio_batch.unsqueeze(0)
            elif audio_batch.dim() == 3 and audio_batch.shape[1] == 1:
                audio_batch = audio_batch.squeeze(1)
            elif audio_batch.dim() != 2:
                raise ValueError("AST backbone expects audio with shape [B, T] or [B, 1, T]")

            audio_batch = audio_batch.detach().cpu().float()
            return [audio_batch[i].numpy() for i in range(audio_batch.shape[0])]

        waveforms = []
        for wav in audio_batch:
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().float()
                if wav.dim() == 2:
                    wav = wav.mean(dim=0)
                elif wav.dim() != 1:
                    raise ValueError("AST backbone expects each sample with shape [T] or [C, T]")
                waveforms.append(wav.numpy())
            else:
                waveforms.append(np.asarray(wav, dtype=np.float32))
        return waveforms

    def _resolve_sampling_rate(self, sample_rates):
        if sample_rates is None:
            return self.default_sample_rate

        if isinstance(sample_rates, torch.Tensor):
            sample_rates = sample_rates.detach().cpu().tolist()

        unique_sample_rates = {int(sr) for sr in sample_rates}
        if len(unique_sample_rates) != 1:
            raise ValueError("Mixed sample rates in batch are not supported")
        return unique_sample_rates.pop()

    def forward(self, audio_batch, sample_rates=None):
        waveforms = self._to_waveform_list(audio_batch)
        sampling_rate = self._resolve_sampling_rate(sample_rates)

        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)

        outputs = self.model(input_values=input_values)

        # pooled AST embedding shape: [B, hidden_size]
        if outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0, :]

        return features.float()

    def encode_audio(self, audio_batch, sample_rates=None):
        return self.forward(audio_batch, sample_rates=sample_rates)


def build_ast_backbone(target_name, device):
    model_id = _resolve_model_id(target_name)
    return ASTAudioBackbone(model_id=model_id, device=device)