import argparse
import base64
import io
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request

import cbm
import data_utils

try:
    import torchaudio
except ImportError:
    torchaudio = None


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class CBMService:
    def __init__(self, model_dir=None, dataset="esc50", device=None):
        self.dataset = dataset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir or self._find_latest_model_dir(dataset)

        if self.model_dir is None:
            raise FileNotFoundError("No CBM model found in saved_models for dataset '{}'".format(dataset))

        self.model = cbm.load_cbm(self.model_dir, self.device)
        self.model.eval()

        self.class_names = data_utils.get_dataset_classes(dataset)
        self.concepts = self._load_concepts()

        default_sr = 16000
        if hasattr(self.model.backbone, "default_sample_rate"):
            default_sr = int(self.model.backbone.default_sample_rate)
        self.sample_rate = default_sr
        self.duration_sec = data_utils.AUDIO_DEFAULTS.get(dataset, {}).get("duration_sec", None)

    def _find_latest_model_dir(self, dataset):
        base = Path("saved_models")
        if not base.exists():
            return None

        pattern = "{}_cbm_*".format(dataset)
        candidates = [p for p in base.glob(pattern) if p.is_dir()]
        if not candidates:
            return None

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])

    def _load_concepts(self):
        concept_path = os.path.join(self.model_dir, "concepts.txt")
        if not os.path.exists(concept_path):
            raise FileNotFoundError("Missing concepts file: {}".format(concept_path))
        with open(concept_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _read_audio(self, upload_file):
        suffix = Path(upload_file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("Unsupported file type '{}'. Allowed: {}".format(suffix, ", ".join(sorted(ALLOWED_EXTENSIONS))))

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name
        tmp.close()
        upload_file.save(tmp_path)

        try:
            # use the built-in wav path to avoid torchaudio/torchcodec runtime issues
            if suffix == ".wav":
                waveform, sr = data_utils._load_wav_audio(
                    tmp_path,
                    target_sample_rate=self.sample_rate,
                    mono=True,
                )
            else:
                if torchaudio is None:
                    raise ValueError("Only .wav is supported in this environment")

                try:
                    waveform, sr = torchaudio.load(tmp_path)
                except Exception as exc:
                    msg = str(exc)
                    if "Could not load libtorchcodec" in msg or "libnvrtc.so.13" in msg:
                        raise RuntimeError(
                            "Non-WAV decoding is unavailable because torchcodec cannot load in this environment. "
                            "Upload a .wav file or install a torchcodec build compatible with torch 2.10.0+cu126."
                        ) from exc
                    raise

                if waveform.dim() != 2:
                    raise ValueError("Expected waveform shape [C, T], got {}".format(tuple(waveform.shape)))
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if int(sr) != int(self.sample_rate):
                    waveform = torchaudio.functional.resample(waveform, int(sr), int(self.sample_rate))
                sr = self.sample_rate

            waveform = data_utils._pad_or_truncate(waveform, int(sr), self.duration_sec)
            return waveform.float()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _make_concept_figure(self, positive, negative):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

        pos_labels = [item["concept"] for item in positive][::-1]
        pos_scores = [item["score"] for item in positive][::-1]
        neg_labels = [item["concept"] for item in negative]
        neg_scores = [item["score"] for item in negative]

        axes[0].barh(pos_labels, pos_scores, color="#1f8a70")
        axes[0].set_title("Top Positive Influence")
        axes[0].set_xlabel("Contribution to predicted class logit")

        axes[1].barh(neg_labels, neg_scores, color="#c44536")
        axes[1].set_title("Top Negative Influence")
        axes[1].set_xlabel("Contribution to predicted class logit")

        for ax in axes:
            ax.axvline(0.0, color="#555", linewidth=1.0)
            ax.tick_params(axis="y", labelsize=8)

        out = io.BytesIO()
        fig.savefig(out, format="png", dpi=150)
        plt.close(fig)
        out.seek(0)
        return base64.b64encode(out.read()).decode("utf-8")

    def predict(self, upload_file, top_k=10):
        waveform = self._read_audio(upload_file)
        audio_batch = waveform.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, proj_c = self.model(audio_batch)
            probs = F.softmax(logits, dim=1)

        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_conf = float(probs[0, pred_idx].item())

        class_name = str(pred_idx)
        if 0 <= pred_idx < len(self.class_names):
            class_name = self.class_names[pred_idx]

        concept_values = proj_c[0].detach().cpu()
        class_weights = self.model.final.weight[pred_idx].detach().cpu()
        influences = concept_values * class_weights

        top_k = max(3, min(int(top_k), 20))
        pos_indices = torch.argsort(influences, descending=True)[:top_k].tolist()
        neg_indices = torch.argsort(influences, descending=False)[:top_k].tolist()

        positive = [
            {
                "concept": self.concepts[i] if i < len(self.concepts) else "concept_{}".format(i),
                "score": float(influences[i].item()),
            }
            for i in pos_indices
        ]
        negative = [
            {
                "concept": self.concepts[i] if i < len(self.concepts) else "concept_{}".format(i),
                "score": float(influences[i].item()),
            }
            for i in neg_indices
        ]

        plot_b64 = self._make_concept_figure(positive, negative)

        return {
            "predicted_index": pred_idx,
            "predicted_class": class_name,
            "confidence": pred_conf,
            "top_positive": positive,
            "top_negative": negative,
            "plot_base64": plot_b64,
            "model_dir": self.model_dir,
        }


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["CBM_SERVICE"] = None

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        if "audio" not in request.files:
            return jsonify({"error": "Missing file field 'audio'"}), 400

        audio_file = request.files["audio"]
        if audio_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        try:
            top_k = request.form.get("top_k", 10)
            service = _get_service(app)
            result = service.predict(audio_file, top_k=top_k)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def _get_service(app):
    service = app.config.get("CBM_SERVICE")
    if service is None:
        model_dir = app.config.get("MODEL_DIR")
        dataset = app.config.get("DATASET", "esc50")
        device = app.config.get("DEVICE")
        service = CBMService(model_dir=model_dir, dataset=dataset, device=device)
        app.config["CBM_SERVICE"] = service
    return service


def parse_args():
    parser = argparse.ArgumentParser(description="Flask web UI for ESC-50 CBM concept inspection")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to saved_models/esc50_cbm_* directory")
    parser.add_argument("--dataset", type=str, default="esc50", help="Dataset label file to use")
    parser.add_argument("--device", type=str, default=None, help="Inference device, e.g. cuda or cpu")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Flask host")
    parser.add_argument("--port", type=int, default=5000, help="Flask port")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app()
    app.config["MODEL_DIR"] = args.model_dir
    app.config["DATASET"] = args.dataset
    app.config["DEVICE"] = args.device
    app.run(host=args.host, port=args.port, debug=args.debug)
