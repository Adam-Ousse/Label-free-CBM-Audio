#!/usr/bin/env python3
"""Task 3 baselines and reproducible experiment runner.

The module deliberately keeps the data/cache path shared with ``train_cbm.py``:
AST and CLAP activations are produced by ``utils.save_audio_activations`` and
the sparse heads use the repository's ``glm_saga`` implementation.  The
top-level runner is guarded by ``--execute``; without it it only validates and
prints the planned matrix.

No function in this module uses the test split for model or lambda selection.
The test split is loaded only after a selected validation/path entry has been
chosen.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import data_utils
import similarity
import utils
from clap import core as clap_core
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga


DATASET_SPECS: dict[str, dict[str, Any]] = {
    "esc50": {
        "train": "fold1_train",
        "val": "fold1_val",
        "test": "fold1_test",
        "ast_backbone": "ast_esc50",
        "reference_variant": "lf_broad",
    },
    "urbansound8k": {
        "train": "fold10_train",
        "val": "fold10_val",
        "test": "fold10_test",
        "ast_backbone": "ast_urbansound8k",
        "reference_variant": "lf_broad",
    },
    "cremad": {
        "train": "train",
        "val": "val",
        "test": "test",
        "ast_backbone": "ast_hf__Adam-ousse__ast-cremad-finetuned",
        "reference_variant": "full",
    },
}

DEFAULT_REFERENCE_ROOT = Path("release/google_drive_bundle/checkpoints")
DEFAULT_CLAP_MODEL = "laion/clap-htsat-unfused"
NNZ_THRESHOLD = 1e-5

# This is intentionally a small, repository-owned pseudo-word generator rather
# than an external dictionary.  It avoids accidental class/concept leakage and
# makes the exact random vocabulary reproducible on every machine.
_PSEUDOWORD_SYLLABLES = (
    "ba", "be", "bi", "bo", "bu", "da", "de", "di", "do", "du",
    "fa", "fe", "fi", "fo", "fu", "ga", "ge", "gi", "go", "gu",
    "ka", "ke", "ki", "ko", "ku", "la", "le", "li", "lo", "lu",
    "ma", "me", "mi", "mo", "mu", "na", "ne", "ni", "no", "nu",
    "pa", "pe", "pi", "po", "pu", "ra", "re", "ri", "ro", "ru",
    "sa", "se", "si", "so", "su", "ta", "te", "ti", "to", "tu",
    "va", "ve", "vi", "vo", "vu", "za", "ze", "zi", "zo", "zu",
)


@dataclass(frozen=True)
class SplitFeatures:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


def set_deterministic_seed(seed: int) -> None:
    """Set all local RNGs used by the Task 3 implementations."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_prompts(dataset_name: str, classes: Sequence[str]) -> list[str]:
    """Return the fixed zero-shot prompt protocol from the revision brief."""

    if dataset_name in {"esc50", "urbansound8k"}:
        return ["the sound of {}".format(name) for name in classes]
    if dataset_name == "cremad":
        return ["a person speaking in a {} voice".format(name) for name in classes]
    raise ValueError("Unsupported Task 3 dataset: {}".format(dataset_name))


def generate_random_vocabulary(width: int, seed: int) -> list[str]:
    """Generate a deterministic, class-independent pseudo-word vocabulary."""

    width = int(width)
    if width < 1:
        raise ValueError("Random vocabulary width must be positive")
    rng = np.random.default_rng(int(seed))
    words: list[str] = []
    used: set[str] = set()
    while len(words) < width:
        syllables = rng.choice(_PSEUDOWORD_SYLLABLES, size=3, replace=True)
        word = "".join(str(part) for part in syllables)
        if word in used:
            continue
        used.add(word)
        words.append(word)
    return words


def generate_random_directions(input_dim: int, bottleneck_dim: int, seed: int) -> torch.Tensor:
    """Generate deterministic Gaussian directions with shape [bottleneck, input]."""

    input_dim = int(input_dim)
    bottleneck_dim = int(bottleneck_dim)
    if input_dim < 1 or bottleneck_dim < 1:
        raise ValueError("Random-direction dimensions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(bottleneck_dim, input_dim, generator=generator, dtype=torch.float32)


def mean_nonzeros_per_class(weights: torch.Tensor | np.ndarray, threshold: float = NNZ_THRESHOLD) -> float:
    """Return mean nonzeros per output class for a [classes, features] matrix."""

    tensor = torch.as_tensor(weights)
    if tensor.ndim != 2:
        raise ValueError("Expected a 2D classifier weight matrix")
    return float((tensor.abs() > float(threshold)).sum(dim=1).float().mean().item())


def select_path_entry(
    path: Sequence[dict[str, Any]],
    *,
    target_nec: float | None = None,
    metric_key: str = "acc_val",
    threshold: float = NNZ_THRESHOLD,
) -> dict[str, Any]:
    """Select a glm_saga path entry without looking at test metrics.

    For NEC selection, lambdas are traversed from low to high and the first
    entry whose mean nonzeros/class is at or below the requested NEC is chosen.
    This is the VLG-CBM-style "increase lambda until NEC is reached" rule.  If
    the path never reaches the target, the strongest-regularized entry is
    returned and ``nec_reached`` is false.  Otherwise validation accuracy is
    maximized, with a deterministic preference for fewer nonzeros and then
    larger lambda.
    """

    if not path:
        raise ValueError("Cannot select from an empty regularization path")

    annotated = []
    for index, entry in enumerate(path):
        if "weight" not in entry:
            raise ValueError("Every path entry must contain 'weight'")
        actual = mean_nonzeros_per_class(entry["weight"], threshold=threshold)
        annotated.append((index, entry, actual))

    if target_nec is not None:
        target_nec = float(target_nec)
        if target_nec < 0:
            raise ValueError("NEC target must be non-negative")
        ordered = sorted(annotated, key=lambda item: (float(item[1].get("lam", 0.0)), item[0]))
        eligible = [item for item in ordered if item[2] <= target_nec]
        selected = eligible[0] if eligible else ordered[-1]
        index, entry, actual = selected
        return {
            **entry,
            "selected_index": int(index),
            "actual_mean_nnz_per_class": float(actual),
            "target_nec": target_nec,
            "nec_reached": bool(actual <= target_nec),
            "selection_rule": "smallest_lambda_entry_reaching_nec",
        }

    def score(item: tuple[int, dict[str, Any], float]) -> tuple[float, float, float, int]:
        index, entry, actual = item
        metric = entry.get("metrics", {}).get(metric_key, entry.get(metric_key, float("nan")))
        metric = float(metric) if metric is not None else float("nan")
        if not np.isfinite(metric):
            metric = -float("inf")
        # Maximize validation metric; then prefer sparse and stronger lambda.
        return metric, -actual, float(entry.get("lam", 0.0)), -index

    index, entry, actual = max(annotated, key=score)
    return {
        **entry,
        "selected_index": int(index),
        "actual_mean_nnz_per_class": float(actual),
        "target_nec": None,
        "nec_reached": None,
        "selection_rule": "maximum_validation_metric",
    }


def merge_vocabularies(paths: Sequence[Path], clean_filter: Path | None = None) -> list[str]:
    """Merge source vocabularies stably, optionally retaining Task 2 clean terms."""

    values: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and value not in seen:
                seen.add(value)
                values.append(value)
    if clean_filter is not None:
        allowed = {
            line.strip()
            for line in clean_filter.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        values = [value for value in values if value in allowed]
    if not values:
        raise ValueError("Vocabulary merge produced no concepts")
    return values


def classification_metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    """Compute the common held-out metrics used by the repository."""

    y_true = torch.as_tensor(labels).long().cpu().numpy()
    y_pred = torch.argmax(torch.as_tensor(logits).float().cpu(), dim=1).numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).float()


def _write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _reference_checkpoint(root: Path, dataset: str, variant: str) -> Path:
    path = root / dataset / variant
    required = [path / name for name in ("W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt")]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError("Missing reference checkpoint files: {}".format(", ".join(missing)))
    return path


def _reference_stats(path: Path) -> dict[str, Any]:
    w_c = _load_tensor(path / "W_c.pt")
    w_g = _load_tensor(path / "W_g.pt")
    return {
        "checkpoint": str(path),
        "input_dim": int(w_c.shape[1]),
        "bottleneck_dim": int(w_c.shape[0]),
        "num_classes": int(w_g.shape[0]),
        "mean_nnz_per_class": mean_nonzeros_per_class(w_g),
    }


def _cache_names(args: dict[str, Any], dataset: str, split: str, concept_path: Path, cache_root: Path):
    spec = DATASET_SPECS[dataset]
    # ``utils.get_audio_save_names`` keys concept text features by vocabulary
    # basename.  The same vocabulary basename is used independently for each
    # dataset (and random-word widths differ by dataset), so a shared cache
    # root can silently reuse the wrong text embedding matrix.  Namespace all
    # Task 3 caches by dataset while preserving the existing utility naming
    # scheme within each namespace.
    dataset_cache_root = Path(cache_root) / dataset
    dataset_cache_root.mkdir(parents=True, exist_ok=True)
    return utils.get_audio_save_names(
        args.get("clap_model", DEFAULT_CLAP_MODEL),
        spec["ast_backbone"],
        args.get("feature_layer", "layer4"),
        split,
        str(concept_path),
        "avg",
        str(dataset_cache_root),
    )


def _ensure_feature_cache(
    args: dict[str, Any],
    dataset: str,
    split: str,
    concept_path: Path,
    cache_root: Path,
    *,
    need_clap: bool,
) -> tuple[Path, Path, Path]:
    target, clap_audio, clap_text = _cache_names(args, dataset, split, concept_path, cache_root)
    dataset_cache_root = Path(cache_root) / dataset
    if need_clap:
        utils.save_audio_activations(
            clap_model_name=args.get("clap_model", DEFAULT_CLAP_MODEL),
            target_name=DATASET_SPECS[dataset]["ast_backbone"],
            target_layers=[args.get("feature_layer", "layer4")],
            dataset_name=dataset,
            split=split,
            concept_set=str(concept_path),
            batch_size=int(args.get("feature_batch_size", 64)),
            device=args.get("device", "cuda"),
            pool_mode="avg",
            save_dir=str(dataset_cache_root),
        )
    elif not Path(target).exists():
        dataset_obj = data_utils.get_audio_dataset(dataset, split)
        target_model, _ = data_utils.get_target_model(DATASET_SPECS[dataset]["ast_backbone"], args.get("device", "cuda"))
        utils.save_backbone_audio_features(
            target_model,
            dataset_obj,
            target,
            batch_size=int(args.get("feature_batch_size", 64)),
            device=args.get("device", "cuda"),
        )
        del target_model
    return Path(target), Path(clap_audio), Path(clap_text)


def _load_split_features(
    args: dict[str, Any], dataset: str, concept_path: Path, cache_root: Path, *, need_clap: bool
) -> tuple[SplitFeatures, dict[str, tuple[Path, Path, Path]]]:
    spec = DATASET_SPECS[dataset]
    paths: dict[str, tuple[Path, Path, Path]] = {}
    tensors: dict[str, torch.Tensor] = {}
    for split_key in ("train", "val", "test"):
        split = spec[split_key]
        paths[split_key] = _ensure_feature_cache(
            args, dataset, split, concept_path, cache_root, need_clap=need_clap
        )
        tensors[split_key] = _load_tensor(paths[split_key][0])
    return SplitFeatures(tensors["train"], tensors["val"], tensors["test"]), paths


def _load_labels(dataset: str) -> torch.Tensor:
    spec = DATASET_SPECS[dataset]
    labels = {}
    for key in ("train", "val", "test"):
        ds = data_utils.get_audio_dataset(dataset, spec[key])
        labels[key] = torch.tensor([int(sample["label_idx"]) for sample in ds.samples], dtype=torch.long)
    return labels["train"], labels["val"], labels["test"]


def _load_clap_matrix(audio_path: Path, text_path: Path) -> torch.Tensor:
    return utils.compute_concept_matrix_from_activations(str(audio_path), str(text_path)).float()


def _ensure_text_cache(args: dict[str, Any], texts: Sequence[str], path: Path) -> None:
    if path.exists():
        cached = _load_tensor(path)
        if cached.shape[0] == len(texts):
            return
        raise ValueError("Stale text cache {} has {} rows; expected {}".format(path, cached.shape[0], len(texts)))
    bundle = clap_core.load_clap_model(
        model_name=args.get("clap_model", DEFAULT_CLAP_MODEL), device=args.get("device", "cuda")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_clap_text_features(
        bundle, list(texts), str(path), batch_size=int(args.get("feature_batch_size", 64))
    )


def _fit_projection(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    device: str,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Match the canonical Adam/validation projection training path."""

    set_deterministic_seed(seed)
    input_dim = int(train_features.shape[1])
    output_dim = int(train_targets.shape[1])
    layer = torch.nn.Linear(input_dim, output_dim, bias=False).to(device)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    indices = list(range(len(train_features)))
    batch_size = min(int(batch_size), len(indices))
    best_val = float("inf")
    best_step = 0
    best_weights = layer.weight.detach().cpu().clone()
    if steps < 1:
        steps = 1
    for step in range(int(steps)):
        batch_indices = random.sample(indices, k=batch_size)
        batch = torch.tensor(batch_indices, dtype=torch.long)
        output = layer(train_features[batch].to(device))
        loss = -similarity.cos_similarity_cubed_single(
            train_targets[batch].to(device), output
        ).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                val_output = layer(val_features.to(device))
                val_loss = -similarity.cos_similarity_cubed_single(
                    val_targets.to(device), val_output
                ).mean()
            val_value = float(val_loss.detach().cpu())
            if val_value < best_val:
                best_val = val_value
                best_step = step
                best_weights = layer.weight.detach().cpu().clone()
            elif step > best_step:
                break
    return best_weights, {"best_step": int(best_step), "best_val_loss": float(best_val)}


def _standardize(
    train_features: torch.Tensor, val_features: torch.Tensor, test_features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (
        (train_features - mean) / std,
        (val_features - mean) / std,
        (test_features - mean) / std,
        mean,
        std,
    )


def _fit_glm_saga_path(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    num_classes: int,
    args: dict[str, Any],
) -> list[dict[str, Any]]:
    set_deterministic_seed(int(args.get("seed", 42)))
    batch_size = min(int(args.get("saga_batch_size", 256)), len(train_features))
    train_loader = DataLoader(
        IndexedTensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_features, val_labels), batch_size=batch_size, shuffle=False
    )
    linear = torch.nn.Linear(train_features.shape[1], int(num_classes)).to(args.get("device", "cuda"))
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    # Let glm_saga calculate the maximum regularization from the train loader.
    output = glm_saga(
        linear,
        train_loader,
        float(args.get("saga_step_size", 0.1)),
        int(args.get("saga_iters", 1000)),
        float(args.get("elastic_alpha", 0.99)),
        epsilon=float(args.get("saga_epsilon", 1e-4)),
        k=int(args.get("saga_path_points", 25)),
        val_loader=val_loader,
        do_zero=bool(args.get("saga_include_zero", False)),
        n_ex=len(train_features),
        n_classes=int(num_classes),
        group=bool(args.get("saga_group", True)),
        metadata=None,
    )
    return output["path"]


def _evaluate_selected_head(
    selected: dict[str, Any], test_features: torch.Tensor, test_labels: torch.Tensor
) -> dict[str, Any]:
    weights = torch.as_tensor(selected["weight"]).float()
    bias = torch.as_tensor(selected["bias"]).float()
    with torch.no_grad():
        logits = F.linear(test_features.float(), weights, bias)
    result = classification_metrics(test_labels, logits)
    result.update(
        {
            "lambda": float(selected.get("lam", 0.0)),
            "actual_mean_nnz_per_class": float(
                selected.get("actual_mean_nnz_per_class", mean_nonzeros_per_class(weights))
            ),
            "target_nec": selected.get("target_nec"),
            "nec_reached": selected.get("nec_reached"),
            "selected_path_index": int(selected.get("selected_index", -1)),
        }
    )
    return result


def _save_head_checkpoint(path: Path, selected: dict[str, Any], projection: dict[str, torch.Tensor] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "W_g": torch.as_tensor(selected["weight"]).float().cpu(),
        "b_g": torch.as_tensor(selected["bias"]).float().cpu(),
        "lambda": float(selected.get("lam", 0.0)),
        "actual_mean_nnz_per_class": float(selected["actual_mean_nnz_per_class"]),
    }
    if projection is not None:
        payload.update({key: value.float().cpu() for key, value in projection.items()})
    torch.save(payload, path)


def _fit_logistic_probe(
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    test_x: torch.Tensor,
    train_y: torch.Tensor,
    val_y: torch.Tensor,
    test_y: torch.Tensor,
    *,
    args: dict[str, Any],
) -> dict[str, Any]:
    c_grid = [float(value) for value in args.get("clap_logistic_c_grid", [0.01, 0.1, 1.0, 10.0])]
    candidates = []
    for c_value in c_grid:
        model = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=int(args.get("clap_logistic_max_iter", 1000)),
            random_state=int(args.get("seed", 42)),
        )
        model.fit(train_x.numpy(), train_y.numpy())
        val_pred = model.predict(val_x.numpy())
        candidates.append((float(accuracy_score(val_y.numpy(), val_pred)), -c_value, c_value))
    _, _, best_c = max(candidates)
    final = LogisticRegression(
        C=best_c,
        solver="lbfgs",
        max_iter=int(args.get("clap_logistic_max_iter", 1000)),
        random_state=int(args.get("seed", 42)),
    )
    final.fit(torch.cat([train_x, val_x]).numpy(), torch.cat([train_y, val_y]).numpy())
    logits = torch.from_numpy(final.decision_function(test_x.numpy())).float()
    result = classification_metrics(test_y, logits)
    result.update({"selected_C": float(best_c), "selection_split": "val", "refit_on": "train+val"})
    return result


def _build_task3_vocabularies(
    repo_root: Path, run_dir: Path, dataset: str, variants: Sequence[dict[str, Any]]
) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for variant in variants:
        name = str(variant["name"])
        source_paths = [
            repo_root / str(path).format(dataset=dataset) for path in variant["source_files"]
        ]
        clean_path = (
            repo_root / str(variant["clean_filter"]).format(dataset=dataset)
            if variant.get("clean_filter")
            else None
        )
        values = merge_vocabularies(source_paths, clean_filter=clean_path)
        path = run_dir / "vocabularies" / dataset / (name + ".txt")
        _write_lines(path, values)
        output[name] = path
        _json_dump(
            path.with_suffix(".json"),
            {
                "dataset": dataset,
                "variant": name,
                "source_files": [str(item) for item in source_paths],
                "clean_filter": str(clean_path) if clean_path else None,
                "count": len(values),
            },
        )
    return output


def _run_zero_shot(
    args: dict[str, Any], dataset: str, features: SplitFeatures, paths: dict[str, tuple[Path, Path, Path]], out: Path
) -> dict[str, Any]:
    classes = data_utils.get_dataset_classes(dataset)
    prompts = class_prompts(dataset, classes)
    text_path = out / "class_prompt_embeddings.pt"
    _ensure_text_cache(args, prompts, text_path)
    logits = _load_clap_matrix(paths["test"][1], text_path)
    metrics = classification_metrics(_load_labels(dataset)[2], logits)
    metrics.update({"method": "clap_zero_shot", "dataset": dataset, "prompts": prompts})
    _json_dump(out / "metrics.json", metrics)
    return metrics


def _run_clap_linear(
    args: dict[str, Any], dataset: str, paths: dict[str, tuple[Path, Path, Path]], out: Path
) -> dict[str, Any]:
    train_y, val_y, test_y = _load_labels(dataset)
    train_x = _load_tensor(paths["train"][1])
    val_x = _load_tensor(paths["val"][1])
    test_x = _load_tensor(paths["test"][1])
    metrics = _fit_logistic_probe(train_x, val_x, test_x, train_y, val_y, test_y, args=args)
    metrics.update({"method": "clap_linear_probe", "dataset": dataset})
    _json_dump(out / "metrics.json", metrics)
    return metrics


def _run_sparse_ast(
    args: dict[str, Any], dataset: str, features: SplitFeatures, out: Path, reference_nnz: float
) -> list[dict[str, Any]]:
    train_y, val_y, test_y = _load_labels(dataset)
    train_x, val_x, test_x, mean, std = _standardize(features.train, features.val, features.test)
    path = _fit_glm_saga_path(train_x, train_y, val_x, val_y, num_classes=len(data_utils.get_dataset_classes(dataset)), args=args)
    selections = [("validation_best", None), ("reference_nnz", reference_nnz)]
    results = []
    for name, target in selections:
        selected = select_path_entry(path, target_nec=target)
        metrics = _evaluate_selected_head(selected, test_x, test_y)
        metrics.update(
            {
                "method": "sparse_ast_probe",
                "dataset": dataset,
                "selection": name,
                "selection_split": "val",
                "feature_mean_shape": list(mean.shape),
                "feature_std_shape": list(std.shape),
            }
        )
        _save_head_checkpoint(out / (name + ".pt"), selected, {"proj_mean": mean, "proj_std": std})
        _json_dump(out / (name + ".json"), metrics)
        results.append(metrics)
    return results


def _run_concept_bottleneck(
    args: dict[str, Any],
    dataset: str,
    variant: dict[str, Any],
    vocab_path: Path,
    features: SplitFeatures,
    paths: dict[str, tuple[Path, Path, Path]],
    out: Path,
    reference_nnz: float,
    *,
    random_word: bool = False,
    labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    train_y, val_y, test_y = labels if labels is not None else _load_labels(dataset)
    train_concepts = _load_clap_matrix(paths["train"][1], paths["train"][2])
    val_concepts = _load_clap_matrix(paths["val"][1], paths["val"][2])
    concepts = [line.strip() for line in vocab_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if train_concepts.shape[1] != len(concepts):
        raise ValueError("Concept cache width mismatch for {}".format(vocab_path))

    keep = torch.ones(len(concepts), dtype=torch.bool)
    if not random_word:
        top_k = min(5, train_concepts.shape[0])
        highest = train_concepts.topk(top_k, dim=0).values.mean(dim=0)
        keep = highest > float(args.get("concept_activation_cutoff", 0.25))
    train_targets = train_concepts[:, keep]
    val_targets = val_concepts[:, keep]
    w_c, projection_meta = _fit_projection(
        features.train,
        train_targets,
        features.val,
        val_targets,
        device=args.get("device", "cuda"),
        steps=int(args.get("projection_steps", 1000)),
        batch_size=int(args.get("projection_batch_size", 50000)),
        seed=int(args.get("seed", 42)),
    )
    if random_word:
        interpretable = torch.ones(w_c.shape[0], dtype=torch.bool)
    else:
        with torch.no_grad():
            val_output = F.linear(features.val, w_c)
            similarity_values = similarity.cos_similarity_cubed_single(val_targets, val_output)
        interpretable = similarity_values > float(args.get("interpretability_cutoff", 0.45))
    w_c = w_c[interpretable]
    train_projected = F.linear(features.train, w_c)
    val_projected = F.linear(features.val, w_c)
    test_projected = F.linear(features.test, w_c)
    train_projected, val_projected, test_projected, mean, std = _standardize(
        train_projected, val_projected, test_projected
    )
    saga_args = dict(args)
    saga_args["saga_iters"] = int(args.get("head_saga_iters", args.get("saga_iters", 1000)))
    path = _fit_glm_saga_path(
        train_projected,
        train_y,
        val_projected,
        val_y,
        num_classes=len(data_utils.get_dataset_classes(dataset)),
        args=saga_args,
    )
    selections: list[tuple[str, float | None]] = [("full", None), ("nec25", 25.0), ("nec10", 10.0), ("nec5", 5.0)]
    results = []
    for selection_name, target in selections:
        selected = select_path_entry(path, target_nec=target)
        metrics = _evaluate_selected_head(selected, test_projected, test_y)
        metrics.update(
            {
                "method": "random_word" if random_word else "cbm_no_class_lf",
                "dataset": dataset,
                "variant": variant["name"],
                "selection": selection_name,
                "selection_split": "val",
                "concepts_input": len(concepts),
                "concepts_activation_kept": int(keep.sum().item()),
                "concepts_output": int(w_c.shape[0]),
                "projection": projection_meta,
                "reference_mean_nnz_per_class": float(reference_nnz),
                "random_seed": int(variant["seed"]) if random_word else None,
            }
        )
        _save_head_checkpoint(
            out / (selection_name + ".pt"),
            selected,
            {"W_c": w_c, "proj_mean": mean, "proj_std": std},
        )
        _json_dump(out / (selection_name + ".json"), metrics)
        results.append(metrics)
    return results


def _run_random_direction(
    args: dict[str, Any], dataset: str, features: SplitFeatures, out: Path, reference: dict[str, Any]
) -> dict[str, Any]:
    train_y, val_y, test_y = _load_labels(dataset)
    directions = generate_random_directions(
        reference["input_dim"], reference["bottleneck_dim"], int(args.get("random_direction_seed", 2027))
    )
    train_projected, val_projected, test_projected, mean, std = _standardize(
        F.linear(features.train, directions),
        F.linear(features.val, directions),
        F.linear(features.test, directions),
    )
    path = _fit_glm_saga_path(
        train_projected, train_y, val_projected, val_y,
        num_classes=len(data_utils.get_dataset_classes(dataset)), args=args
    )
    selected = select_path_entry(path)
    metrics = _evaluate_selected_head(selected, test_projected, test_y)
    metrics.update(
        {
            "method": "random_direction",
            "dataset": dataset,
            "selection": "validation_best",
            "selection_split": "val",
            "random_seed": int(args.get("random_direction_seed", 2027)),
            "bottleneck_dim": int(reference["bottleneck_dim"]),
        }
    )
    _save_head_checkpoint(out / "validation_best.pt", selected, {"W_c": directions, "proj_mean": mean, "proj_std": std})
    _json_dump(out / "metrics.json", metrics)
    return metrics


def _prepare_matrix_config(repo_root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["repo_root"] = str(repo_root.resolve())
    return config


def _plan(config: dict[str, Any]) -> list[str]:
    methods = config.get("methods", [])
    rows = []
    for dataset in config.get("datasets", DATASET_SPECS):
        for method in methods:
            rows.append("{}:{}".format(dataset, method))
    return rows


def run_matrix(
    config: dict[str, Any],
    *,
    execute: bool,
    run_id: str | None = None,
    resume: bool = False,
    only_dataset: str | None = None,
) -> Path | None:
    repo_root = Path(config["repo_root"])
    output_root = repo_root / config.get("output_root", "results/revision/t3_baselines")
    if not execute:
        print("Task 3 dry-run; no features, embeddings, models, or results will be generated.")
        print("Planned jobs:")
        for row in _plan(config):
            print("  " + row)
        return None

    if run_id is None:
        run_id = "t3_{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir = output_root / run_id
    if run_dir.exists() and not resume:
        raise FileExistsError("Refusing to overwrite existing Task 3 run: {}".format(run_dir))
    if resume:
        if not run_dir.is_dir() or not (run_dir / "config.json").exists():
            raise FileNotFoundError("Cannot resume missing or invalid Task 3 run: {}".format(run_dir))
    else:
        run_dir.mkdir(parents=True)
        _json_dump(run_dir / "config.json", config)
    set_deterministic_seed(int(config.get("seed", 42)))

    reference_root = repo_root / config.get("reference_root", str(DEFAULT_REFERENCE_ROOT))
    all_results: list[dict[str, Any]] = []
    cache_root = run_dir / "cache"
    datasets = [only_dataset] if only_dataset else config.get("datasets", list(DATASET_SPECS))
    unknown = sorted(set(datasets) - set(DATASET_SPECS))
    if unknown:
        raise ValueError("Unknown Task 3 dataset(s): {}".format(", ".join(unknown)))
    cbm_variants = config.get("cbm_variants", [])
    for dataset in datasets:
        spec = DATASET_SPECS[dataset]
        reference_path = _reference_checkpoint(reference_root, dataset, spec["reference_variant"])
        reference = _reference_stats(reference_path)
        vocab_paths = _build_task3_vocabularies(repo_root, run_dir, dataset, cbm_variants)

        # A concept vocabulary is used to establish the shared AST/CLAP audio
        # cache.  The audio cache filename is independent of the vocabulary.
        cache_concept = next(iter(vocab_paths.values()))
        need_clap = any(method in config.get("methods", []) for method in ("clap_zero_shot", "clap_linear_probe", "cbm_no_class_lf", "random_word"))
        features, base_paths = _load_split_features(config, dataset, cache_concept, cache_root, need_clap=need_clap)
        dataset_root = run_dir / dataset
        dataset_root.mkdir(parents=True, exist_ok=True)
        labels = _load_labels(dataset)

        if "clap_zero_shot" in config.get("methods", []):
            all_results.append(_run_zero_shot(config, dataset, features, base_paths, dataset_root / "clap_zero_shot"))
        if "clap_linear_probe" in config.get("methods", []):
            all_results.append(_run_clap_linear(config, dataset, base_paths, dataset_root / "clap_linear_probe"))
        if "sparse_ast_probe" in config.get("methods", []):
            all_results.extend(_run_sparse_ast(config, dataset, features, dataset_root / "sparse_ast_probe", reference["mean_nnz_per_class"]))
        if "random_direction" in config.get("methods", []):
            all_results.append(_run_random_direction(config, dataset, features, dataset_root / "random_direction", reference))

        for variant in cbm_variants:
            name = str(variant["name"])
            variant_path = vocab_paths[name]
            variant_reference_path = _reference_checkpoint(
                reference_root,
                dataset,
                str(variant.get("reference_variant", spec["reference_variant"])),
            )
            variant_reference = _reference_stats(variant_reference_path)
            if "cbm_no_class_lf" in config.get("methods", []):
                variant_paths = _load_split_features(config, dataset, variant_path, cache_root, need_clap=True)[1]
                all_results.extend(
                    _run_concept_bottleneck(
                        config, dataset, variant, variant_path, features, variant_paths,
                        dataset_root / "cbm_no_class_lf" / name, variant_reference["mean_nnz_per_class"],
                    )
                )
            if "random_word" in config.get("methods", []):
                word_variant = {**variant, "name": name + "_random_word", "seed": int(config.get("random_word_seed", 2028))}
                words = generate_random_vocabulary(variant_reference["bottleneck_dim"], word_variant["seed"])
                word_path = run_dir / "vocabularies" / dataset / (word_variant["name"] + ".txt")
                _write_lines(word_path, words)
                _json_dump(word_path.with_suffix(".json"), {"seed": word_variant["seed"], "count": len(words), "vocabulary": words})
                word_paths = _load_split_features(config, dataset, word_path, cache_root, need_clap=True)[1]
                all_results.extend(
                    _run_concept_bottleneck(
                        config, dataset, word_variant, word_path, features, word_paths,
                        dataset_root / "random_word" / name, variant_reference["mean_nnz_per_class"], random_word=True,
                    )
                )

        _json_dump(dataset_root / "labels.json", {"train": len(labels[0]), "val": len(labels[1]), "test": len(labels[2])})

    _json_dump(run_dir / "metrics.json", all_results)
    if all_results:
        fields = sorted({key for row in all_results for key, value in row.items() if not isinstance(value, (dict, list))})
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
    print("Task 3 run complete: {}".format(run_dir))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true", help="Run the matrix; otherwise only print the plan")
    parser.add_argument("--run-id", default=None, help="Unique output directory name; existing runs are never overwritten")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run directory; requires --run-id and never recreates its config",
    )
    parser.add_argument(
        "--only-dataset",
        choices=sorted(DATASET_SPECS),
        default=None,
        help="Run only this dataset (useful with --resume for an incomplete dataset)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _prepare_matrix_config(args.repo_root.resolve(), args.config.resolve())
    if args.resume and not args.run_id:
        raise SystemExit("--resume requires --run-id")
    run_matrix(
        config,
        execute=args.execute,
        run_id=args.run_id,
        resume=args.resume,
        only_dataset=args.only_dataset,
    )


if __name__ == "__main__":
    main()
