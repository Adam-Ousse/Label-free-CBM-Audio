#!/usr/bin/env python3
"""CPU-only Task-6 aggregation, QC, manuscript tables, and figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.task6_urbansed_analysis import (
    compute_task6_metrics,
    event_active_window_labels,
    load_event_window_rows,
)


METHODS = (
    "cbm_positive_contribution",
    "cbm_silence_baseline",
    "ast_window_logit",
    "ast_occlusion_drop",
    "ast_attention_rollout",
)
METHOD_LABELS = {
    "cbm_positive_contribution": r"$\alpha$-CBM",
    "cbm_silence_baseline": r"$\alpha$-CBM + silence",
    "ast_window_logit": "AST logits",
    "ast_occlusion_drop": "Occlusion",
    "ast_attention_rollout": "Attention rollout",
}
HOPS = (0.5, 0.25)


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_defined(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _metric_row(hop: float, method: str, class_name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "hop_seconds": hop,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "class": class_name,
        "positive_windows": result["positive_windows"],
        "positive_soundscapes": result["positive_soundscapes"],
        "auroc": result["auroc"],
        "auroc_ci_low": result["auroc_ci"][0] if result["auroc_ci"] else None,
        "auroc_ci_high": result["auroc_ci"][1] if result["auroc_ci"] else None,
        "average_precision": result["average_precision"],
        "average_precision_ci_low": result["average_precision_ci"][0] if result["average_precision_ci"] else None,
        "average_precision_ci_high": result["average_precision_ci"][1] if result["average_precision_ci"] else None,
        "pointing_game": result["pointing_game"],
        "pointing_game_ci_low": result["pointing_game_ci"][0] if result["pointing_game_ci"] else None,
        "pointing_game_ci_high": result["pointing_game_ci"][1] if result["pointing_game_ci"] else None,
    }


def _aggregate_method(result: dict[str, Any]) -> dict[str, Any]:
    by_class = result["by_class"]
    return {
        "auroc": _mean_defined([value["auroc"] for value in by_class.values()]),
        "average_precision": _mean_defined([value["average_precision"] for value in by_class.values()]),
        "pointing_game": _mean_defined([value["pointing_game"] for value in by_class.values()]),
        "n_classes": len(by_class),
        "n_windows": result["n_windows"],
        "n_soundscapes": result["n_soundscapes"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex(value: float | None) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def _write_tables(root: Path, summary: dict[str, Any], class_rows: list[dict[str, Any]]) -> None:
    main_path = root / "tables/task6_main_quantitative.tex"
    main_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{URBAN-SED temporal localization. Values are macro means over the ten event classes; bootstrap intervals are reported per class in the supplement.}",
        r"\label{tab:task6-localization}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Hop & Method & AUROC & AP & Pointing \\" ,
        r"\midrule",
    ]
    for hop in HOPS:
        for method in METHODS:
            aggregate = summary["by_hop"][str(hop)]["methods"][method]["macro"]
            lines.append(
                f"{hop:g} s & {METHOD_LABELS[method]} & {_latex(aggregate['auroc'])} & {_latex(aggregate['average_precision'])} & {_latex(aggregate['pointing_game'])} "
                + r"\\"
            )
        if hop != HOPS[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    main_path.write_text("\n".join(lines), encoding="utf-8")

    supplement_path = root / "tables/task6_supplementary_per_class.tex"
    lines = [
        r"\begin{longtable}{llrrrrrr}",
        r"\caption{Per-class URBAN-SED localization metrics. Intervals are 95\% soundscape-level bootstrap CIs.}\\",
        r"\toprule",
        r"Hop & Method & Class & AUROC & AP & Pointing & $n_w$ & $n_s$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule Hop & Method & Class & AUROC & AP & Pointing & $n_w$ & $n_s$ \\\midrule",
        r"\endhead",
    ]
    for row in class_rows:
        escaped_class = str(row["class"]).replace("_", r"\_")
        lines.append(
            f"{row['hop_seconds']:g} & {row['method_label']} & {escaped_class} & "
            f"{_latex(row['auroc'])} & {_latex(row['average_precision'])} & {_latex(row['pointing_game'])} & "
            f"{row['positive_windows']} & {row['positive_soundscapes']} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    supplement_path.write_text("\n".join(lines), encoding="utf-8")


def _make_figures(root: Path, summary: dict[str, Any], class_rows: list[dict[str, Any]], examples: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(METHODS))
    width = 0.34
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    metric_keys = [("auroc", "AUROC"), ("average_precision", "Average precision"), ("pointing_game", "Pointing game")]
    for axis, (key, title) in zip(axes, metric_keys):
        for offset, hop in zip((-width / 2, width / 2), HOPS):
            values = [summary["by_hop"][str(hop)]["methods"][method]["macro"][key] for method in METHODS]
            axis.bar(x + offset, values, width, label=f"hop={hop:g} s")
        axis.set_title(title)
        axis.set_xticks(x, ["$\\alpha$-CBM", "CBM+silence", "AST", "Occlusion", "Rollout"], rotation=35, ha="right")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("macro score")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.savefig(figure_dir / "task6_quantitative_comparison.png", dpi=220)
    figure.savefig(figure_dir / "task6_quantitative_comparison.pdf")
    plt.close(figure)

    selected_hop = 0.5
    matrix = np.asarray(
        [[summary["by_hop"][str(selected_hop)]["methods"][method]["by_class"][class_name]["auroc"] for class_name in summary["class_names"]] for method in METHODS],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(10.5, 3.4))
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set_xticks(np.arange(len(summary["class_names"])), [name.replace("_", " ") for name in summary["class_names"]], rotation=40, ha="right")
    axis.set_yticks(np.arange(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
    axis.set_title("Task 6 AUROC by method and event class (hop=0.5 s)")
    figure.colorbar(image, ax=axis, label="AUROC")
    figure.tight_layout()
    figure.savefig(figure_dir / "task6_concept_response_heatmap.png", dpi=220)
    figure.savefig(figure_dir / "task6_concept_response_heatmap.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(len(examples), 1, figsize=(12, 2.4 * len(examples)), squeeze=False)
    for axis, example in zip(axes[:, 0], examples):
        starts = np.asarray(example["window_starts"], dtype=float)
        ends = np.asarray(example["window_ends"], dtype=float)
        centers = (starts + ends) / 2.0
        for name, values in example["scores"].items():
            axis.plot(centers, values, marker=".", linewidth=1.0, label=name)
        for event in example["events"]:
            axis.axvspan(float(event["event_start"]), float(event["event_end"]), color="tab:orange", alpha=0.18)
        axis.set_xlim(0, 10.75)
        axis.set_ylabel("score")
        axis.set_title(f"{example['soundscape_id']} | target={example['target_class']} | {example['selection']}", loc="left", fontsize=9)
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("time (s); orange spans are annotated events")
    axes[0, 0].legend(frameon=False, ncol=5, fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(figure_dir / "task6_qualitative_examples.png", dpi=220)
    figure.savefig(figure_dir / "task6_qualitative_examples.pdf")
    plt.close(figure)


def _select_examples(root: Path, preparation: Path, class_names: list[str]) -> list[dict[str, Any]]:
    rows = load_event_window_rows(preparation / "event_windows_1s_hop_0.5.jsonl", hop_seconds=0.5)
    labels, _, _ = event_active_window_labels(rows, class_names=class_names)
    archive = {}
    for method in METHODS:
        with np.load(root / f"inference/scores_1s_hop_0.5_{method}.npz", allow_pickle=False) as data:
            archive[method] = np.asarray(data["scores"], dtype=float)
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(str(row["soundscape_id"]), []).append(index)
    candidates: list[dict[str, Any]] = []
    for soundscape_id in sorted(groups):
        indices = groups[soundscape_id]
        active_classes = np.flatnonzero(labels[indices].any(axis=0))
        if active_classes.size == 0:
            continue
        target_index = int(active_classes[0])
        target = class_names[target_index]
        best_window = indices[int(np.argmax(archive["cbm_positive_contribution"][indices, target_index]))]
        alpha_hit = bool(labels[best_window, target_index])
        active_count = int(active_classes.size)
        selection = "success" if alpha_hit and active_count == 1 else "ambiguous" if active_count > 1 else "failure"
        rank_key = float(np.max(archive["cbm_positive_contribution"][indices, target_index]))
        candidates.append({
            "soundscape_id": soundscape_id,
            "indices": indices,
            "target_index": target_index,
            "target_class": target,
            "selection": selection,
            "alpha_hit": alpha_hit,
            "active_class_count": active_count,
            "rank_key": rank_key,
        })
    selected: list[dict[str, Any]] = []
    for selection, count in (("success", 2), ("ambiguous", 2), ("failure", 2)):
        pool = sorted((item for item in candidates if item["selection"] == selection), key=lambda item: (-item["rank_key"], item["soundscape_id"]))
        selected.extend(pool[:count])
    if len(selected) < 6:
        remaining = [item for item in candidates if item not in selected]
        selected.extend(sorted(remaining, key=lambda item: item["soundscape_id"])[: 6 - len(selected)])
    result: list[dict[str, Any]] = []
    for item in selected[:6]:
        indices = item["indices"]
        target_index = item["target_index"]
        row0 = rows[indices[0]]
        result.append({
            "soundscape_id": item["soundscape_id"],
            "target_class": item["target_class"],
            "selection": item["selection"],
            "window_starts": [float(rows[index]["window_start"]) for index in indices],
            "window_ends": [float(rows[index]["window_end"]) for index in indices],
            "events": row0.get("events", []),
            "scores": {
                METHOD_LABELS[method]: archive[method][indices, target_index].tolist() for method in METHODS
            },
        })
    return result


def run(args: argparse.Namespace) -> None:
    root = args.root
    preparation = args.preparation
    class_names = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
        "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
    ]
    summary: dict[str, Any] = {
        "protocol": "Task 6 URBAN-SED temporal localization",
        "class_names": class_names,
        "bootstrap": {"resamples": 2000, "seed": 42, "unit": "soundscape"},
        "by_hop": {},
        "methods": list(METHODS),
        "null_comparison": {
            "status": "not_applicable",
            "reason": "The persisted Task-6 protocol specifies no random-pairing null; no post-hoc null test was introduced.",
        },
    }
    class_rows: list[dict[str, Any]] = []
    for hop in HOPS:
        rows = load_event_window_rows(preparation / f"event_windows_1s_hop_{hop:g}.jsonl", hop_seconds=hop)
        labels, _, soundscape_ids = event_active_window_labels(rows, class_names=class_names)
        hop_result: dict[str, Any] = {"n_windows": len(rows), "n_soundscapes": len(set(soundscape_ids)), "methods": {}}
        for method in METHODS:
            path = root / f"inference/scores_1s_hop_{hop:g}_{method}.npz"
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"missing final score artifact: {path}")
            with np.load(path, allow_pickle=False) as archive:
                scores = np.asarray(archive["scores"], dtype=float)
                archive_ids = [str(value) for value in archive["soundscape_ids"].tolist()]
                starts = np.asarray(archive["window_starts"], dtype=float)
                ends = np.asarray(archive["window_ends"], dtype=float)
                archive_classes = [str(value) for value in archive["class_names"].tolist()]
            if archive_classes != class_names or archive_ids != soundscape_ids or scores.shape != labels.shape:
                raise ValueError(f"score manifest alignment failed for {path}")
            if not np.allclose(starts, [float(row["window_start"]) for row in rows]) or not np.allclose(ends, [float(row["window_end"]) for row in rows]):
                raise ValueError(f"score timestamps do not match prepared rows for {path}")
            if not np.isfinite(scores).all():
                raise ValueError(f"non-finite score values in {path}")
            metrics = compute_task6_metrics(labels, scores, class_names, soundscape_ids, seed=42, bootstrap_resamples=2000)
            macro = _aggregate_method(metrics)
            hop_result["methods"][method] = {"metrics": metrics, "macro": macro, "by_class": metrics["by_class"]}
            class_rows.extend(_metric_row(hop, method, class_name, metrics["by_class"][class_name]) for class_name in class_names)
        summary["by_hop"][str(hop)] = hop_result

    summary["qualitative_examples"] = _select_examples(root, preparation, class_names)
    score_manifest: dict[str, Any] = {
        "protocol": "Task 6 URBAN-SED temporal localization",
        "methods": list(METHODS),
        "hops": list(HOPS),
        "files": [],
    }
    for hop in HOPS:
        rows = load_event_window_rows(preparation / f"event_windows_1s_hop_{hop:g}.jsonl", hop_seconds=hop)
        expected_shape = [len(rows), len(class_names)]
        for method in METHODS:
            path = root / f"inference/scores_1s_hop_{hop:g}_{method}.npz"
            with np.load(path, allow_pickle=False) as archive:
                scores = np.asarray(archive["scores"])
            score_manifest["files"].append({
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "hop_seconds": hop,
                "method": method,
                "shape": list(scores.shape),
                "expected_shape": expected_shape,
                "soundscape_count": len(set(str(row["soundscape_id"]) for row in rows)),
                "finite": bool(np.isfinite(scores).all()),
            })
    _dump(root / "analysis/task6_score_manifest.json", score_manifest)
    _dump(root / "analysis/task6_final_metrics.json", summary)
    _write_csv(root / "tables/task6_per_class_metrics.csv", class_rows)
    macro_rows = []
    for hop in HOPS:
        for method in METHODS:
            macro = summary["by_hop"][str(hop)]["methods"][method]["macro"]
            macro_rows.append({"hop_seconds": hop, "method": method, "method_label": METHOD_LABELS[method], **macro})
    _write_csv(root / "tables/task6_macro_metrics.csv", macro_rows)
    _write_tables(root, summary, class_rows)
    _make_figures(root, summary, class_rows, summary["qualitative_examples"])
    _dump(root / "analysis/task6_qc.json", {
        "datasets": ["URBAN-SED"],
        "hops": [0.5, 0.25],
        "methods": list(METHODS),
        "all_scores_finite": True,
        "all_score_manifests_aligned": True,
        "bootstrap_reproducible": True,
        "bootstrap_resamples": 2000,
        "gpu_work": "none in finalization",
        "qualitative_count": len(summary["qualitative_examples"]),
    })
    print(json.dumps({"status": "complete", "metrics": str(root / "analysis/task6_final_metrics.json"), "rows": len(class_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/revision/t6_urbansed"))
    parser.add_argument("--preparation", type=Path, default=Path("results/revision/t6_urbansed/preparation"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
