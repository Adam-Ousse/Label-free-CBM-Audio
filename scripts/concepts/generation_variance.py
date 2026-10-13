#!/usr/bin/env python3
"""Prepare, generate, and compare ESC-50 concept-generation replicates.

The runner keeps the canonical prompts and postprocessing in the repository.
It never falls back to another model. If the DeepSeek credential is absent it
writes the required blocked record and exits without creating fake concept sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from concept_generation_deepseek import (
    CONTRASTIVE_PROMPT,
    SYSTEM_PROMPT,
    _clean_concepts,
    concept_mentions_class,
    get_broad_prompt,
    get_contrastive_prompt,
    get_grouping_prompt,
)


DATASET = "esc50"
CANONICAL_MODEL = "deepseek-v4-flash"
CANONICAL_TEMPERATURE = 0.4
CANONICAL_TOP_P = 0.9
DEFAULT_REPLICATE_TEMPERATURES = (0.35, 0.45)
DATA_ROOT = Path("data/concept_sets/esc50/revision_variance")
RESULT_ROOT = Path("results/revision/t8_checks/generation_variance")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_config(repo_root: Path) -> dict[str, Any]:
    """Record recoverable settings and explicitly retain unresolved settings."""

    summary = _read_json(repo_root / "data/concept_sets/esc50/generation_summary.json")
    groups_payload = _read_json(
        repo_root / "data/concept_sets/esc50/contrastive_groups.json"
    )
    groups = groups_payload.get("groups", [])
    broad_size = summary.get("broad_size", 80)
    concepts_per_group = summary.get("concepts_per_group", 8)
    classes = [
        line.strip()
        for line in (repo_root / "data/esc50_classes.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return {
        "model": summary.get("model", CANONICAL_MODEL),
        "temperature": summary.get("temperature", CANONICAL_TEMPERATURE),
        "seed": "UNKNOWN",
        "top_p": 0.9,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "retry_count": 4,
        "base_url": "https://api.deepseek.com",
        "broad": {
            "num_trials": summary.get("broad_trials", 1),
            "num_concepts": broad_size,
            "max_tokens": max(512, broad_size * 12),
        },
        "contrastive": {
            "num_trials": summary.get("num_trials", 2),
            "concepts_per_group": concepts_per_group,
            "max_tokens": max(256, concepts_per_group * 24),
            "groups": groups,
        },
        "group_discovery": {
            "temperature": min(summary.get("temperature", CANONICAL_TEMPERATURE), 0.3),
            "num_trials": summary.get("group_trials", 1),
            "max_tokens": max(768, len(classes) * 32),
            "exact_prompt": get_grouping_prompt(
                DATASET, classes, max_group_size=summary.get("max_group_size", 5)
            ),
        },
        "prompts": {
            "system": SYSTEM_PROMPT,
            "broad_user_prompt": get_broad_prompt(num_concepts=broad_size),
            "contrastive_system": SYSTEM_PROMPT,
            "contrastive_user_template": CONTRASTIVE_PROMPT,
            "contrastive_rendered_prompts": [
                get_contrastive_prompt(
                    DATASET, group, num_concepts=concepts_per_group
                )
                for group in groups
            ],
        },
        "postprocessing": {
            "generator_cleaning": (
                "_clean_concepts: trim punctuation, case-insensitive dedupe, "
                "maximum six words"
            ),
            "contrastive_filter": (
                "remove concepts matching a literal class name via "
                "concept_mentions_class"
            ),
            "downstream_clap_filter": {
                "max_len": 30,
                "class_sim_cutoff": 0.85,
                "other_sim_cutoff": 0.90,
                "device": "UNKNOWN (CLI default is auto; canonical run device is not recorded)",
            },
        },
        "sources": [
            "data/concept_sets/esc50/generation_summary.json",
            "data/concept_sets/esc50/contrastive_groups.json",
            "scripts/concepts/generate_deepseek_concept_sets.py",
            "concept_generation_deepseek.py",
        ],
    }


def write_configuration_audit(repo_root: Path, result_root: Path) -> dict[str, Any]:
    config = _canonical_config(repo_root)
    _write_json(result_root / "configuration_audit.json", config)
    return config


def _blocked(repo_root: Path, result_root: Path, reason: str) -> None:
    command = (
        "cd {root} && test -n \"$DEEPSEEK_API_KEY\" && "
        "/venv/main/bin/python scripts/concepts/generation_variance.py "
        "--repo-root . --generate"
    ).format(root=repo_root)
    lines = [
        "# Concept-generation variance: BLOCKED",
        "",
        "## Missing requirement",
        "",
        reason,
        "",
        "No LLM was substituted and no replicate was fabricated.",
        "",
        "## Exact resume command",
        "",
        command,
        "",
        "The command reads DEEPSEEK_API_KEY from the process environment only; "
        "the value must not be printed or committed.",
        "",
        "## Expected output locations",
        "",
        "- data/concept_sets/esc50/revision_variance/broad/replicate_01/",
        "- data/concept_sets/esc50/revision_variance/broad/replicate_02/",
        "- data/concept_sets/esc50/revision_variance/contrastive/replicate_01/",
        "- data/concept_sets/esc50/revision_variance/contrastive/replicate_02/",
        "- results/revision/t8_checks/generation_variance/comparison_summary.json",
        "- results/revision/t8_checks/generation_variance/comparison.md",
        "",
    ]
    (result_root / "BLOCKED.md").write_text("\n".join(lines), encoding="utf-8")


def _response_dump(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if isinstance(response, dict):
        return response
    raise TypeError("DeepSeek response is not JSON serializable")


def _response_content(response_payload: dict[str, Any]) -> str:
    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("DeepSeek response has no choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("DeepSeek response content is empty")
    return content


def _parse_concepts(response_payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    payload = json.loads(_response_content(response_payload))
    if not isinstance(payload, dict) or not isinstance(payload.get("concepts"), list):
        raise ValueError("DeepSeek response JSON has no concepts list")
    raw = [str(item) for item in payload["concepts"]]
    return raw, _clean_concepts(raw)


def _prompt_record(system_prompt: str, user_prompt: str) -> dict[str, str]:
    combined = system_prompt + "\n\n" + user_prompt
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_hash": _sha256(combined),
    }


def _client(model: str, base_url: str) -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The DeepSeek client requires the repository openai dependency"
        ) from exc
    return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"].strip(), base_url=base_url)


def _request(
    client: Any,
    model: str,
    prompt: dict[str, str],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": prompt["user_prompt"]},
        ],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return _response_dump(response)


def _normalized_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _save_replicate(
    replicate_dir: Path,
    *,
    source: str,
    replicate: int,
    temperature: float,
    raw_responses: list[dict[str, Any]],
    raw_concepts: list[str],
    concepts: list[str],
    prompts: list[dict[str, str]],
    config: dict[str, Any],
) -> None:
    replicate_dir.mkdir(parents=True, exist_ok=True)
    _write_json(replicate_dir / "raw_api_responses.json", raw_responses)
    _write_json(replicate_dir / "prompts.json", prompts)
    _write_json(
        replicate_dir / "parsed_concept_set.json",
        {"source": source, "concepts": concepts},
    )
    (replicate_dir / "parsed_concepts.txt").write_text(
        "\n".join(concepts) + ("\n" if concepts else ""), encoding="utf-8"
    )
    normalized = [_normalized_phrase(concept) for concept in raw_concepts]
    _write_json(
        replicate_dir / "generation_summary.json",
        {
            "dataset": DATASET,
            "source": source,
            "replicate": replicate,
            "model": config["model"],
            "seed": "UNKNOWN",
            "temperature": temperature,
            "canonical_temperature": config["temperature"],
            "top_p": config["top_p"],
            "max_tokens": config[source]["max_tokens"],
            "response_format": config["response_format"],
            "thinking": config["thinking"],
            "timestamp_utc": _now(),
            "prompt_hash": _sha256("\n".join(p["prompt_hash"] for p in prompts)),
            "prompt_hashes": [p["prompt_hash"] for p in prompts],
            "exact_prompts_path": "prompts.json",
            "raw_response_path": "raw_api_responses.json",
            "parsed_concept_set_path": "parsed_concept_set.json",
            "raw_concept_count": len(raw_concepts),
            "duplicate_count_exact": len(raw_concepts) - len(set(raw_concepts)),
            "duplicate_count_normalized": len(raw_concepts) - len(set(normalized)),
            "vocabulary_size": len(concepts),
            "postprocessing": config["postprocessing"],
        },
    )


def generate_replicates(
    repo_root: Path,
    data_root: Path,
    config: dict[str, Any],
    temperatures: Sequence[float],
) -> None:
    load_dotenv(repo_root / ".env")
    client = _client(config["model"], config["base_url"])
    groups = config["contrastive"]["groups"]
    for replicate, temperature in enumerate(temperatures, start=1):
        broad_dir = data_root / "broad" / "replicate_{:02d}".format(replicate)
        broad_prompt = _prompt_record(
            SYSTEM_PROMPT,
            get_broad_prompt(num_concepts=config["broad"]["num_concepts"]),
        )
        broad_raw = []
        broad_concepts_raw = []
        broad_concepts = []
        for _ in range(config["broad"]["num_trials"]):
            response = _request(
                client,
                config["model"],
                broad_prompt,
                temperature,
                config["broad"]["max_tokens"],
            )
            raw, cleaned = _parse_concepts(response)
            broad_raw.append(response)
            broad_concepts_raw.extend(raw)
            broad_concepts.extend(cleaned)
        _save_replicate(
            broad_dir,
            source="broad",
            replicate=replicate,
            temperature=temperature,
            raw_responses=broad_raw,
            raw_concepts=broad_concepts_raw,
            concepts=_clean_concepts(broad_concepts),
            prompts=[broad_prompt],
            config=config,
        )

        contrastive_dir = data_root / "contrastive" / "replicate_{:02d}".format(replicate)
        raw_responses = []
        raw_concepts = []
        concepts = []
        prompts = []
        for group in groups:
            prompt = _prompt_record(
                SYSTEM_PROMPT,
                get_contrastive_prompt(
                    DATASET,
                    group,
                    num_concepts=config["contrastive"]["concepts_per_group"],
                ),
            )
            prompts.append(prompt)
            for _ in range(config["contrastive"]["num_trials"]):
                response = _request(
                    client,
                    config["model"],
                    prompt,
                    temperature,
                    config["contrastive"]["max_tokens"],
                )
                raw, cleaned = _parse_concepts(response)
                raw_responses.append(response)
                raw_concepts.extend(raw)
                concepts.extend(
                    concept
                    for concept in cleaned
                    if not concept_mentions_class(concept, group)
                )
        _save_replicate(
            contrastive_dir,
            source="contrastive",
            replicate=replicate,
            temperature=temperature,
            raw_responses=raw_responses,
            raw_concepts=raw_concepts,
            concepts=_clean_concepts(concepts),
            prompts=prompts,
            config=config,
        )


def _load_vocab(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _overlap(candidate: Iterable[str], canonical: Iterable[str]) -> dict[str, Any]:
    candidate_list = list(candidate)
    canonical_list = list(canonical)
    exact_candidate = set(candidate_list)
    exact_canonical = set(canonical_list)
    normalized_candidate = {_normalized_phrase(value) for value in candidate_list}
    normalized_canonical = {_normalized_phrase(value) for value in canonical_list}
    exact_intersection = exact_candidate & exact_canonical
    exact_union = exact_candidate | exact_canonical
    normalized_intersection = normalized_candidate & normalized_canonical
    normalized_union = normalized_candidate | normalized_canonical
    return {
        "vocabulary_size": len(exact_candidate),
        "canonical_vocabulary_size": len(exact_canonical),
        "exact_overlap_count": len(exact_intersection),
        "exact_overlap": sorted(exact_intersection, key=str.casefold),
        "exact_jaccard": len(exact_intersection) / len(exact_union)
        if exact_union
        else 0.0,
        "normalized_phrase_overlap_count": len(normalized_intersection),
        "normalized_phrase_overlap": sorted(normalized_intersection),
        "normalized_phrase_jaccard": len(normalized_intersection)
        / len(normalized_union)
        if normalized_union
        else 0.0,
        "duplicate_count_exact": len(candidate_list) - len(exact_candidate),
        "duplicate_count_normalized": len(candidate_list) - len(normalized_candidate),
    }


def _leakage_summary(repo_root: Path, concepts: list[str]) -> dict[str, Any]:
    """Use Task 2 rules without modifying or filtering the input vocabulary."""

    from scripts.concepts.leakage_audit import (
        CLAP_MODEL,
        MPNET_MODEL,
        class_name_token_matches,
        compute_similarity_scores,
        make_audit_rows,
    )

    classes = _load_vocab(repo_root / "data/esc50_classes.txt")
    lexical = [bool(class_name_token_matches(concept, classes)) for concept in concepts]
    try:
        scores = compute_similarity_scores(
            concepts,
            classes,
            mpnet_model=MPNET_MODEL,
            clap_model=CLAP_MODEL,
            device="cpu",
        )
        rows = make_audit_rows(DATASET, "replicate", concepts, classes, scores)
        return {
            "status": "complete",
            "rule": "existing scripts/concepts/leakage_audit.py",
            "lexical_flagged_count": sum(lexical),
            "flagged_count": sum(bool(row["flagged"]) for row in rows),
            "mpnet_model": MPNET_MODEL,
            "clap_model": CLAP_MODEL,
        }
    except Exception as exc:
        return {
            "status": "similarity_unavailable",
            "rule": "existing scripts/concepts/leakage_audit.py lexical rule only",
            "lexical_flagged_count": sum(lexical),
            "flagged_count": "UNKNOWN",
            "error_type": type(exc).__name__,
        }


def compare_replicates(
    repo_root: Path, data_root: Path, result_root: Path
) -> dict[str, Any]:
    results: dict[str, Any] = {"dataset": DATASET, "sources": {}}
    for source in ("broad", "contrastive"):
        canonical = _load_vocab(
            repo_root / "data/concept_sets/esc50/concepts_{}.txt".format(source)
        )
        rows = []
        for replicate_dir in sorted((data_root / source).glob("replicate_*")):
            parsed = replicate_dir / "parsed_concepts.txt"
            summary_path = replicate_dir / "generation_summary.json"
            if not parsed.is_file() or not summary_path.is_file():
                continue
            concepts = _load_vocab(parsed)
            row = _overlap(concepts, canonical)
            row.update(
                {
                    "replicate": replicate_dir.name,
                    "generation": _read_json(summary_path),
                    "leakage": _leakage_summary(repo_root, concepts),
                }
            )
            rows.append(row)
        results["sources"][source] = {
            "canonical_path": "data/concept_sets/esc50/concepts_{}.txt".format(source),
            "replicates": rows,
        }
    _write_json(result_root / "comparison_summary.json", results)
    lines = ["# ESC-50 concept-generation variance", "", "No CBM training was run.", ""]
    for source, payload in results["sources"].items():
        lines += [
            "## {}".format(source),
            "",
            "| Replicate | Vocab | Exact overlap | Exact Jaccard | Normalized overlap | Normalized Jaccard | Leakage flagged |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in payload["replicates"]:
            leakage = row["leakage"].get("flagged_count", "UNKNOWN")
            lines.append(
                "| {replicate} | {vocabulary_size} | {exact_overlap_count} | "
                "{exact_jaccard:.6f} | {normalized_phrase_overlap_count} | "
                "{normalized_phrase_jaccard:.6f} | {leakage} |".format(
                    leakage=leakage, **row
                )
            )
        lines.append("")
    (result_root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--temperatures",
        nargs=2,
        type=float,
        default=DEFAULT_REPLICATE_TEMPERATURES,
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Call DeepSeek and write four replicates.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare existing replicate outputs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    data_root = (repo_root / args.data_root).resolve()
    result_root = (repo_root / args.result_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    config = write_configuration_audit(repo_root, result_root)
    load_dotenv(repo_root / ".env")

    if args.generate:
        if not os.getenv("DEEPSEEK_API_KEY"):
            _blocked(
                repo_root,
                result_root,
                "DEEPSEEK_API_KEY is not available in the process environment "
                "or repository .env.",
            )
            print(
                "BLOCKED: DEEPSEEK_API_KEY is unavailable; see "
                + str(result_root / "BLOCKED.md")
            )
            return 2
        if config["model"] != CANONICAL_MODEL:
            raise RuntimeError(
                "Canonical model is not deepseek-v4-flash; refusing to substitute a model"
            )
        generate_replicates(repo_root, data_root, config, args.temperatures)
    if args.compare:
        compare_replicates(repo_root, data_root, result_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
