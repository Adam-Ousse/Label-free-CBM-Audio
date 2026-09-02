#!/usr/bin/env python3
"""Generate a larger, speech-targeted CREMA-D concept vocabulary with DeepSeek.

This is an isolated follow-up experiment. It does not change the baseline prompt
families or write into data/concept_sets/cremad.
"""

import argparse
import json
import re
from pathlib import Path

import torch

import concept_pipeline
from concept_generation_deepseek import DeepSeekGenerator


CLASSES = ("anger", "disgust", "fear", "happy", "neutral", "sad")
FAMILIES = {
    "prosody": """Generate {count} short atomic acoustic properties of acted speech that could help distinguish the target emotion class {target} from the other CREMA-D classes. Concentrate on pitch level, pitch range, pitch contour, loudness, energy variation, speech rate, rhythm, pausing, and syllabic stress. Use properties of the voice, not emotion words. Return diverse opposing levels and dynamic patterns, not synonyms.""",
    "voice_quality": """Generate {count} short atomic voice-quality properties that could help distinguish the target emotion class {target} from other acted speech. Cover phonation, breathiness, roughness, vocal tension, creakiness, voicing stability, resonance, nasality, spectral tilt, brightness, and articulation. Prefer plain audible phrases that CLAP can understand. Do not name or paraphrase emotions.""",
    "delivery": """Generate {count} short atomic temporal-delivery properties that could help distinguish the target emotion class {target} from other acted speech. Cover onset strength, attack, pause density, continuity, phrase-final contour, pitch or energy modulation, clipped versus sustained syllables, trembling, and within-utterance change. Each item must be directly audible and independent of the emotion name.""",
}

BROAD_PROMPT = """Generate {count} short atomic perceptual concepts for describing human speech recordings without naming any emotion. Build a balanced speech-acoustics vocabulary spanning pitch level and range, intonation contours, energy and dynamic range, timing and pauses, articulation, phonation, breathiness, roughness, vocal tension, creakiness, resonance, nasality, spectral balance, voicing stability, tremor, syllabic stress, and phrase-final dynamics. Include useful opposites. Prefer 2-5 word natural phrases such as narrow pitch range, rising final pitch, tense phonation, breathy voice, clipped syllables, weak vocal onset, or long silent pauses. Do not use objects, meanings, demographics, personality, visual cues, emotion names, or long descriptions."""

GROUPS = (
    ("anger", "fear", "happy"),
    ("disgust", "neutral", "sad"),
    ("anger", "disgust", "neutral"),
    ("fear", "happy", "neutral"),
    ("fear", "neutral", "sad"),
    ("happy", "neutral", "sad"),
)

ABLATION_VARIANTS = {
    "lf": {"lf_targeted"},
    "lf_broad": {"lf_targeted", "speech_broad"},
    "lf_contrastive": {"lf_targeted", "contrastive_targeted"},
    "full": {"lf_targeted", "speech_broad", "contrastive_targeted"},
}

CONTRAST_PROMPT = """CREMA-D confusion group: {classes}. Generate {count} short atomic audible dimensions on which these acted-speech classes can differ. Focus on pitch level/range/contour, energy and its variation, spectral tilt or brightness, voicing stability, breathiness, roughness, tension, resonance, articulation, pauses, rate, stress, onset/offset, and within-utterance modulation. The output concepts must be general acoustic coordinates, not descriptions or synonyms of a named emotion. Use 2-5 words and include dimensions that can distinguish classes sharing the same arousal or pitch level."""

BANNED = {
    "anger", "angry", "disgust", "disgusted", "fear", "fearful", "happy",
    "happiness", "joy", "joyful", "neutral", "sad", "sadness", "emotion",
    "emotional",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--class-trials", type=int, default=3)
    parser.add_argument("--broad-trials", type=int, default=3)
    parser.add_argument("--contrast-trials", type=int, default=3)
    parser.add_argument("--class-count", type=int, default=14)
    parser.add_argument("--broad-count", type=int, default=120)
    parser.add_argument("--contrast-count", type=int, default=18)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--export-existing",
        action="store_true",
        help="Only rebuild source-ablation text files from concepts_metadata.json; do not call DeepSeek.",
    )
    return parser.parse_args()


def clean(values):
    output, seen = [], set()
    for value in values:
        value = re.sub(r"\s+", " ", str(value).strip().strip(".,;:-")).lower()
        words = re.findall(r"[a-z]+", value)
        if not value or not (1 <= len(words) <= 6) or len(value) > 52:
            continue
        if any(word in BANNED for word in words):
            continue
        if value in {"voice", "speech", "spoken voice", "human voice"}:
            continue
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def load(path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_trials(generator, prompt, trials, temperature, max_tokens=1024):
    concepts = []
    for trial in range(trials):
        print("  trial {}/{}".format(trial + 1, trials), flush=True)
        concepts.extend(generator.generate_concepts(prompt, max_tokens=max_tokens, temperature=temperature))
    return clean(concepts)


def add_record(records, concept, source, family, classes):
    key = concept.casefold()
    record = records.setdefault(key, {"concept": concept, "sources": [], "families": [], "classes": []})
    if source not in record["sources"]:
        record["sources"].append(source)
    if family not in record["families"]:
        record["families"].append(family)
    for name in classes:
        if name not in record["classes"]:
            record["classes"].append(name)


def export_ablation_sets(root, metadata):
    """Export source subsets in the canonical full-vocabulary filter order."""
    output_dir = root / "source_ablations_canonical"
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = root / "concepts_atomic_filtered.txt"
    if not canonical_path.exists():
        raise FileNotFoundError(canonical_path)
    canonical = [value for value in canonical_path.read_text(encoding="utf-8").splitlines() if value]
    records_by_key = {record["concept"].casefold(): record for record in metadata}
    ordered_records = []
    for concept in canonical:
        record = records_by_key.get(concept.casefold())
        if record is None or not record.get("retained_by_text_filter"):
            raise ValueError("Missing retained provenance for canonical concept: {}".format(concept))
        ordered_records.append(record)
    manifest = {
        "selection": "A concept is included when any of its provenance sources belongs to the variant.",
        "ordering": "Every source subset preserves concepts_atomic_filtered.txt order.",
        "variants": {},
    }
    for name, allowed_sources in ABLATION_VARIANTS.items():
        selected = [
            record for record in ordered_records
            if allowed_sources.intersection(record.get("sources", []))
        ]
        atomic = [record["concept"] for record in selected]
        grounded = [record.get("grounding_text", "a voice with {}".format(record["concept"])) for record in selected]
        concept_pipeline.save_concept_text(str(output_dir / "cremad_targeted_{}_canonical_atomic.txt".format(name)), atomic)
        concept_pipeline.save_concept_text(str(output_dir / "cremad_targeted_{}_canonical_grounded.txt".format(name)), grounded)
        manifest["variants"][name] = {
            "sources": sorted(allowed_sources),
            "input_concepts": len(selected),
            "atomic_file": "cremad_targeted_{}_canonical_atomic.txt".format(name),
            "grounded_file": "cremad_targeted_{}_canonical_grounded.txt".format(name),
        }
    save(output_dir / "manifest.json", manifest)
    return manifest


def main():
    args = parse_args()
    root = args.output_root
    if args.export_existing:
        metadata_path = root / "concepts_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        manifest = export_ablation_sets(root, load(metadata_path, []))
        print("exported source ablations:", {
            name: row["input_concepts"] for name, row in manifest["variants"].items()
        })
        return
    checkpoints = root / "init"
    root.mkdir(parents=True, exist_ok=True)
    generator = DeepSeekGenerator()
    records = {}

    class_path = checkpoints / "class_conditioned.json"
    class_payload = {} if args.restart else load(class_path, {})
    for family, template in FAMILIES.items():
        family_payload = class_payload.setdefault(family, {})
        for target in CLASSES:
            if target not in family_payload or not family_payload[target]:
                print("[class] {} / {}".format(family, target), flush=True)
                family_payload[target] = generate_trials(
                    generator,
                    template.format(count=args.class_count, target=target),
                    args.class_trials,
                    args.temperature,
                )
                save(class_path, class_payload)
            for concept in family_payload[target]:
                add_record(records, concept, "lf_targeted", family, [target])

    broad_path = checkpoints / "speech_broad.json"
    broad_payload = {} if args.restart else load(broad_path, {})
    if not broad_payload.get("concepts"):
        print("[broad speech taxonomy]", flush=True)
        broad_payload = {"concepts": generate_trials(
            generator,
            BROAD_PROMPT.format(count=args.broad_count),
            args.broad_trials,
            args.temperature,
            # A 120-item JSON array can exceed 1,500 tokens; leave enough room
            # to avoid a syntactically truncated response.
            max_tokens=max(4096, args.broad_count * 32),
        )}
        save(broad_path, broad_payload)
    for concept in broad_payload["concepts"]:
        add_record(records, concept, "speech_broad", "speech_taxonomy", [])

    contrast_path = checkpoints / "contrastive.json"
    contrast_payload = {} if args.restart else load(contrast_path, {})
    groups_payload = contrast_payload.setdefault("groups", {})
    for group in GROUPS:
        key = "|".join(group)
        if key not in groups_payload or not groups_payload[key].get("concepts"):
            print("[contrastive] {}".format(key), flush=True)
            groups_payload[key] = {
                "classes": list(group),
                "concepts": generate_trials(
                    generator,
                    CONTRAST_PROMPT.format(classes=", ".join(group), count=args.contrast_count),
                    args.contrast_trials,
                    args.temperature,
                ),
            }
            save(contrast_path, contrast_payload)
        for concept in groups_payload[key]["concepts"]:
            add_record(records, concept, "contrastive_targeted", "groupwise", list(group))

    metadata = list(records.values())
    atomic = [record["concept"] for record in metadata]
    concept_pipeline.save_concept_text(str(root / "concepts_atomic_raw.txt"), atomic)
    save(root / "concepts_metadata_raw.json", metadata)

    print("[existing LF-CBM text filters] {} candidates".format(len(atomic)), flush=True)
    filtered = concept_pipeline.filter_concepts(
        atomic,
        list(CLASSES),
        max_len=52,
        class_sim_cutoff=0.92,
        other_sim_cutoff=0.94,
        device=args.device,
        print_prob=0.0,
    )
    filtered = concept_pipeline.dedupe_case_insensitive(filtered)
    retained = {value.casefold() for value in filtered}
    grounded = ["a voice with {}".format(value) for value in filtered]
    for record in metadata:
        record["retained_by_text_filter"] = record["concept"].casefold() in retained
        record["grounding_text"] = "a voice with {}".format(record["concept"])
    concept_pipeline.save_concept_text(str(root / "concepts_atomic_filtered.txt"), filtered)
    concept_pipeline.save_concept_text(str(root / "concepts_grounded_filtered.txt"), grounded)
    save(root / "concepts_metadata.json", metadata)
    ablation_manifest = export_ablation_sets(root, metadata)
    save(root / "generation_summary.json", {
        "dataset": "cremad",
        "deepseek_model": generator.model,
        "class_trials": args.class_trials,
        "broad_trials": args.broad_trials,
        "contrast_trials": args.contrast_trials,
        "temperature": args.temperature,
        "raw_unique": len(atomic),
        "text_filtered": len(filtered),
        "grounding_template": "a voice with {concept}",
        "text_filter": {"max_len": 52, "class_sim_cutoff": 0.92, "other_sim_cutoff": 0.94},
        "groups": [list(group) for group in GROUPS],
    })
    print("completed: {} raw -> {} text-filtered".format(len(atomic), len(filtered)))


if __name__ == "__main__":
    main()
