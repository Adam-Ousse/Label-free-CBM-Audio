import json
from types import SimpleNamespace

from concept_generation_deepseek import (
    GROUP_SYSTEM_PROMPT,
    concept_mentions_class,
    discover_confusion_groups,
    generate_broad_concepts,
    generate_contrastive_concepts,
    get_broad_prompt,
    get_contrastive_prompt,
    get_grouping_prompt,
)
from generate_deepseek_concept_sets import parse_args, save_candidate_outputs


class FakeGenerator:
    def generate_json(self, prompt, system_prompt, **kwargs):
        assert system_prompt == GROUP_SYSTEM_PROMPT
        return {
            "groups": [
                ["air conditioner", "engine_idling"],
                ["drilling", "jackhammer", "engine idling"],
                ["unknown class", "siren"],
            ]
        }

    def generate_concepts(self, prompt, **kwargs):
        if "dataset-independent vocabulary" in prompt:
            return ["high-pitched", "rough", "metallic"]
        return [
            "rapid repetitive impacts",
            "engine idling",
            "metallic resonance",
        ]


def test_broad_prompt_is_dataset_independent():
    prompt = get_broad_prompt(60)
    assert "urbansound8k" not in prompt.casefold()
    assert "target class names" in prompt
    assert "pitch variation" in prompt
    assert "acoustic production mechanism" in prompt


def test_contrastive_prompts_only_propose_acoustic_dimensions():
    grouping = get_grouping_prompt("urbansound8k", ["drilling", "jackhammer"])
    contrastive = get_contrastive_prompt(
        "urbansound8k", ["drilling", "jackhammer"], num_concepts=7
    )
    assert "A class may\nbelong to multiple groups" in grouping
    assert "Do not propose concepts" in grouping
    assert "Generate 7 general audible attributes" in contrastive
    assert "claims\nthat a property is actually present" in contrastive


def test_group_discovery_canonicalizes_labels_and_allows_overlap(tmp_path):
    classes = [
        "air_conditioner",
        "engine_idling",
        "drilling",
        "jackhammer",
        "siren",
    ]
    groups = discover_confusion_groups(
        "urbansound8k",
        classes,
        FakeGenerator(),
        tmp_path / "groups.json",
        resume=False,
    )
    assert groups == [
        ["air_conditioner", "engine_idling"],
        ["drilling", "jackhammer", "engine_idling"],
    ]
    assert sum("engine_idling" in group for group in groups) == 2


def test_contrastive_generation_rejects_literal_class_names(tmp_path):
    records = generate_contrastive_concepts(
        "urbansound8k",
        [["drilling", "engine_idling"]],
        FakeGenerator(),
        tmp_path / "contrastive.json",
        num_trials=1,
        resume=False,
    )
    assert records[0]["concepts"] == [
        "rapid repetitive impacts",
        "metallic resonance",
    ]
    assert concept_mentions_class("steady engine idling", ["engine_idling"])
    assert not concept_mentions_class("steady low-frequency hum", ["engine_idling"])


def test_broad_vocabulary_is_resumable(tmp_path):
    path = tmp_path / "broad.json"
    first = generate_broad_concepts(
        FakeGenerator(), path, num_trials=1, num_concepts=20, resume=False
    )
    second = generate_broad_concepts(
        FakeGenerator(), path, num_trials=1, num_concepts=20, resume=True
    )
    assert first == second == ["high-pitched", "rough", "metallic"]


def test_cli_preserves_legacy_default_and_accepts_singular_dataset():
    legacy = parse_args([])
    assert legacy.mode == "legacy"
    assert legacy.datasets == ["esc50", "urbansound8k", "cremad"]

    extended = parse_args(
        ["--dataset", "urbansound8k", "--mode", "contrastive", "--stage", "generate"]
    )
    assert extended.datasets == ["urbansound8k"]
    assert extended.mode == "contrastive"


def test_candidate_exports_include_flat_files_and_provenance(tmp_path):
    init_dir = tmp_path / "init"
    output_dir = tmp_path / "out"
    init_dir.mkdir()
    lf_payloads = {
        "important": {"drilling": ["rapid pulses", "metallic resonance"]},
        "superclass": {"drilling": ["mechanical noise"]},
        "around": {"drilling": ["traffic rumble"]},
    }
    for prompt_type, payload in lf_payloads.items():
        (init_dir / "deepseek_urbansound8k_{}.json".format(prompt_type)).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    (init_dir / "deepseek_broad.json").write_text(
        json.dumps({"concepts": ["rough", "high-pitched"]}), encoding="utf-8"
    )
    (init_dir / "deepseek_urbansound8k_contrastive.json").write_text(
        json.dumps(
            {
                "groups": {
                    "drilling|jackhammer": {
                        "classes": ["drilling", "jackhammer"],
                        "concepts": ["rapid pulses", "strong transient attacks"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        init_dir=str(init_dir),
        output_dir=str(output_dir),
        mode="all",
        model="test-model",
        temperature=0.4,
        num_trials=2,
        group_trials=1,
        broad_trials=1,
        broad_size=80,
        concepts_per_group=8,
        max_group_size=5,
    )
    save_candidate_outputs(
        "urbansound8k", args, required_sources=("lf", "broad", "contrastive")
    )

    dataset_dir = output_dir / "urbansound8k"
    for name in (
        "concepts_lf.txt",
        "concepts_broad.txt",
        "concepts_contrastive.txt",
        "concepts_all.txt",
        "concepts_metadata.json",
    ):
        assert (dataset_dir / name).exists()

    all_concepts = (dataset_dir / "concepts_all.txt").read_text(encoding="utf-8").splitlines()
    assert all_concepts.count("rapid pulses") == 1
    metadata = json.loads(
        (dataset_dir / "concepts_metadata.json").read_text(encoding="utf-8")
    )
    contrastive = next(
        record
        for record in metadata
        if record["concept"] == "strong transient attacks"
    )
    assert contrastive["source"] == "contrastive"
    assert contrastive["classes"] == ["drilling", "jackhammer"]
