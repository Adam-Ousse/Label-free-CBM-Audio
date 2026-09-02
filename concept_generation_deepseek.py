"""DeepSeek-backed candidate concept generation for audio CBMs.

The LLM only proposes candidates. Dataset grounding and projectability are left
to the existing LF-CBM/CLAP training pipeline.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv


SYSTEM_PROMPT = """You generate concise, directly audible concepts for an audio
concept bottleneck model. Return valid JSON only, using this exact shape:
{"concepts": ["concept one", "concept two"]}
Do not include explanations, Markdown, or reasoning."""

GROUP_SYSTEM_PROMPT = """You organize audio classes by likely acoustic confusion.
Return valid JSON only, using this exact shape:
{"groups": [["class_one", "class_two"], ["class_two", "class_three"]]}
Use only class names supplied by the user. Do not include explanations, Markdown,
scores, concepts, or reasoning."""

ENVIRONMENTAL_PROMPTS = {
    "important": """Dataset: {dataset}. Target class: {class_name}.
Generate 5 directly audible properties of the target sound itself. Describe useful
temporal, pitch, timbre, intensity, rhythm, texture, or acoustic-event cues.
Use concrete phrases of 1-3 words. Do not use the class name, a synonym of it,
visual information, locations, causes that cannot be heard, co-occurring sounds,
or generic labels such as 'sound', 'acoustic event', and 'sound source'.""",
    "superclass": """Dataset: {dataset}. Target class: {class_name}.
Generate 3 useful audible sound categories at different levels of abstraction.
Use concrete phrases of 1-3 words. Each category must convey acoustic information.
Do not use the class name, visual/contextual information, or generic labels such
as 'sound', 'acoustic event', 'auditory event', and 'sound source'.""",
    "around": """Dataset: {dataset}. Target class: {class_name}.
Generate 4 distinct sounds that may genuinely co-occur with the target in a real
recording. Use concrete, directly audible phrases of 1-3 words. Do not include the
target itself, synonyms, visual information, or vague background/context labels.
These are contextual concepts, not properties of the target.""",
}

SPEECH_EMOTION_PROMPTS = {
    "important": """Dataset: CREMA-D acted emotional speech. Target emotion: {class_name}.
Generate 5 acoustic or prosodic properties of the speaker's voice that could help
recognize this emotion. Consider pitch, pitch variation, intensity, speech rate,
pauses, articulation, breathiness, roughness, vocal tension, rhythm, and spectral
quality. Use concrete phrases of 1-3 words. Do not use the emotion name or synonyms.
Do not generate situations, meanings, background sounds, bodily actions, music,
weather, facial expressions, or stereotypical semantic associations.""",
    "superclass": """Dataset: CREMA-D acted emotional speech. Target emotion: {class_name}.
Generate 3 broader but still audible voice-quality or prosodic categories relevant
to this emotion. Use concrete phrases of 1-3 words. Do not use emotion taxonomy,
psychological states, situations, background sounds, or generic phrases such as
'affective sound', 'auditory signal', and 'human vocalization'.""",
}

BROAD_PROMPT = """Create a dataset-independent vocabulary of short, atomic auditory
properties for audio concept bottleneck models. Generate {num_concepts} distinct
concepts spanning all of these perceptual dimensions:

- loudness and energy
- pitch and pitch variation
- spectral balance and bandwidth
- tonality, harmonicity, and inharmonicity
- timbre, texture, roughness, and sharpness
- continuity, periodicity, repetition, and rhythm
- attack, decay, modulation, impulsiveness, and temporal density
- reverberation, spatial impression, and perceived distance
- acoustic production mechanism described as an audible property

Use one interpretable property per concept and preferably 1-3 words. Concepts such
as "high-pitched", "broadband", "rough", "metallic", "continuous", "sharp attack",
and "reverberant" illustrate the desired level of abstraction.

Do not use any dataset or target class names. Do not name objects, sound sources,
events, actions, emotions, scenes, or visual information. Do not write sentences,
definitions, explanations, or long descriptions. Propose potentially useful
dimensions only; do not claim that any concept occurs in a dataset."""

GROUPING_PROMPT = """Dataset: {dataset}.
Group the target classes below into acoustically confusable groups before concept
generation. A group should contain 2-{max_group_size} classes that could be hard to
distinguish from audio alone because they share audible structure. A class may
belong to multiple groups. Prefer compact, meaningful groups over all class pairs,
and do not force unrelated classes together.

Target classes (copy these spellings exactly):
{class_list}

This step only discovers confusion groups. Do not propose concepts or decide what
is present in the recordings."""

CONTRASTIVE_PROMPT = """Dataset: {dataset}.
The following target classes are likely to be acoustically confusable:
{class_list}

Generate {num_concepts} general audible attributes that would help distinguish
members of this group. Do not generate class synonyms or descriptions tied to only
one named class. Instead, identify acoustic dimensions on which the sounds may
differ: pitch, spectrum, bandwidth, timbre, temporal structure, periodicity,
repetition, attack/decay, modulation, energy, texture, spatial properties, or an
audible production mechanism.

Each concept must be directly audible, express one interpretable property, use
preferably 1-4 words, and make sense independently of the class names. Do not use,
quote, paraphrase, or embed any class name. Do not use object identities, event
labels, emotions, visual information, recording context, explanations, or claims
that a property is actually present in the dataset."""


def get_prompt(dataset, prompt_type, class_name):
    prompts = (
        SPEECH_EMOTION_PROMPTS
        if dataset.lower().replace("-", "") == "cremad"
        else ENVIRONMENTAL_PROMPTS
    )
    if prompt_type not in prompts:
        raise ValueError(
            "Prompt type '{}' is not valid for {}; choose: {}".format(
                prompt_type, dataset, ", ".join(prompts)
            )
        )
    return prompts[prompt_type].format(dataset=dataset, class_name=class_name)


def get_broad_prompt(num_concepts=80):
    """Return the class-independent broad auditory vocabulary prompt."""
    return BROAD_PROMPT.format(num_concepts=num_concepts)


def get_grouping_prompt(dataset, classes, max_group_size=5):
    """Return the acoustic confusion-group discovery prompt."""
    return GROUPING_PROMPT.format(
        dataset=dataset,
        max_group_size=max_group_size,
        class_list="\n".join("- {}".format(name) for name in classes),
    )


def get_contrastive_prompt(dataset, classes, num_concepts=8):
    """Return the group-wise contrastive acoustic-dimension prompt."""
    return CONTRASTIVE_PROMPT.format(
        dataset=dataset,
        num_concepts=num_concepts,
        class_list="\n".join("- {}".format(name) for name in classes),
    )


def _clean_concepts(values, max_words=6):
    cleaned = []
    seen = set()
    for value in values:
        concept = str(value).strip().strip(".,;:-")
        key = concept.casefold()
        if concept and key not in seen and len(concept.split()) <= max_words:
            cleaned.append(concept)
            seen.add(key)
    return cleaned


def _normalized_label(value):
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def concept_mentions_class(concept, classes):
    """Conservatively detect a literal class name embedded in a concept."""
    normalized_concept = " {} ".format(_normalized_label(concept))
    return any(
        " {} ".format(_normalized_label(class_name)) in normalized_concept
        for class_name in classes
        if _normalized_label(class_name)
    )


class DeepSeekGenerator:
    """OpenAI-compatible DeepSeek adapter with JSON validation and retries."""

    def __init__(
        self,
        model=None,
        api_key=None,
        base_url="https://api.deepseek.com",
        max_retries=4,
    ):
        load_dotenv()
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is missing. Set it in the environment or a local .env file."
            )
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The DeepSeek client requires the 'openai' package. Install the "
                "project requirements before running generation."
            ) from exc
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.max_retries = max_retries
        self.client = OpenAI(api_key=api_key.strip(), base_url=base_url)

    def generate_json(
        self,
        prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=256,
        temperature=0.4,
        top_p=0.9,
        required_list_key=None,
    ):
        """Generate and decode one JSON object, retrying transient/format errors."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                payload = json.loads(response.choices[0].message.content or "")
                if not isinstance(payload, dict):
                    raise ValueError("response JSON is not an object")
                if required_list_key is not None and not isinstance(
                    payload.get(required_list_key), list
                ):
                    raise ValueError(
                        "response JSON has no '{}' list".format(required_list_key)
                    )
                return payload
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            "DeepSeek generation failed after {} attempts: {}".format(
                self.max_retries, last_error
            )
        )

    def generate_concepts(
        self, prompt, max_tokens=256, temperature=0.4, top_p=0.9
    ):
        payload = self.generate_json(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            required_list_key="concepts",
        )
        concepts = payload.get("concepts")
        if not isinstance(concepts, list):
            raise ValueError("response JSON has no 'concepts' list")
        cleaned = _clean_concepts(concepts)
        if not cleaned:
            raise ValueError("response contained no usable concepts")
        return cleaned

    def generate(self, prompt, max_new_tokens=256, temperature=0.4, top_p=0.9, enable_thinking=False):
        """Compatibility method matching LocalQwenGenerator."""
        del enable_thinking
        return "\n".join(
            self.generate_concepts(prompt, max_new_tokens, temperature, top_p)
        )


def _load_json_if_resuming(save_path, resume, default):
    save_path = Path(save_path)
    if resume and save_path.exists():
        with save_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return default


def _save_json(save_path, payload):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def generate_dataset_concepts(
    dataset,
    classes,
    prompt_type,
    generator,
    save_path,
    num_trials=2,
    resume=True,
    temperature=0.4,
):
    """Generate vanilla LF-CBM per-class concepts and checkpoint each class."""
    results = _load_json_if_resuming(save_path, resume, {})
    for index, class_name in enumerate(classes):
        if resume and class_name in results and results[class_name]:
            print("[{}/{}] skip {}".format(index + 1, len(classes), class_name))
            continue
        print("[{}/{}] generate {}".format(index + 1, len(classes), class_name))
        concepts = []
        for _ in range(num_trials):
            concepts.extend(
                generator.generate_concepts(
                    get_prompt(dataset, prompt_type, class_name),
                    temperature=temperature,
                )
            )
        results[class_name] = list(dict.fromkeys(concepts))
        _save_json(save_path, results)
    return results


def generate_broad_concepts(
    generator,
    save_path,
    num_trials=1,
    num_concepts=80,
    resume=True,
    temperature=0.4,
):
    """Generate and cache a dataset-independent perceptual vocabulary."""
    cached = _load_json_if_resuming(save_path, resume, {})
    if isinstance(cached, dict) and cached.get("concepts"):
        print("[broad] reuse {} concepts from {}".format(len(cached["concepts"]), save_path))
        return _clean_concepts(cached["concepts"])

    concepts = []
    for trial in range(num_trials):
        print("[broad] generation trial {}/{}".format(trial + 1, num_trials))
        concepts.extend(
            generator.generate_concepts(
                get_broad_prompt(num_concepts=num_concepts),
                max_tokens=max(512, num_concepts * 12),
                temperature=temperature,
            )
        )
    concepts = _clean_concepts(concepts)
    _save_json(save_path, {"concepts": concepts})
    return concepts


def _canonicalize_groups(raw_groups, classes, max_group_size=5):
    by_normalized = {_normalized_label(name): name for name in classes}
    groups = []
    seen = set()
    if not isinstance(raw_groups, list):
        return groups

    for raw_group in raw_groups:
        if isinstance(raw_group, dict):
            raw_group = raw_group.get("classes", [])
        if not isinstance(raw_group, list):
            continue
        group = []
        for value in raw_group:
            canonical = by_normalized.get(_normalized_label(value))
            if canonical is not None and canonical not in group:
                group.append(canonical)
        group = group[:max_group_size]
        key = tuple(sorted(group, key=str.casefold))
        if len(group) >= 2 and key not in seen:
            groups.append(group)
            seen.add(key)
    return groups


def discover_confusion_groups(
    dataset,
    classes,
    generator,
    save_path,
    num_trials=1,
    max_group_size=5,
    resume=True,
    temperature=0.2,
):
    """Ask DeepSeek for overlapping, group-wise acoustic confusions."""
    cached = _load_json_if_resuming(save_path, resume, {})
    if isinstance(cached, dict) and cached.get("groups"):
        groups = _canonicalize_groups(cached["groups"], classes, max_group_size)
        if groups:
            print("[contrastive] reuse {} confusion groups from {}".format(len(groups), save_path))
            return groups

    groups = []
    for trial in range(num_trials):
        print("[contrastive] group discovery trial {}/{}".format(trial + 1, num_trials))
        payload = generator.generate_json(
            get_grouping_prompt(dataset, classes, max_group_size=max_group_size),
            system_prompt=GROUP_SYSTEM_PROMPT,
            max_tokens=max(768, len(classes) * 32),
            temperature=temperature,
            required_list_key="groups",
        )
        groups.extend(
            _canonicalize_groups(payload.get("groups"), classes, max_group_size)
        )
    groups = _canonicalize_groups(groups, classes, max_group_size)
    if not groups:
        raise ValueError("DeepSeek returned no valid acoustic confusion groups")
    _save_json(save_path, {"dataset": dataset, "groups": groups})
    return groups


def generate_contrastive_concepts(
    dataset,
    groups,
    generator,
    save_path,
    num_trials=2,
    concepts_per_group=8,
    resume=True,
    temperature=0.4,
    forbidden_classes=None,
):
    """Generate general acoustic dimensions jointly for every confusion group."""
    cached = _load_json_if_resuming(save_path, resume, {})
    records = cached.get("groups", {}) if isinstance(cached, dict) else {}

    for index, group in enumerate(groups):
        group_key = "|".join(group)
        if resume and group_key in records and records[group_key].get("concepts"):
            print("[contrastive {}/{}] skip {}".format(index + 1, len(groups), group_key))
            continue
        print("[contrastive {}/{}] generate {}".format(index + 1, len(groups), group_key))
        concepts = []
        for _ in range(num_trials):
            concepts.extend(
                generator.generate_concepts(
                    get_contrastive_prompt(
                        dataset, group, num_concepts=concepts_per_group
                    ),
                    max_tokens=max(256, concepts_per_group * 24),
                    temperature=temperature,
                )
            )
        concepts = [
            concept
            for concept in _clean_concepts(concepts)
            if not concept_mentions_class(concept, forbidden_classes or group)
        ]
        if not concepts:
            raise ValueError(
                "DeepSeek returned no usable contrastive concepts for group: {}".format(
                    ", ".join(group)
                )
            )
        records[group_key] = {"classes": list(group), "concepts": concepts}
        _save_json(save_path, {"dataset": dataset, "groups": records})

    return [records["|".join(group)] for group in groups]
