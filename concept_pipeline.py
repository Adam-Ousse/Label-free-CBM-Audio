import json
import os

import conceptset_utils
import data_utils


def load_classes(dataset):
    return data_utils.get_dataset_classes(dataset)


def save_json(path, payload):
    save_dir = os.path.dirname(path)
    if len(save_dir) > 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_concept_text(path, concepts):
    if len(concepts) == 0:
        raise ValueError("Cannot save empty concept list")

    save_dir = os.path.dirname(path)
    if len(save_dir) > 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(path, "w", encoding="utf-8") as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write("\n" + concept)


def generate_and_save_prompt_concepts(
    dataset,
    prompt_type,
    generator,
    save_path,
    num_trials=2,
    max_new_tokens=196,
    temperature=0.7,
    top_p=0.9,
):
    raise RuntimeError(
        "generation was simplified: use LocalQwenGenerator directly in GPT_initial_concepts.ipynb to prompt, loop, and save"
    )


def merge_prompt_dicts(prompt_dicts):
    concepts = set()
    for prompt_dict in prompt_dicts:
        for values in prompt_dict.values():
            concepts.update(set(values))
    return concepts


def filter_concepts(concepts, classes, max_len, class_sim_cutoff, other_sim_cutoff, device="cuda", print_prob=0):
    concepts = conceptset_utils.remove_too_long(concepts, max_len, print_prob)
    concepts = conceptset_utils.filter_too_similar_to_cls(concepts, classes, class_sim_cutoff, device, print_prob)
    concepts = conceptset_utils.filter_too_similar(concepts, other_sim_cutoff, device, print_prob)
    return concepts


def process_prompt_jsons(
    dataset,
    important_path,
    superclass_path,
    around_path,
    save_path,
    max_len=30,
    class_sim_cutoff=0.85,
    other_sim_cutoff=0.9,
    device="cuda",
    print_prob=0,
):
    classes = load_classes(dataset)
    important = load_json(important_path)
    superclass = load_json(superclass_path)
    around = load_json(around_path)

    concepts = merge_prompt_dicts([important, superclass, around])
    concepts = filter_concepts(
        concepts,
        classes,
        max_len=max_len,
        class_sim_cutoff=class_sim_cutoff,
        other_sim_cutoff=other_sim_cutoff,
        device=device,
        print_prob=print_prob,
    )

    save_concept_text(save_path, concepts)
    return concepts
