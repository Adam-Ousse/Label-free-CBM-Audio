import argparse
import os

import train_cbm


def _manifest_path(split):
    return os.path.join("data", "esc50", "manifests", "{}.jsonl".format(split))


def _count_jsonl_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _validate_inputs(args):
    if not os.path.exists(args.concept_set):
        raise FileNotFoundError("Concept set file not found: {}".format(args.concept_set))

    for split in (args.train_split, args.val_split, args.test_split):
        manifest = _manifest_path(split)
        if not os.path.exists(manifest):
            raise FileNotFoundError("Missing ESC-50 manifest for split '{}': {}".format(split, manifest))
        count = _count_jsonl_rows(manifest)
        if count == 0:
            raise ValueError("Manifest has no samples for split '{}': {}".format(split, manifest))


def parse_args():
    parser = argparse.ArgumentParser(description="Train CBM on ESC-50 with fold protocol")
    parser.add_argument("--concept_set", type=str, default="data/concept_sets/esc50_filtered_qwen.txt")
    parser.add_argument("--backbone", type=str, default="ast_esc50")
    parser.add_argument("--clap_model", type=str, default="laion/clap-htsat-unfused")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--saga_batch_size", type=int, default=256)
    parser.add_argument("--proj_batch_size", type=int, default=50000)

    parser.add_argument("--feature_layer", type=str, default="layer4")
    parser.add_argument("--activation_dir", type=str, default="saved_activations")
    parser.add_argument("--save_dir", type=str, default="saved_models")

    parser.add_argument("--clip_cutoff", type=float, default=0.25)
    parser.add_argument("--concept_activation_cutoff", type=float, default=None)
    parser.add_argument("--proj_steps", type=int, default=1000)
    parser.add_argument("--interpretability_cutoff", type=float, default=0.45)
    parser.add_argument("--lam", type=float, default=0.0007)
    parser.add_argument("--n_iters", type=int, default=1000)
    parser.add_argument("--print", action="store_true")

    parser.add_argument("--train_split", type=str, default="fold1_train")
    parser.add_argument("--val_split", type=str, default="fold1_val")
    parser.add_argument("--test_split", type=str, default="fold1_test")
    parser.add_argument("--allow_custom_splits", action="store_true", help="Allow non-default split names")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.allow_custom_splits:
        if args.train_split != "fold1_train":
            raise ValueError("Default ESC-50 protocol requires train_split='fold1_train'")
        if args.val_split != "fold1_val":
            raise ValueError("Default ESC-50 protocol requires val_split='fold1_val'")
        if args.test_split != "fold1_test":
            raise ValueError("Default ESC-50 protocol requires test_split='fold1_test'")

    _validate_inputs(args)

    print(
        "ESC-50 CBM splits -> train: {} val: {} test: {}".format(
            args.train_split,
            args.val_split,
            args.test_split,
        )
    )
    print("Concept set: {}".format(args.concept_set))
    print("Backbone: {} (frozen feature extraction)".format(args.backbone))

    cbm_args = argparse.Namespace(
        dataset="esc50",
        concept_set=args.concept_set,
        backbone=args.backbone,
        clap_model=args.clap_model,
        device=args.device,
        batch_size=args.batch_size,
        saga_batch_size=args.saga_batch_size,
        proj_batch_size=args.proj_batch_size,
        feature_layer=args.feature_layer,
        activation_dir=args.activation_dir,
        save_dir=args.save_dir,
        clip_cutoff=args.clip_cutoff,
        concept_activation_cutoff=args.concept_activation_cutoff,
        proj_steps=args.proj_steps,
        interpretability_cutoff=args.interpretability_cutoff,
        lam=args.lam,
        n_iters=args.n_iters,
        print=args.print,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        enforce_esc50_fold1_protocol=(not args.allow_custom_splits),
        audioset_streaming=False,
        audioset_cache_dir=None,
        audioset_max_items=None,
    )

    train_cbm.train_cbm_and_save(cbm_args)


if __name__ == "__main__":
    main()
