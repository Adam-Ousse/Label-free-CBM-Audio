import argparse
import os

import train_cbm


def _validate_inputs(args):
    if not os.path.exists(args.concept_set):
        raise FileNotFoundError("Concept set file not found: {}".format(args.concept_set))


def parse_args():
    parser = argparse.ArgumentParser(description="Train CBM on AudioSet with a frozen pretrained AST backbone")
    parser.add_argument("--concept_set", type=str, default="data/concept_sets/audioset_filtered_qwen.txt")
    parser.add_argument("--backbone", type=str, default="ast_audioset")
    parser.add_argument("--clap_model", type=str, default="laion/clap-htsat-unfused")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
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

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--train_subset", type=str, default="balanced")
    parser.add_argument("--val_split", type=str, default="eval")
    parser.add_argument("--val_subset", type=str, default="full")
    parser.add_argument("--test_split", type=str, default=None)
    parser.add_argument("--test_subset", type=str, default=None)

    parser.add_argument("--audioset_streaming", action="store_true", help="Use Hugging Face streaming mode")
    parser.add_argument("--audioset_cache_dir", type=str, default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--audioset_max_items", type=int, default=None, help="Optional cap on loaded samples")
    return parser.parse_args()


def main():
    args = parse_args()
    _validate_inputs(args)

    print(
        "AudioSet CBM config -> train: {}/{} val: {}/{} test: {}/{}".format(
            args.train_subset,
            args.train_split,
            args.val_subset,
            args.val_split,
            args.test_subset,
            args.test_split,
        )
    )
    print("Concept set: {}".format(args.concept_set))
    print("Backbone: {} (frozen pretrained AST)".format(args.backbone))
    print("CLAP model: {}".format(args.clap_model))

    cbm_args = argparse.Namespace(
        dataset="audioset",
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
        audioset_train_subset=args.train_subset,
        audioset_val_subset=args.val_subset,
        audioset_test_subset=args.test_subset,
        enforce_esc50_fold1_protocol=False,
        audioset_streaming=args.audioset_streaming,
        audioset_cache_dir=args.audioset_cache_dir,
        audioset_max_items=args.audioset_max_items,
    )

    train_cbm.train_cbm_and_save(cbm_args)


if __name__ == "__main__":
    main()
