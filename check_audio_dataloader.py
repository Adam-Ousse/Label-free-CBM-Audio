#!/usr/bin/env python3
"""Quick sanity check for ESC-50 and AudioSet dataloaders."""

from __future__ import annotations

import argparse

import data_utils


def _shape_str(tensor):
    return tuple(tensor.shape)


def check_dataset(name, split, batch_size, num_workers):
    print(f"\n=== Checking {name} ({split}) ===")
    classes = data_utils.get_dataset_classes(name)
    print(f"Classes: {len(classes)}")

    dataset = data_utils.get_audio_dataset(name, split=split)
    print(f"Dataset size: {len(dataset)}")

    loader = data_utils.get_audio_dataloader(
        dataset_name=name,
        split=split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    batch = next(iter(loader))
    print(f"Batch audio shape: {_shape_str(batch['audio'])}")
    print(f"Batch target shape: {_shape_str(batch['target'])}")
    print(f"Batch sr shape: {_shape_str(batch['sr'])}")
    print(f"First sample id: {batch['id'][0]}")

    if name == "audioset":
        print(f"AudioSet target dim (multi-label): {batch['target'].shape[-1]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity-check ESC-50 and AudioSet audio dataloaders")
    parser.add_argument("--esc50_split", type=str, default="train", help="ESC-50 manifest split name")
    parser.add_argument("--audioset_split", type=str, default="balanced_train", help="AudioSet manifest split name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for sanity check")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        check_dataset("esc50", args.esc50_split, args.batch_size, args.num_workers)
        check_dataset("audioset", args.audioset_split, args.batch_size, args.num_workers)
    except Exception as exc:
        raise RuntimeError(
            "Audio dataloader sanity check failed. Verify manifest files and audio paths are correct."
        ) from exc

    print("\nSanity check completed successfully.")


if __name__ == "__main__":
    main()
