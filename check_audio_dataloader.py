#!/usr/bin/env python3
"""Quick sanity check for ESC-50 and AudioSet dataloaders."""

from __future__ import annotations

import os

# Disable TorchCodec audio decoding (unavailable on cluster due to missing CUDA deps)
os.environ.setdefault("HF_DATASETS_DISABLE_TORCHCODEC", "1")

import argparse

import data_utils


def _shape_str(tensor):
    return tuple(tensor.shape)


def check_dataset(
    name,
    split,
    batch_size,
    num_workers,
    hf_streaming=False,
    hf_cache_dir=None,
    max_items=None,
    hf_decode_audio=False,
):
    print(f"\n=== Checking {name} ({split}) ===")
    classes = data_utils.get_dataset_classes(name)
    print(f"Classes: {len(classes)}")

    dataset = data_utils.get_audio_dataset(
        name,
        split=split,
        hf_streaming=hf_streaming if name == "audioset" else False,
        hf_cache_dir=hf_cache_dir,
        max_items=max_items,
        hf_decode_audio=hf_decode_audio if name == "audioset" else True,
    )
    try:
        print(f"Dataset size: {len(dataset)}")
    except TypeError:
        print("Dataset size: unknown (streaming iterable)")

    loader = data_utils.get_audio_dataloader(
        dataset_name=name,
        split=split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        hf_streaming=hf_streaming if name == "audioset" else False,
        hf_cache_dir=hf_cache_dir,
        max_items=max_items,
        hf_decode_audio=hf_decode_audio if name == "audioset" else True,
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
    parser.add_argument("--audioset_split", type=str, default="balanced", help="AudioSet HF split name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for sanity check")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument(
        "--audioset_streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Hugging Face streaming for AudioSet (default: enabled)",
    )
    parser.add_argument("--audioset_cache_dir", type=str, default=None, help="Optional HF cache directory")
    parser.add_argument("--audioset_max_items", type=int, default=16, help="Optional cap for AudioSet samples")
    parser.add_argument(
        "--audioset_decode_audio",
        action="store_true",
        help="Decode AudioSet waveforms (requires a working TorchCodec runtime)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        check_dataset("esc50", args.esc50_split, args.batch_size, args.num_workers)
        check_dataset(
            "audioset",
            args.audioset_split,
            args.batch_size,
            args.num_workers,
            hf_streaming=args.audioset_streaming,
            hf_cache_dir=args.audioset_cache_dir,
            max_items=args.audioset_max_items,
            hf_decode_audio=args.audioset_decode_audio,
        )
    except Exception as exc:
        raise RuntimeError(
            "Audio dataloader sanity check failed. Verify ESC-50 manifests and Hugging Face AudioSet connectivity."
        ) from exc

    print("\nSanity check completed successfully.")


if __name__ == "__main__":
    main()
