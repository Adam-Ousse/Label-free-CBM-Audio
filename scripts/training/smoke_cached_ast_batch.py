"""One-batch cached AST forward/backward smoke check; not a benchmark."""

from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from scripts.training.ast_input_cache import CachedASTInputDataset, collate_cached_ast_batch
from scripts.training.train_urbansound8k_ast import build_model, load_label_maps_from_metadata, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--base-model-id", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    args = parser.parse_args()
    if args.batch_size * args.gradient_accumulation_steps != 32:
        raise ValueError("Urban smoke must preserve effective batch size 32")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the batch smoke check")
    device = torch.device("cuda")
    set_seed(42)
    id2label, label2id = load_label_maps_from_metadata()
    dataset = CachedASTInputDataset(args.cache, args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_cached_ast_batch)
    batch = next(iter(loader))
    model = build_model(args.base_model_id, device, id2label, label2id)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1)
    torch.cuda.reset_peak_memory_stats(device)
    inputs = batch["input_values"].to(device, non_blocking=True)
    labels = batch["target"].to(device, non_blocking=True)
    if labels.min().item() < 0 or labels.max().item() >= 10:
        raise ValueError("Urban labels are outside 0..9")
    outputs = model(input_values=inputs, labels=labels)
    if tuple(outputs.logits.shape) != (args.batch_size, 10):
        raise RuntimeError("Expected logits shape ({}, 10), got {}".format(args.batch_size, tuple(outputs.logits.shape)))
    if not torch.isfinite(outputs.loss):
        raise RuntimeError("Smoke loss is not finite")
    outputs.loss.backward()
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
        raise RuntimeError("Smoke gradients contain NaN or Inf")
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print("cached AST batch optimizer+scheduler-step smoke passed: batch={} effective_batch={} logits_shape={} loss={:.6f} peak_allocated_gib={:.3f}".format(args.batch_size, args.batch_size * args.gradient_accumulation_steps, tuple(outputs.logits.shape), float(outputs.loss.detach().cpu()), peak), flush=True)


if __name__ == "__main__":
    main()
