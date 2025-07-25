#!/usr/bin/env python
"""Run evaluation for baseline models and a trained network, writing a
CSV that can later be plotted.

Usage examples
--------------
    # Evaluate identity + notch + your trained network
    python scripts/evaluate_models.py \
        --dataset dataset --split test --batch 8 \
        --checkpoint checkpoints/best.pt \
        --out results/results.csv

    # Only baselines (identity + notch)
    python scripts/evaluate_models.py --dataset dataset --split val

The script reuses the :func:`evaluate.evaluate` function so results are
fully comparable to the numbers you already inspected manually.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path



ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


import argparse
import csv
from typing import List

import torch
from torch.utils.data import DataLoader

from dataset.data_laoding import DenoiseDataset

from models.baselines import IdentityMask, NotchMask
from models.spectrogram_unet import SpectrogramUNet
from evaluate import evaluate, collate_pad  # reuse existing function and collate


BASELINE_CHOICES = [
    "identity",
    "notch",
]


def build_model(kind: str, checkpoint: str | None, device: torch.device):
    if kind == "identity":
        return IdentityMask().to(device)
    elif kind == "notch":
        return NotchMask().to(device)
    elif kind == "unet":
        if checkpoint is None:
            raise ValueError("--checkpoint is required when kind='unet'")
        model = SpectrogramUNet().to(device)
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        return model
    else:
        raise ValueError(f"Unknown model kind: {kind}")


def main():
    p = argparse.ArgumentParser(description="Evaluate baseline + trained models and save CSV")
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--checkpoint", type=str, default=None, help="Path to trained UNet checkpoint (.pt)")
    p.add_argument(
        "--baselines",
        nargs="*",
        default=BASELINE_CHOICES,
        choices=BASELINE_CHOICES,
        help="Which baseline models to run (default: all)",
    )
    p.add_argument("--out", default="results/results.csv", help="CSV output path")
    args = p.parse_args()

    # device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ds = DenoiseDataset(args.dataset, args.split)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_pad)

    results: List[dict] = []

    # baselines
    for kind in args.baselines:
        model = build_model(kind, None, device)
        sdr, psnr = evaluate(model, loader, device)
        print(f"{kind:9s} | SI-SDR: {sdr:.2f} dB | PSNR: {psnr:.2f} dB")
        results.append({"method": kind, "split": args.split, "si_sdr": sdr, "psnr": psnr})

    # trained network 
    if args.checkpoint is not None:
        model = build_model("unet", args.checkpoint, device)
        sdr, psnr = evaluate(model, loader, device)
        model_name = Path(args.checkpoint).stem
        print(f"{model_name:9s} | SI-SDR: {sdr:.2f} dB | PSNR: {psnr:.2f} dB")
        results.append({"method": model_name, "split": args.split, "si_sdr": sdr, "psnr": psnr})

    # write CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "split", "si_sdr", "psnr"])
        writer.writeheader()
        writer.writerows(results)
    # pretty print path
    try:
        rel = out_path.relative_to(Path.cwd())
    except ValueError:
        rel = out_path
    print(f"→ Results written to {rel}")


if __name__ == "__main__":
    main() 