#!/usr/bin/env python
"""Generate publication-ready plots from the CSV produced by
``scripts/evaluate_models.py``.

It creates two figures side by side (SI-SDR and PSNR) and saves them to
PNG and PDF for convenience.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Improved styling
sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")
palette = sns.color_palette("colorblind")


def main():
    p = argparse.ArgumentParser(description="Plot SI-SDR / PSNR results as bar charts")
    p.add_argument("csv", help="Path to results CSV (method, split, si_sdr, psnr)")
    p.add_argument("--out_dir", default="results/figures", help="Where to save plots")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df_sorted = df.sort_values("si_sdr", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # SI-SDR
    sns.barplot(ax=axes[0], data=df_sorted, x="method", y="si_sdr", palette=palette, errorbar=None)
    axes[0].set_title("SI-SDR (dB)")
    axes[0].set_ylabel("dB")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=45)

    # Annotate values
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.2f", padding=3, fontsize=8)

    # PSNR
    sns.barplot(ax=axes[1], data=df_sorted, x="method", y="psnr", palette=palette, errorbar=None)
    axes[1].set_title("PSNR (dB)")
    axes[1].set_ylabel("dB")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=45)

    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f", padding=3, fontsize=8)

    plt.tight_layout()

    png_path = out_dir / "results.png"
    pdf_path = out_dir / "results.pdf"
    svg_path = out_dir / "results.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)

    print(f"Figures saved to {png_path}, {pdf_path} and {svg_path}")


if __name__ == "__main__":
    main() 