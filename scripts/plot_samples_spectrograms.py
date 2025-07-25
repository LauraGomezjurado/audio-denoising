#!/usr/bin/env python
"""Plot spectrograms (noisy / denoised / clean) for every instrument in the
`samples/` directory

Run:
    python scripts/plot_samples_spectrograms.py

Optional flags:
    --samples_dir   Root folder that contains one sub-folder per instrument
    --out_dir       Where to save the figures (PNG). Defaults to samples/figures
    --show          Display the plots interactively instead of / in addition to
                    saving them.

Each instrument sub-folder must contain exactly one of each file pattern:
    *_clean.wav
    *_noisy_*.wav
    denoised_*.wav

Uses the project-wide STFT helper already defined in evaluate.py.
"""
# Standard library
from pathlib import Path
import argparse

# Third-party
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Project root to sys.path BEFORE other project imports
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Project imports
from evaluate import stft_fn  # existing helper (complex STFT)

plt.rcParams.update({"figure.dpi": 150})

def amplitude_to_db(spec: torch.Tensor) -> torch.Tensor:
    """Convert magnitude spectrogram to decibels."""
    return 20 * torch.log10(spec + 1e-8)

def load_wav_mono(path: Path) -> np.ndarray:
    """Load WAV file and convert to mono float32 numpy array."""
    wav, _ = sf.read(path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32)

def compute_mag_db(wav: np.ndarray) -> torch.Tensor:
    """Compute magnitude spectrogram in dB for 1-D numpy array."""
    wav_t = torch.from_numpy(wav).unsqueeze(0)
    mag = torch.abs(stft_fn(wav_t)).squeeze(0)
    return amplitude_to_db(mag)

def plot_triplet(spec_noisy, spec_denoised, spec_clean, title: str, out_path: Path, show: bool = False):
    """Save a 1×3 spectrogram figure to *out_path* (PNG)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, S, ttl in zip(
        axes,
        [spec_noisy, spec_denoised, spec_clean],
        ["Noisy", "Denoised", "Clean"],
    ):
        im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(ttl)
        ax.set_xlabel("Frames")
    axes[0].set_ylabel("Frequency bins")
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description="Plot spectrogram triplets for all instruments in samples/")
    p.add_argument("--samples_dir", default="samples", help="Root directory containing instrument sub-folders")
    p.add_argument("--out_dir", default=None, help="Destination for figures (default: <samples_dir>/figures)")
    p.add_argument("--show", action="store_true", help="Display figures interactively as well")
    args = p.parse_args()

    samples_dir = Path(args.samples_dir)
    if args.out_dir is None:
        out_dir = samples_dir / "figures"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instrument_dirs = [d for d in samples_dir.iterdir() if d.is_dir()]
    if not instrument_dirs:
        raise RuntimeError(f"No sub-folders found under {samples_dir!s}")

    for inst_dir in instrument_dirs:
        name = inst_dir.name
        try:
            clean_path = next(inst_dir.glob("*_clean.wav"))
            noisy_path = next(inst_dir.glob("*_noisy_*.wav"))
            denoised_path = next(inst_dir.glob("denoised_*.wav"))
        except StopIteration:
            print(f"[Warning] Missing expected files in {inst_dir}. Skipping.")
            continue

        clean_wav = load_wav_mono(clean_path)
        noisy_wav = load_wav_mono(noisy_path)
        deno_wav = load_wav_mono(denoised_path)

        spec_n = compute_mag_db(noisy_wav)
        spec_d = compute_mag_db(deno_wav)
        spec_c = compute_mag_db(clean_wav)

        out_path = out_dir / f"{name}.png"
        plot_triplet(spec_n, spec_d, spec_c, title=name.capitalize(), out_path=out_path, show=args.show)
        print(f"Saved figure: {out_path}")

if __name__ == "__main__":
    main() 