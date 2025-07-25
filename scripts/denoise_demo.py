#!/usr/bin/env python
"""Minimal Gradio demo to let users upload a noisy WAV and hear the denoised output.

Run with:
    python scripts/denoise_demo.py

Dependencies: gradio, torch, torchaudio, soundfile (already in requirements).
"""
# Standard lib
import os, sys, argparse
from pathlib import Path

# Ensure project root is on the Python path BEFORE other imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch
import numpy as np
import soundfile as sf  # for offline file IO

# Module-global model handle (populated at runtime in __main__)
model = None

from evaluate import stft_fn, istft_fn  # reuse helpers
from models.spectrogram_unet import SpectrogramUNet



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Helper: load model checkpoint (we defer actual loading until we know which
# checkpoint the user requested – default remains the same).
# -----------------------------------------------------------------------------

def load_model(checkpoint_path: str):
    """Return SpectrogramUNet loaded with weights from *checkpoint_path*."""
    model = SpectrogramUNet().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


# =====================
#   Inference helpers
# =====================

def _infer_tensor(wav: torch.Tensor, model: SpectrogramUNet) -> torch.Tensor:
    """Core denoising given 1-D float32 tensor on *device*. Returns tensor."""
    wav = wav.unsqueeze(0)  # (1, samples)
    with torch.no_grad():
        spec = stft_fn(wav)
        mask = model(torch.abs(spec).unsqueeze(1)).squeeze(1)
        est_spec = mask * spec
        est_wave = istft_fn(est_spec, length=wav.shape[-1])
    return est_wave.squeeze(0)


def denoise_gradio(audio_tuple):
    """Gradio callback: audio_tuple = (sr, np.ndarray). Returns same format."""
    if audio_tuple is None:
        return None

    sr, wav_np = audio_tuple

    # Convert to mono if stereo
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)

    wav = torch.from_numpy(wav_np).float().to(device)
    est_wave = _infer_tensor(wav, model)
    out = est_wave.cpu().numpy().squeeze()
    # Normalise to prevent clipping
    out = out / (np.abs(out).max() + 1e-8)
    return (sr, out.astype(np.float32))


# Interface definition

# Build Gradio interface (constructed regardless so that CLI users still have
# the option to fall back to it without re-importing).
demo = gr.Interface(
    fn=denoise_gradio,
    inputs=gr.Audio(sources=["upload"], type="numpy", label="Noisy audio (.wav)"),
    outputs=gr.Audio(type="numpy", label="Denoised output"),
    title="60 Hz Hum-Removal Demo (Spectrogram U-Net)",
    description=(
        "Upload a speech recording containing mains hum. The model estimates a soft mask "
        "in the STFT domain to suppress the hum while preserving speech."
    ),
)

# -----------------------------------------------------------------------------
#   CLI or interactive demo entry-point
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Denoise a WAV through Spectrogram U-Net or launch Gradio demo.")
    parser.add_argument("--input", "-i", help="Path to noisy WAV (16 kHz, mono).")
    parser.add_argument("--output", "-o", help="Destination path for denoised WAV.")
    parser.add_argument("--checkpoint", "-c", default="checkpoints/best.pt",
                        help="Model checkpoint (.pt) to load. Default: checkpoints/best.pt")
    args = parser.parse_args()

    # Select mode -------------------------------------------------------------

    if args.input:
        # ------ Offline single-file mode ------
        if not args.output:
            parser.error("--output is required when using --input")

        noisy, sr = sf.read(args.input)
        if noisy.ndim == 2:
            noisy = noisy.mean(axis=1)

        wav_tensor = torch.from_numpy(noisy).float().to(device)

        # (Re)load model with chosen checkpoint – keeps GPU mem tidy.
        model = load_model(args.checkpoint)

        est_wave = _infer_tensor(wav_tensor, model).cpu().numpy()
        # Normalise to prevent clipping
        est_wave = est_wave / (np.abs(est_wave).max() + 1e-8)

        # Ensure output directory exists
        os.makedirs(Path(args.output).parent, exist_ok=True)
        sf.write(args.output, est_wave, sr)
        print(f"Saved denoised file to {args.output}")
    else:
        # ------ Interactive demo mode ------
        model = load_model(args.checkpoint)
        demo.launch() 