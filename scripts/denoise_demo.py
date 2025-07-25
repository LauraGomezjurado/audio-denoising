#!/usr/bin/env python
"""Minimal Gradio demo to let users upload a noisy WAV and hear the denoised output.

Run with:
    python scripts/denoise_demo.py

Dependencies: gradio, torch, torchaudio, soundfile (already in requirements).
"""
import os, sys
from pathlib import Path

# Ensure project root is on the Python path BEFORE other imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch
import numpy as np
from evaluate import stft_fn, istft_fn  # reuse helpers
from models.spectrogram_unet import SpectrogramUNet



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpectrogramUNet().to(device)
state = torch.load("checkpoints/best.pt", map_location=device)
model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
model.eval()


#Inference helper

def denoise(audio_tuple):
    """Gradio callback: audio_tuple = (sr, np.ndarray). Returns same format."""
    if audio_tuple is None:
        return None

    sr, wav_np = audio_tuple

    # Convert to mono if stereo
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)

    wav = torch.from_numpy(wav_np).float().to(device)

    # Add batch dimension
    wav = wav.unsqueeze(0)  # (1, samples)

    with torch.no_grad():
        spec = stft_fn(wav)
        mask = model(torch.abs(spec).unsqueeze(1)).squeeze(1)
        est_spec = mask * spec
        est_wave = istft_fn(est_spec, length=wav.shape[-1])

    out = est_wave.cpu().numpy().squeeze()
    # Normalise to prevent clipping
    out = out / (np.abs(out).max() + 1e-8)
    return (sr, out.astype(np.float32))


# Interface definition

demo = gr.Interface(
    fn=denoise,
    inputs=gr.Audio(sources=["upload"], type="numpy", label="Noisy audio (.wav)"),
    outputs=gr.Audio(type="numpy", label="Denoised output"),
    title="60 Hz Hum-Removal Demo (Spectrogram U-Net)",
    description=(
        "Upload a speech recording containing mains hum. The model estimates a soft mask "
        "in the STFT domain to suppress the hum while preserving speech."
    ),
)

if __name__ == "__main__":
    demo.launch() 