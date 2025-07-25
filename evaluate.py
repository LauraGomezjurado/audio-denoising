import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset.data_laoding import DenoiseDataset
from models.spectrogram_unet import SpectrogramUNet
from utils.metrics import si_sdr, psnr


def stft_fn(wave, n_fft=1024, hop=256):
    return torch.stft(wave, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft, device=wave.device), return_complex=True)


def istft_fn(spec, length, n_fft=1024, hop=256):
    return torch.istft(spec, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft, device=spec.device), length=length)


# helper to pad variable length to same size per batch
def collate_pad(batch):
    noisy_list, clean_list, snr = zip(*batch)
    lengths = [len(x) for x in noisy_list]
    max_len = max(lengths)
    def pad(arr, length):
        if len(arr) < length:
            return np.pad(arr, (0, length - len(arr)))
        return arr
    noisy_pad = [pad(x, max_len) for x in noisy_list]
    clean_pad = [pad(x, max_len) for x in clean_list]
    return np.stack(noisy_pad), np.stack(clean_pad), np.array(snr)


def evaluate(model, loader, device):
    model.eval()
    sdr_total, psnr_total = 0.0, 0.0
    with torch.no_grad():
        for noisy, clean, _ in tqdm(loader):
            if isinstance(noisy, np.ndarray):
                noisy = torch.from_numpy(noisy).float()
            if isinstance(clean, np.ndarray):
                clean = torch.from_numpy(clean).float()
            noisy = noisy.to(device)
            clean = clean.to(device)
            spec_n = stft_fn(noisy)
            mask = model(torch.abs(spec_n).unsqueeze(1)).squeeze(1)
            est_spec = mask * spec_n
            est_wave = istft_fn(est_spec, length=noisy.shape[-1])
            sdr_total += si_sdr(est_wave, clean).item() * noisy.size(0)
            psnr_total += psnr(est_wave, clean).item() * noisy.size(0)
    n = len(loader.dataset)
    return sdr_total / n, psnr_total / n


def main():
    p = argparse.ArgumentParser(description="Evaluate trained Spectrogram U-Net")
    p.add_argument("--dataset", default="dataset", help="dataset root directory")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", required=True, help="path to model .pt file (state_dict)")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = SpectrogramUNet().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    ds = DenoiseDataset(args.dataset, args.split)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate_pad)

    sdr, pval = evaluate(model, loader, device)
    print(f"Split: {args.split} | SI-SDR: {sdr:.2f} dB | PSNR: {pval:.2f} dB")


if __name__ == "__main__":
    main() 