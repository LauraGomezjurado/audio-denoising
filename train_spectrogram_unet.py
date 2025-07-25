import argparse
import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset.data_laoding import DenoiseDataset  
from models.spectrogram_unet import SpectrogramUNet
from utils.metrics import si_sdr, psnr

# optional wandb import (only if used)
wandb_spec = importlib.util.find_spec("wandb")
if wandb_spec is not None:
    import wandb
else:
    wandb = None  # type: ignore


class RandomCropWrapper(Dataset):
    """Wraps DenoiseDataset and returns random crops of fixed length."""

    def __init__(self, base_ds: DenoiseDataset, crop_samples: int = 44100):
        self.base = base_ds
        self.crop = crop_samples

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        noisy, clean, snr_db = self.base[idx]
        length = len(noisy)
        if length <= self.crop:
            pad = self.crop - length
            noisy = np.pad(noisy, (0, pad))
            clean = np.pad(clean, (0, pad))
        else:
            start = np.random.randint(0, length - self.crop)
            noisy = noisy[start : start + self.crop]
            clean = clean[start : start + self.crop]
        # normalise to [-1,1]
        max_val = max(np.abs(noisy).max(), 1e-4)
        noisy = noisy / max_val
        clean = clean / max_val
        return torch.from_numpy(noisy).float(), torch.from_numpy(clean).float()


def get_dataloaders(root: str, crop_samples: int, batch_size: int, num_workers: int = 4, subset: int | None = None):
    """Return train/val DataLoaders.

    When *subset* is not None, restrict both splits to the first *subset* items (useful for quick debugging).
    """
    train_base = DenoiseDataset(root, "train")
    val_base = DenoiseDataset(root, "val")
    if subset is not None:
        indices_train = list(range(min(len(train_base), subset)))
        indices_val = list(range(min(len(val_base), subset)))
        from torch.utils.data import Subset

        train_base = Subset(train_base, indices_train)
        val_base = Subset(val_base, indices_val)

    train_ds = RandomCropWrapper(train_base, crop_samples)
    val_ds = RandomCropWrapper(val_base, crop_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, val_loader


def stft_noisy(wave: torch.Tensor, n_fft: int = 1024, hop: int = 256):
    window = torch.hann_window(n_fft, device=wave.device)
    return torch.stft(wave, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)


def istft(stft_tensor: torch.Tensor, length: int, n_fft: int = 1024, hop: int = 256):
    window = torch.hann_window(n_fft, device=stft_tensor.device)
    return torch.istft(stft_tensor, n_fft=n_fft, hop_length=hop, window=window, length=length)


def train_epoch(model, loader, optimizer, scaler, device, max_batches: int | None = None):
    model.train()
    total_loss = 0.0
    for b_idx, (noisy, clean) in enumerate(tqdm(loader, desc="train", leave=False)):
        noisy = noisy.to(device)
        clean = clean.to(device)
        optimizer.zero_grad()
        # Autocast (mixed precision) only on CUDA devices
        with autocast(enabled=(device.type == "cuda")):
            stft_noi = stft_noisy(noisy)
            mag_noi = torch.abs(stft_noi)
            mask = model(mag_noi.unsqueeze(1)).squeeze(1)
            est_stft = mask * stft_noi
            est_audio = istft(est_stft, length=noisy.shape[-1])
            l1_time = F.l1_loss(est_audio, clean)
            mag_clean = torch.abs(stft_noisy(clean))
            l1_spec = F.l1_loss(mask * mag_noi, mag_clean)
            loss = 0.7 * l1_spec + 0.3 * l1_time
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * noisy.size(0)
        if max_batches is not None and (b_idx + 1) >= max_batches:
            break
    # prevent division by zero when subset ends early
    denom = min(len(loader.dataset), (max_batches or len(loader)) * loader.batch_size)
    return total_loss / denom


def validate(model, loader, device, max_batches: int | None = None):
    model.eval()
    loss_acc = 0.0
    si_sdr_acc = 0.0
    with torch.no_grad():
        for b_idx, (noisy, clean) in enumerate(tqdm(loader, desc="val", leave=False)):
            noisy = noisy.to(device)
            clean = clean.to(device)
            stft_noi = stft_noisy(noisy)
            mag_noi = torch.abs(stft_noi)
            mask = model(mag_noi.unsqueeze(1)).squeeze(1)
            est_stft = mask * stft_noi
            est_audio = istft(est_stft, length=noisy.shape[-1])
            l1_time = F.l1_loss(est_audio, clean)
            mag_clean = torch.abs(stft_noisy(clean))
            l1_spec = F.l1_loss(mask * mag_noi, mag_clean)
            loss = 0.7 * l1_spec + 0.3 * l1_time
            # note: we don't compute PSNR during training loop to save time
            loss_acc += loss.item() * noisy.size(0)
            si_sdr_acc += si_sdr(est_audio, clean).item() * noisy.size(0)
            if max_batches is not None and (b_idx + 1) >= max_batches:
                break
    N = min(len(loader.dataset), (max_batches or len(loader)) * loader.batch_size)
    return loss_acc / N, si_sdr_acc / N


def main():
    p = argparse.ArgumentParser(description="Train Spectrogram U-Net for 60 Hz hum removal")
    p.add_argument("--dataset", default="dataset", help="root path of dataset")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--crop", type=int, default=44100, help="segment length in samples")
    p.add_argument("--save_dir", default="checkpoints", help="where to save models")
    p.add_argument("--workers", type=int, default=4)
    # Debug/quick-run arguments
    p.add_argument("--subset", type=int, default=None, help="Use only the first N items of each split for quick testing")
    p.add_argument("--max_batches", type=int, default=None, help="Limit number of batches per epoch (train/val)")
    # wandb options
    p.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases")
    p.add_argument("--wandb_project", default="audio-denoising", help="wandb project name")
    p.add_argument("--wandb_entity", default=None, help="wandb entity (user or team)")
    p.add_argument("--resume", type=str, default=None,
               help="Path to a checkpoint (.pt) to resume from")
    args = p.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Determine the best available device: CUDA (NVIDIA), MPS (Apple Silicon GPUs), or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    model = SpectrogramUNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    # Mixed-precision & GradScaler only make sense on CUDA; disable elsewhere
    scaler = GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1          # continue AFTER the saved one
        scheduler.last_epoch = ckpt["epoch"]     # keep the LR curve in sync
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    # initialise wandb if requested and available
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb flag is set but wandb is not installed. pip install wandb.")
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        wandb.watch(model, log="all", log_freq=100)

    train_loader, val_loader = get_dataloaders(args.dataset, args.crop, args.batch, args.workers, subset=args.subset)

    best_sdr = -np.inf
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optim, scaler, device, max_batches=args.max_batches)
        val_loss, val_sdr = validate(model, val_loader, device, max_batches=args.max_batches)
        scheduler.step()
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_si_sdr={val_sdr:.2f} dB")
        if args.wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_si_sdr": val_sdr})
        ckpt_path = Path(args.save_dir) / f"epoch{epoch:03d}.pt"
        torch.save({"model": model.state_dict(), "opt": optim.state_dict(), "epoch": epoch}, ckpt_path)
        if val_sdr > best_sdr:
            best_sdr = val_sdr
            torch.save(model.state_dict(), Path(args.save_dir) / "best.pt")
            print(f"   New best model (SI-SDR={best_sdr:.2f} dB) saved.")
            if args.wandb:
                wandb.run.summary["best_si_sdr"] = best_sdr


if __name__ == "__main__":
    main() 