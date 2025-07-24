import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

# --------------------------
# Utility functions
# --------------------------

def list_wav_files(root: Path) -> List[Path]:
    """Recursively collect all .wav files under root."""
    return sorted([p for p in root.rglob("*.wav") if p.is_file()])


def partition_list(lst: List, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Tuple[List, List, List]:
    """Split lst into train/val/test according to ratios."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1."
    n = len(lst)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train = lst[:n_train]
    val = lst[n_train : n_train + n_val]
    test = lst[n_train + n_val :]
    return train, val, test


def calculate_power(audio: np.ndarray) -> float:
    """Return power of audio signal."""
    return np.mean(audio ** 2)


def mix_audio(clean: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Mix noise into clean signal at desired SNR (in dB)."""
    clean_power = calculate_power(clean)
    noise_power = calculate_power(noise)

    if noise_power == 0:
        raise ValueError("Noise segment power is zero; check noise file.")

    # Compute required noise scaling factor
    snr_linear = 10 ** (target_snr_db / 10)
    desired_noise_power = clean_power / snr_linear
    scaling_factor = np.sqrt(desired_noise_power / noise_power)
    noisy = clean + noise * scaling_factor
    # Clip to [-1,1]
    return np.clip(noisy, -1.0, 1.0)


def extract_noise_segment(noise: np.ndarray, length: int) -> np.ndarray:
    """Extract a random segment from noise of given length. If noise is shorter, tile it."""
    if len(noise) >= length:
        start = random.randint(0, len(noise) - length)
        return noise[start : start + length]
    # if noise shorter, tile until long enough
    repeats = int(np.ceil(length / len(noise)))
    tiled = np.tile(noise, repeats)[:length]
    return tiled


# --------------------------
# Main processing
# --------------------------

def process_split(
    split_name: str,
    files: List[Path],
    out_root: Path,
    noise_audio: np.ndarray,
    sample_rate: int,
    snr_values: List[float],
):
    """Generate noisy-clean pairs for split."""
    clean_out_dir = out_root / split_name / "clean"
    noisy_out_dir = out_root / split_name / "noisy"
    clean_out_dir.mkdir(parents=True, exist_ok=True)
    noisy_out_dir.mkdir(parents=True, exist_ok=True)

    metadata_entries = []

    for clean_path in files:
        clean_audio, sr = sf.read(clean_path)
        if sr != sample_rate:
            raise ValueError(
                f"Sample rate mismatch for {clean_path} (expected {sample_rate}, got {sr})."
            )

        clean_audio = clean_audio.astype(np.float32)
        # if stereo, convert to mono
        if clean_audio.ndim > 1:
            clean_audio = np.mean(clean_audio, axis=1)
        length = len(clean_audio)

        # Choose SNR randomly from list
        snr_db = random.choice(snr_values)

        noise_segment = extract_noise_segment(noise_audio, length)
        noisy_audio = mix_audio(clean_audio, noise_segment, snr_db)

        # Save files
        rel_name = clean_path.stem  # without extension
        clean_out_path = clean_out_dir / f"{rel_name}_clean.wav"
        noisy_out_path = noisy_out_dir / f"{rel_name}_noisy_snr{snr_db:.1f}dB.wav"

        sf.write(clean_out_path, clean_audio, samplerate=sample_rate)
        sf.write(noisy_out_path, noisy_audio, samplerate=sample_rate)

        metadata_entries.append(
            f"{noisy_out_path.relative_to(out_root)},{clean_out_path.relative_to(out_root)},{snr_db:.1f}"
        )

    # Write metadata
    meta_file = out_root / f"{split_name}_metadata.csv"
    with open(meta_file, "w") as f:
        f.write("noisy_path,clean_path,snr_db\n")
        f.write("\n".join(metadata_entries))

    print(f"Finished {split_name}: {len(files)} samples -> {clean_out_dir.parent}")



def main():
    parser = argparse.ArgumentParser(description="Build noisy-clean audio dataset for denoising tasks.")
    parser.add_argument(
        "--clean-dir",
        type=str,
        default="audio",
        help="Directory containing clean .wav files (will search recursively).",
    )
    parser.add_argument(
        "--noise-file", type=str, default="noise.wav", help="Path to noise .wav file."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dataset",
        help="Output directory where dataset splits will be stored.",
    )
    parser.add_argument(
        "--snr-values",
        type=float,
        nargs="*",
        default=[0, 5, 10, 15],
        help="List of SNR values (dB) to randomly sample from when mixing noise.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Ratios for train/val/test splits.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    clean_dir = Path(args.clean_dir)
    noise_file = Path(args.noise_file)
    out_root = Path(args.out_dir)

    print("Loading noise file ...")
    noise_audio, noise_sr = sf.read(noise_file)
    if noise_audio.ndim > 1:
        noise_audio = np.mean(noise_audio, axis=1)

    print(f"Noise file loaded: {noise_file} (sr={noise_sr}, duration={len(noise_audio)/noise_sr:.1f}s)")

    print("Gathering clean files ...")
    clean_files = list_wav_files(clean_dir)
    if not clean_files:
        raise RuntimeError(f"No .wav files found under {clean_dir}")

    print(f"Found {len(clean_files)} clean files.")
    random.shuffle(clean_files)

    train_files, val_files, test_files = partition_list(clean_files, ratios=tuple(args.ratios))

    print("Building dataset ...")
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name, files in splits.items():
        process_split(
            split_name,
            files,
            out_root,
            noise_audio,
            sample_rate=noise_sr,
            snr_values=args.snr_values,
        )

    print("All done! Dataset available at", out_root.resolve())


if __name__ == "__main__":
    main() 