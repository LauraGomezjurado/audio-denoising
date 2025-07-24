# Denoising Dataset (Philharmonia + 60 Hz Hum)


We start from

* **Clean recordings** – individual instrument samples from the open-source *Philharmonia Orchestra Sound Sample Library* (`audio/…` directory).
* **Noise recording** – a continuous 60 Hz mains-hum clip (`noise.wav`).

The dataset is pairs each clean clip with a synthetically noised counterpart, allowing supervised learning of the denoising task.

---

##  Generation workflow

1. **Create per-SNR datasets**  
   The script `build_dataset.py` mixes every clean clip with a random segment of the noise file at a chosen Signal-to-Noise Ratio (SNR).

   ```bash
   for snr in 15 10 5 0; do
     python build_dataset.py \
         --snr-values $snr \
         --clean-dir audio \
         --noise-file noise.wav \
         --out-dir dataset_snr${snr}
   done
   ```
   Each run produces `dataset_snr{SNR}/train|val|test/{clean,noisy}` plus a metadata CSV.

2. **Merge into a single dataset**  
   We combine the four folders so that *one* clean copy of every file exists and **all** noisy versions live together.  Run:

   ```bash
   python merge_datasets.py
   ```
   This script assembles the final structure shown below in `dataset/` and merges CSVs.

---

## 3. Directory layout
```
dataset/
├── train/
│   ├── clean/           # pristine clips (wav)
│   ├── noisy/          # all noisy variants (wav)
│   └── metadata.csv    # noisy_path,clean_path,snr_db
├── val/
│   ├── clean/
│   ├── noisy/
│   └── metadata.csv
├── test/
│   ├── clean/
│   ├── noisy/
│   └── metadata.csv
└── README.md  
```

CSV columns:
* `noisy_path`  – relative path to the noisy wav file
* `clean_path`  – relative path to the corresponding clean wav file
* `snr_db`     – SNR used when mixing (integer, e.g. 10)

Splits respect an 80 / 10 / 10 ratio (train / val / test).

---

## Signal-to-Noise Ratios used
Generated four difficulty levels:

| SNR (dB) | Description                  |
|----------|-----------------------------|
| 15       | Very mild hum, barely audible |
| 10       | Mild hum (good phone-call quality) |
| 5        | Clearly audible hum          |
| 0        | Signal and hum equally strong |

i could extend this list (e.g. –5 dB) by re-running `build_dataset.py` with additional values.

---

##  Loading the data (PyTorch example)
```python
from data_loading import DenoiseDataset

train_ds = DenoiseDataset("dataset", split="train")
noisy, clean, snr_db = train_ds[0]
```
Filter by SNR if desired:
```python
train_easy = DenoiseDataset("dataset", split="train", snr_subset=[10, 15])
```

---

## Reproducibility / Dependencies
* Python ≥ 3.8
* `numpy` and `soundfile` for I/O (see `requirements.txt`)
* `pandas`, `torch` for loading example (optional)

All scripts are deterministic given `--seed` (default **42**).
