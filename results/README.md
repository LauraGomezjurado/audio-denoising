# Audio Denoising – Hum-Removal Benchmark

This repository trains and evaluates a Spectrogram U-Net that removes the 60 Hz mains hum (and its harmonics) from speech recordings.

## Evaluation summary

| Method    | Split | SI-SDR (dB) | PSNR (dB) |
|-----------|-------|------------:|----------:|
| Identity  | test  | 7.50        | 44.24     |
| Notch     | test  | 7.04        | 43.93     |
| **UNet (best.pt)** | test  | **26.30** | **62.79** |

*Numbers correspond to the `results/results.csv` produced on the full **test** split (188 files).*

### Interpretation

* **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio) measures how much of the desired signal remains relative to residual noise and artefacts after allowing a global gain adjustment. Higher is better.
* **PSNR** (Peak Signal-to-Noise Ratio) is –10 log₁₀(MSE) with peak value 1. Higher indicates a lower mean-squared error to the clean reference.

The raw noisy signals ("Identity") achieve roughly 7 dB SI-SDR.  A fixed 60 Hz notch filter removes part of the hum but also degrades speech components around those frequencies, so overall quality does not improve.  The learned Spectrogram U-Net raises SI-SDR to 26 dB (≈ +19 dB over the raw input), reducing distortion power by a factor of **~75×** and pushing PSNR above 62 dB.

A bar chart of the same results is saved to `results/figures/results.png`:

```
results/figures/results.png
```

## Re-running the evaluation

1. **Evaluate models**

   ```bash
   # identity + notch + trained UNet
   python scripts/evaluate_models.py \
       --dataset dataset --split test --batch 8 \
       --checkpoint checkpoints/best.pt \
       --out results/results.csv
   ```

2. **Plot metrics**

   ```bash
   python scripts/plot_results.py results/results.csv
   ```

The commands regenerate the CSV and figure so the README can be kept in sync with future experiments (e.g. alternative checkpoints or additional baselines).

---

*Last updated automatically by `scripts/evaluate_models.py` and `scripts/plot_results.py`.* 