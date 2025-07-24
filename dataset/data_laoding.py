import random, soundfile as sf, pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class DenoiseDataset(Dataset):
    def __init__(self, root, split="train", snr_subset=None):
        root = Path(root)
        meta = pd.read_csv(root / split / "metadata.csv")
        if snr_subset is not None:
            meta = meta[meta["snr_db"].isin(snr_subset)]
        self.meta = meta.reset_index(drop=True)
        self.base = root

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        clean, _ = sf.read(self.base / row.clean_path)
        noisy, _ = sf.read(self.base / row.noisy_path)
        return noisy.astype("float32"), clean.astype("float32"), row.snr_db
