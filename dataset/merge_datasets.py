from pathlib import Path
import csv, shutil

src_roots = [Path(f"dataset_snr{snr}") for snr in (15, 10, 5, 0)]
dst_root  = Path("dataset")
splits    = ["train", "val", "test"]

for split in splits:
    (dst_root / split / "clean").mkdir(parents=True, exist_ok=True)
    (dst_root / split / "noisy").mkdir(parents=True, exist_ok=True)

    # copy clean once
    clean_src = src_roots[0] / split / "clean"
    for f in clean_src.glob("*.wav"):
        shutil.copy2(f, dst_root / split / "clean" / f.name)

    # copy noisy + collect metadata
    rows = []
    for root in src_roots:
        noisy_src = root / split / "noisy"
        for f in noisy_src.glob("*.wav"):
            shutil.copy2(f, dst_root / split / "noisy" / f.name)
        with open(root / f"{split}_metadata.csv") as csvfile:
            rdr = csv.DictReader(csvfile)
            rows.extend(rdr)

    # write merged metadata
    with open(dst_root / split / "metadata.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["noisy_path", "clean_path", "snr_db"])
        writer.writeheader()
        writer.writerows(rows)

print("Done. Combined dataset at", dst_root.resolve())
