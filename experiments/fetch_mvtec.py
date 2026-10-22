"""Download the MVTec AD test splits into data/mvtec/<object>/<defect>/.

MVTec AD is CC BY-NC-SA 4.0 (non-commercial); the images stay out of git.
"""

from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

DATASET = "TheoM55/mvtec_all_objects_split"
OUT = Path(__file__).resolve().parent.parent / "data" / "mvtec"

repo = Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=["data/*.test-*.parquet"]))

total = 0
for parquet in sorted(repo.glob("data/*.test-*.parquet")):
    table = pq.read_table(parquet, columns=["image_path", "object", "defect"]).to_pylist()
    for i, row in enumerate(table):
        dest = OUT / row["object"] / row["defect"] / f"{i:04d}.png"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(row["image_path"]["bytes"])
    total += len(table)
print(f"{total} images in {OUT}")
