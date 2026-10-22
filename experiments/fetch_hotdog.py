"""Download truepositive/hotdog_nothotdog into data/hotdog/<split>/<label>/."""

import json
import urllib.request
from pathlib import Path

DATASET = "truepositive/hotdog_nothotdog"
API = "https://datasets-server.huggingface.co/rows"
OUT = Path(__file__).resolve().parent.parent / "data" / "hotdog"
LABELS = ["hotdog", "not_hotdog"]

for split in ["train", "validation"]:
    offset = 0
    while True:
        url = f"{API}?dataset={DATASET}&config=default&split={split}&offset={offset}&length=100"
        rows = json.load(urllib.request.urlopen(url))["rows"]
        if not rows:
            break
        for r in rows:
            label = LABELS[r["row"]["label"]]
            dest = OUT / split / label / f"{r['row_idx']:04d}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                urllib.request.urlretrieve(r["row"]["image"]["src"], dest)
        offset += len(rows)
    print(split, offset)
