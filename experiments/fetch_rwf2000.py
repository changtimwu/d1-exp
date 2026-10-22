"""Download the RWF-2000 validation clips (200 Fight, 200 NonFight) into data/rwf2000/."""

from pathlib import Path

from huggingface_hub import snapshot_download

DATASET = "A1mal/RWF-2000-Dataset"
OUT = Path(__file__).resolve().parent.parent / "data" / "rwf2000"

snapshot_download(DATASET, repo_type="dataset", local_dir=OUT, allow_patterns=["data/RWF-2000 Sliced/val/*"])
print(f"{len(list(OUT.rglob('*.avi')))} clips in {OUT}")
