"""Extract 4 evenly spaced frames (640 px wide JPEG) from every RWF-2000 clip.

data/rwf2000/data/RWF-2000 Sliced/val/<Fight|NonFight>/<clip>.avi
  -> data/rwf2000/frames/<Fight|NonFight>/<clip>/f0..f3.jpg
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data" / "rwf2000"
CLIPS = sorted((ROOT / "data").rglob("*.avi"))
TIMES = [0.6, 1.8, 3.0, 4.2]  # seconds into each 5 s clip


def extract(clip):
    out = ROOT / "frames" / clip.parent.name / clip.stem
    out.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(TIMES):
        dest = out / f"f{i}.jpg"
        if not dest.exists():
            subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(clip), "-frames:v", "1",
                 "-vf", "scale=640:-2", "-q:v", "3", str(dest)],
                check=True,
            )


with ThreadPoolExecutor(4) as pool:
    list(pool.map(extract, CLIPS))
print(f"{len(CLIPS)} clips -> {len(list((ROOT / 'frames').rglob('*.jpg')))} frames")
