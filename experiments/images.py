"""Ask d1 the same questions about every image in images/ and save the answers.

Usage: uv run experiments/images.py [questions.json]
"""

import json
import sys
from pathlib import Path

from mlx_vlm import load, predict

MODEL = "LiquidAI/d1-3B"
ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "images"
OUT = ROOT / "outputs" / "images.json"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic"}

QUESTIONS = {
    "people": {"type": "bool", "instructions": "Are there any people in the image?"},
    "scene": {
        "type": "choice",
        "instructions": "What kind of scene is this?",
        "criteria": {
            "indoor": "Inside a building or vehicle",
            "outdoor": "Outside",
            "document": "A document, screen, or text-heavy page",
            "other": "None of these",
        },
    },
    "clutter": {
        "type": "score",
        "instructions": "How cluttered is the image?",
        "criteria": ["Clean and simple", "Some clutter", "Very busy"],
    },
}

if len(sys.argv) > 1:
    QUESTIONS = json.loads(Path(sys.argv[1]).read_text())

paths = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in EXTS)
if not paths:
    sys.exit(f"No images in {IMAGES}")

model, processor = load(MODEL)

results = {}
for path in paths:
    out = predict(model, processor, None, QUESTIONS, images=[str(path)])
    results[path.name] = out["answers"]
    summary = ", ".join(
        f"{k}={a['value']:.2f}" if isinstance(a["value"], float) else f"{k}={a['value']}"
        for k, a in out["answers"].items()
    )
    print(f"{path.name}: {summary}")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(results, indent=2, default=str))
print(f"\nSaved {OUT}")
