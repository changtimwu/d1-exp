"""Zero-shot violence detection of d1 on RWF-2000 val clips (run extract_rwf2000_frames.py first).

Compares three ways to turn a 5 s surveillance clip into one decision:
  single  - the middle frame alone
  max     - each of the 4 frames asked separately, highest P(violence)
  multi   - all 4 frames passed together as one multi-image state
"""

import json
import time
from pathlib import Path

from mlx_vlm import load, predict

MODEL = "LiquidAI/d1-3B"
ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "rwf2000" / "frames"
OUT = ROOT / "outputs" / "rwf2000_eval.json"

CRITERIA = {
    "true": "Someone is fighting, hitting, punching, kicking, shoving, or attacking another person.",
    "false": "People are walking, standing, talking, or doing normal activities with no physical aggression.",
}
FRAME_Q = {"violence": {
    "type": "bool",
    "instructions": "This is a frame from a surveillance camera. Is there physical violence in it?",
    "criteria": CRITERIA,
}}
CLIP_Q = {"violence": {
    "type": "bool",
    "instructions": "Is there physical violence in this clip?",
    "criteria": CRITERIA,
}}
CLIP_STATE = "These images are 4 frames, in order, from a 5-second surveillance camera clip."


def p_violence(state, questions, images):
    return predict(model, processor, state, questions, images=images)["answers"]["violence"]["probability"]


def auroc(pos, neg):
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


model, processor = load(MODEL)

rows, start = [], time.time()
clips = sorted(p for p in FRAMES.glob("*/*") if p.is_dir())
for i, clip in enumerate(clips):
    frames = [str(clip / f"f{j}.jpg") for j in range(4)]
    per_frame = [p_violence(None, FRAME_Q, [f]) for f in frames]
    rows.append({
        "clip": f"{clip.parent.name}/{clip.name}",
        "truth": clip.parent.name == "Fight",
        "frames": per_frame,
        "single": per_frame[2],
        "max": max(per_frame),
        "multi": p_violence(CLIP_STATE, CLIP_Q, frames),
    })
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(clips)} clips, {time.time() - start:.0f}s", flush=True)

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(rows, indent=2))

print(f"\n{'method':<8} {'AUROC':>6} {'acc@0.5':>8} {'recall':>7} {'spec':>6} {'best_acc':>9} {'@thr':>5}")
for m in ["single", "max", "multi"]:
    pos = [r[m] for r in rows if r["truth"]]
    neg = [r[m] for r in rows if not r["truth"]]
    rec = sum(p >= 0.5 for p in pos) / len(pos)
    spec = sum(p < 0.5 for p in neg) / len(neg)
    best = max(
        ((sum(p >= t for p in pos) + sum(p < t for p in neg)) / len(rows), t)
        for t in [x / 100 for x in range(1, 100)]
    )
    print(f"{m:<8} {auroc(pos, neg):>6.3f} {(rec * len(pos) + spec * len(neg)) / len(rows):>8.3f} "
          f"{rec:>7.3f} {spec:>6.3f} {best[0]:>9.3f} {best[1]:>5.2f}")
print(f"\n{time.time() - start:.0f}s for {len(rows)} clips")
