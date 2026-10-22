"""Zero-shot hot dog / not hot dog accuracy of d1 on data/hotdog (run fetch_hotdog.py first)."""

import json
from pathlib import Path

from mlx_vlm import load, predict

MODEL = "LiquidAI/d1-3B"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "hotdog"
OUT = ROOT / "outputs" / "hotdog_eval.json"

QUESTIONS = {
    "hotdog": {"type": "bool", "instructions": "Is there a hot dog in this image?"},
}

model, processor = load(MODEL)

rows = []
for path in sorted(DATA.glob("*/*/*.jpg")):
    truth = path.parent.name == "hotdog"
    p = predict(model, processor, None, QUESTIONS, images=[str(path)])["answers"]["hotdog"]["probability"]
    rows.append({"path": str(path.relative_to(DATA)), "truth": truth, "p": p})

tp = sum(r["truth"] and r["p"] >= 0.5 for r in rows)
tn = sum(not r["truth"] and r["p"] < 0.5 for r in rows)
fp = sum(not r["truth"] and r["p"] >= 0.5 for r in rows)
fn = sum(r["truth"] and r["p"] < 0.5 for r in rows)
pos = [r["p"] for r in rows if r["truth"]]
neg = [r["p"] for r in rows if not r["truth"]]
auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))

print(f"n={len(rows)}  accuracy={(tp + tn) / len(rows):.3f}  AUC={auc:.3f}")
print(f"TP={tp} FN={fn} FP={fp} TN={tn}")
print("\nMisses:")
for r in rows:
    if r["truth"] != (r["p"] >= 0.5):
        print(f"  {r['path']}  p(hotdog)={r['p']:.3f}")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(rows, indent=2))
