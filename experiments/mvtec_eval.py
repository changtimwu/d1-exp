"""Zero-shot defect detection of d1 on the MVTec AD test set (run fetch_mvtec.py first).

Usage: uv run experiments/mvtec_eval.py [limit_per_object]
Reports image-level AUROC per object, the standard MVTec AD metric, plus accuracy at 0.5.
"""

import json
import sys
import time
from pathlib import Path

from mlx_vlm import load, predict

MODEL = "LiquidAI/d1-3B"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mvtec"
OUT = ROOT / "outputs" / "mvtec_eval.json"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None


def question(obj):
    name = obj.replace("_", " ")
    return {
        "defect": {
            "type": "bool",
            "instructions": f"This is a close-up photo from a factory inspection camera of a {name}. "
            f"Is this {name} defective?",
            "criteria": {
                "true": "It has a visible defect such as a crack, scratch, hole, cut, stain, "
                "contamination, deformation, or a missing or misplaced part.",
                "false": "It looks like a normal, intact, defect-free product.",
            },
        }
    }


def auroc(pos, neg):
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


model, processor = load(MODEL)

rows, start = [], time.time()
for obj_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
    q = question(obj_dir.name)
    paths = sorted(obj_dir.glob("*/*.png"))
    if LIMIT:
        good = [p for p in paths if p.parent.name == "good"][: LIMIT // 2]
        bad = [p for p in paths if p.parent.name != "good"][: LIMIT - len(good)]
        paths = good + bad
    for path in paths:
        p = predict(model, processor, None, q, images=[str(path)])["answers"]["defect"]["probability"]
        rows.append({"object": obj_dir.name, "defect": path.parent.name, "path": str(path.relative_to(DATA)), "p": p})
    print(f"{obj_dir.name}: {len(paths)} images, {time.time() - start:.0f}s elapsed", flush=True)

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(rows, indent=2))

print(f"\n{'object':<12} {'n':>4} {'AUROC':>6} {'acc@0.5':>8} {'recall':>7} {'spec':>6}")
aucs = []
for obj in sorted({r["object"] for r in rows}) + ["ALL"]:
    sub = rows if obj == "ALL" else [r for r in rows if r["object"] == obj]
    pos = [r["p"] for r in sub if r["defect"] != "good"]
    neg = [r["p"] for r in sub if r["defect"] == "good"]
    auc = auroc(pos, neg)
    if obj != "ALL":
        aucs.append(auc)
    acc = (sum(p >= 0.5 for p in pos) + sum(p < 0.5 for p in neg)) / len(sub)
    rec = sum(p >= 0.5 for p in pos) / len(pos)
    spec = sum(p < 0.5 for p in neg) / len(neg)
    print(f"{obj:<12} {len(sub):>4} {auc:>6.3f} {acc:>8.3f} {rec:>7.3f} {spec:>6.3f}")
print(f"{'MEAN':<12} {'':>4} {sum(aucs) / len(aucs):>6.3f}   (mean of per-object AUROC)")
print(f"\n{time.time() - start:.0f}s for {len(rows)} images")
