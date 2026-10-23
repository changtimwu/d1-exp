"""Summarize outputs/medmnist_{model}_{task}.json files written by medmnist_eval.py."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Published MedMNIST v2 test AUC / ACC (medmnist.com), trained on each dataset.
BASELINES = {
    "pneumonia": {"ResNet-18 (224)": (0.956, 0.864), "ResNet-50 (224)": (0.962, 0.884), "AutoML Vision": (0.991, 0.946)},
    "derma": {"ResNet-18 (224)": (0.920, 0.754), "ResNet-50 (224)": (0.912, 0.731), "AutoML Vision": (0.914, 0.768)},
    "blood": {"ResNet-18 (224)": (0.998, 0.963), "ResNet-50 (224)": (0.997, 0.950), "AutoML Vision": (0.998, 0.966)},
}


def auroc(scores, truth):
    """Rank-based AUROC (ties get average rank)."""
    scores, truth = np.asarray(scores, float), np.asarray(truth, bool)
    order = scores.argsort()
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos, neg = truth.sum(), (~truth).sum()
    return (ranks[truth].sum() - pos * (pos + 1) / 2) / (pos * neg)


def binary(rows, key):
    p = np.array([r[key] for r in rows])
    y = np.array([r["label"] == "pneumonia" for r in rows])
    best = max(((p >= t) == y).mean() for t in np.unique(p))
    return {
        "auc": auroc(p, y), "acc": ((p >= 0.5) == y).mean(), "best_acc": best,
        "recall": (p[y] >= 0.5).mean(), "spec": (p[~y] < 0.5).mean(),
    }


def multiclass(rows, key):
    classes = list(rows[0][key])
    P = np.array([[r[key][c] for c in classes] for r in rows])
    y = np.array([classes.index(r["label"]) for r in rows])
    pred = P.argmax(1)
    present = [k for k in range(len(classes)) if (y == k).any()]
    per_class_recall = {classes[k]: float((pred[y == k] == k).mean()) for k in present}
    return {
        "auc": np.mean([auroc(P[:, k], y == k) for k in present]),
        "acc": (pred == y).mean(),
        "bal_acc": np.mean(list(per_class_recall.values())),
        "per_class_recall": per_class_recall,
        "pred_share": {classes[k]: v / len(y) for k, v in Counter(pred).most_common(3)},
        "majority": Counter(y).most_common(1)[0][1] / len(y),
    }


def main():
    for task in ["pneumonia", "derma", "blood"]:
        print(f"\n=== {task}")
        for which in ["d1", "base"]:
            path = ROOT / "outputs" / f"medmnist_{which}_{task}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            rows = data["rows"]
            print(f"{which}: n={len(rows)}, {data['seconds'] / len(rows):.2f}s/image (both questions)")
            for key in ["concept", "visual"]:
                m = binary(rows, key) if task == "pneumonia" else multiclass(rows, key)
                if task == "pneumonia":
                    print(f"  {key:<8} AUC={m['auc']:.3f} acc@0.5={m['acc']:.3f} best_acc={m['best_acc']:.3f} "
                          f"recall={m['recall']:.3f} spec={m['spec']:.3f}")
                else:
                    print(f"  {key:<8} AUC={m['auc']:.3f} acc={m['acc']:.3f} bal_acc={m['bal_acc']:.3f} "
                          f"(majority={m['majority']:.3f}) predicted={ {k: round(v, 2) for k, v in m['pred_share'].items()} }")
                    print(f"           recall per class: { {k: round(v, 2) for k, v in m['per_class_recall'].items()} }")
        print("  published (trained):", "  ".join(f"{k} AUC={a:.3f} ACC={c:.3f}" for k, (a, c) in BASELINES[task].items()))


if __name__ == "__main__":
    main()
