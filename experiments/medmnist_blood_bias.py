"""Diagnostic (not tuning): why does the blood 'concept' prompt pick immature granulocyte?

On an evenly spaced 400-image subsample, compare the original concept prompt with
  short    - the same prompt with the long 'ig' description cut to "Immature granulocyte"
  reversed - the original options in reverse order (ig moves from option D to E)
"""

import sys
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from medmnist_eval import DATA, TASKS  # noqa: E402
from medmnist_report import auroc  # noqa: E402

from mlx_vlm import load

task = TASKS["blood"]
original = task["concept"]
short = {**original, "criteria": {**original["criteria"], "ig": "Immature granulocyte"}}
reversed_ = {**original, "criteria": dict(reversed(original["criteria"].items()))}
questions = {"original": original, "short": short, "reversed": reversed_}

npz = np.load(DATA / task["file"])
idx = np.linspace(0, len(npz["test_labels"]) - 1, 400).astype(int)
images, labels = npz["test_images"][idx], npz["test_labels"].reshape(-1)[idx]

model, processor = load("LiquidAI/d1-3B")
probs = {k: [] for k in questions}
for arr in images:
    answers = model.predict(processor, None, questions, images=[Image.fromarray(arr).convert("RGB")])["answers"]
    for k, a in answers.items():
        probs[k].append([a["probabilities"][c] for c in task["classes"]])

y = np.array(labels)
for k, P in probs.items():
    P = np.array(P)
    pred = P.argmax(1)
    auc = np.mean([auroc(P[:, c], y == c) for c in range(len(task["classes"]))])
    share = {task["classes"][c]: round(n / len(y), 2) for c, n in Counter(pred).most_common(3)}
    print(f"{k:<9} AUC={auc:.3f} acc={(pred == y).mean():.3f} predicted={share}")
