"""Zero-shot d1 (or the base LFM2.5-VL-3B) on MedMNIST+ 224 px test splits.

Usage: uv run experiments/medmnist_eval.py {d1|base} {pneumonia|derma|blood} [limit]

Every image gets two questions in one pass (they share the image prefix):
  concept - names the medical classes
  visual  - describes only what each class looks like, without naming it
`base` runs LFM2.5-VL-3B weights through d1's exact prompt and readout, so the
weights are the only difference. Run fetch_medmnist.sh first.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from mlx_vlm import load
from mlx_vlm.models.d1 import Model as D1Model

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "medmnist"
MODELS = {"d1": "LiquidAI/d1-3B", "base": "LiquidAI/LFM2.5-VL-3B"}

# Class keys follow the official MedMNIST label order (index = label id).
TASKS = {
    "pneumonia": {
        "file": "pneumoniamnist_224.npz",
        "classes": ["normal", "pneumonia"],
        "concept": {
            "type": "bool",
            "instructions": "This is a pediatric chest X-ray. Does it show pneumonia?",
            "criteria": {"true": "Pneumonia is present.", "false": "Normal chest X-ray, no pneumonia."},
        },
        "visual": {
            "type": "bool",
            "instructions": "This is a chest X-ray. Are there hazy or patchy white areas clouding parts of the lungs?",
            "criteria": {
                "true": "Parts of the lung fields look cloudy, hazy, or patchy white.",
                "false": "The lung fields look clear and dark, with sharp outlines.",
            },
        },
    },
    "derma": {
        "file": "dermamnist_224.npz",
        "classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "concept": {
            "type": "choice",
            "instructions": "This is a dermatoscope image of a skin lesion. What is the diagnosis?",
            "criteria": {
                "akiec": "Actinic keratosis or intraepithelial carcinoma (Bowen's disease)",
                "bcc": "Basal cell carcinoma",
                "bkl": "Benign keratosis (seborrheic keratosis, solar lentigo, or lichen planus-like keratosis)",
                "df": "Dermatofibroma",
                "mel": "Melanoma",
                "nv": "Melanocytic nevus (a common mole)",
                "vasc": "Vascular lesion (angioma, angiokeratoma, pyogenic granuloma, or haemorrhage)",
            },
        },
        "visual": {
            "type": "choice",
            "instructions": "This is a magnified close-up photo of a spot on the skin. Which description fits it best?",
            "criteria": {
                "akiec": "A rough, scaly, red or pink flat patch",
                "bcc": "A pearly or shiny pink bump, often with fine branching blood vessels or a central sore",
                "bkl": "A waxy, brown, stuck-on looking spot with a warty or crusty surface",
                "df": "A small firm brownish spot with a pale centre surrounded by a darker ring",
                "mel": "An irregular, asymmetric spot with uneven colours (black, brown, blue, red) and ragged edges",
                "nv": "A round, evenly coloured brown spot with smooth, regular edges",
                "vasc": "A red, purple, or dark-blue spot made of blood vessels",
            },
        },
    },
    "blood": {
        "file": "bloodmnist_224.npz",
        "classes": ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"],
        "concept": {
            "type": "choice",
            "instructions": "This is a microscope image of a stained blood smear, centred on one cell. What type of cell is it?",
            "criteria": {
                "basophil": "Basophil",
                "eosinophil": "Eosinophil",
                "erythroblast": "Erythroblast",
                "ig": "Immature granulocyte (myelocyte, metamyelocyte, or promyelocyte)",
                "lymphocyte": "Lymphocyte",
                "monocyte": "Monocyte",
                "neutrophil": "Neutrophil",
                "platelet": "Platelet",
            },
        },
        "visual": {
            "type": "choice",
            "instructions": "This is a microscope image of stained blood, centred on one object. Which description fits the object in the centre best?",
            "criteria": {
                "basophil": "A cell packed with large dark purple-blue granules that hide its nucleus",
                "eosinophil": "A cell full of bright orange-red granules, with a nucleus of two lobes",
                "erythroblast": "A small cell with a very round, dense, dark nucleus and a bluish rim",
                "ig": "A large cell with a big oval or kidney-shaped nucleus and pale, slightly granular insides",
                "lymphocyte": "A small cell almost filled by one round dark nucleus, with a thin rim of blue",
                "monocyte": "A large cell with a folded or kidney-shaped nucleus and grey-blue insides",
                "neutrophil": "A cell whose nucleus is split into 3 to 5 connected lobes, with pale pink insides",
                "platelet": "Only tiny purple fragments, with no complete cell or nucleus",
            },
        },
    },
}


def main():
    which, task_name = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    task = TASKS[task_name]

    npz = np.load(DATA / task["file"])
    images, labels = npz["test_images"], npz["test_labels"].reshape(-1)
    if limit:
        # Evenly spaced subsample keeps the class mix of the full split.
        idx = np.linspace(0, len(labels) - 1, limit).astype(int)
        images, labels = images[idx], labels[idx]

    model, processor = load(MODELS[which])
    if which == "base":
        # Same architecture as d1; borrow d1's prompt and readout unchanged.
        model.__class__ = D1Model

    questions = {"concept": task["concept"], "visual": task["visual"]}
    rows, start = [], time.time()
    for i, (arr, label) in enumerate(zip(images, labels)):
        image = Image.fromarray(arr).convert("RGB")
        answers = model.predict(processor, None, questions, images=[image])["answers"]
        row = {"label": task["classes"][int(label)]}
        for name, a in answers.items():
            row[name] = a["probabilities"] if a["type"] == "choice" else a["probability"]
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(labels)} images, {time.time() - start:.0f}s", flush=True)

    out = ROOT / "outputs" / f"medmnist_{which}_{task_name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"seconds": time.time() - start, "rows": rows}, indent=1))
    print(f"{len(rows)} images in {time.time() - start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
