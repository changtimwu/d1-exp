# Where Does a Decision Model's Generality Come From? d1 on Medical Images

*Fourth experiment with LiquidAI's d1-3B on Apple Silicon via mlx-vlm — October 2026.
Earlier: [Hot Dog or Not](hotdog-zero-shot.md) ·
[Factory Defects on MVTec AD](mvtec-zero-shot.md) ·
[Violence in Surveillance Video](rwf2000-violence-zero-shot.md).*

> **Not for clinical use.** This is a research experiment on public benchmark
> images. Nothing here is a diagnostic tool, and the results below show why.

## The question

The first three experiments showed that d1-3B, a decision model with vision,
handles hot dogs, surface defects and fights without any training. But *why*
does it generalize?

1. **Pretraining knowledge.** d1 is a finetune of LiquidAI's LFM2.5-VL-3B,
   which learned from web-scale image–text data. Perhaps it already "knows"
   hot dogs and scratches, and d1 only changed how answers come out.
2. **The decision structure itself.** An image plus a typed question with
   described options might let the model reason about *what it sees* against
   *what the question describes*, even for concepts it never learned.

Medical images are a good test because they're far from everyday photos.
We picked three datasets along a gradient of distance from web images, and
designed the experiment to separate the two explanations.

## Setup

### Hardware and software

| | |
|---|---|
| Machine | Apple M1 Max, 32 GB unified memory |
| OS | macOS 26.5.1 |
| Inference | [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.10 (MLX 0.32.4) |
| Models | `LiquidAI/d1-3B` and its base `LiquidAI/LFM2.5-VL-3B`, both bfloat16 as published |

### Datasets: MedMNIST+ at 224 px

[MedMNIST v2](https://medmnist.com/) is a standardized collection of
biomedical classification datasets. It has official train, validation and test
splits, and published baselines trained on each one. We used the **full
official test splits** of the native 224×224 versions, known as MedMNIST+:

| Dataset | Images | Task | Classes | Distance from web photos | Licence |
|---|---|---|---|---|---|
| **DermaMNIST** | 2,005 | Skin lesion diagnosis (dermatoscope photos) | 7 | Close: magnified colour photos | CC BY-NC 4.0 |
| **PneumoniaMNIST** | 624 | Pediatric chest X-ray: normal vs pneumonia | 2 | Medium: X-rays are common online | CC BY 4.0 |
| **BloodMNIST** | 3,421 | Blood-cell type (stained smear microscopy) | 8 | Far: specialist microscopy | CC BY 4.0 |

We chose BloodMNIST over PathMNIST for the microscopy end because
PathMNIST at 224 px is a 12.6 GB download. Every file was checked against the
md5 checksum published on [Zenodo](https://zenodo.org/records/10519652).

**Class imbalance matters for reading the results.** In DermaMNIST, 67% of
images are one class (nevus, a common mole), so a model that always answers
"mole" scores 67% accuracy. For that reason we report **AUROC** (one-vs-rest,
averaged over classes) and **balanced accuracy** (the average of per-class
recall), alongside plain accuracy.

### The design: two controls

**Control 1: d1 vs its own base model, with everything else identical.** In
mlx-vlm, d1's model class is a thin subclass of the LFM2.5-VL model class, and
the two checkpoints have the same architecture. So we loaded the **base
LFM2.5-VL-3B weights and ran them through d1's exact prompt and answer
readout**. The only difference between the two runs is the finetuned weights.

- If the base model does as well as d1, the ability comes from pretraining.
- If d1 does clearly better, the finetune added something.

**Control 2: name the concept, or only describe it.** Every image got two
questions, asked in one pass that shares the image:

- **concept**: names the medical classes, for example "Does it show
  pneumonia?" or "Melanoma", "Basophil".
- **visual**: describes only what each class *looks like*, without naming it,
  for example "Are there hazy or patchy white areas clouding parts of the lungs?"
  or "A cell packed with large dark purple-blue granules that hide its nucleus".

If the model only works when the medical name is used, it's relying on learned
knowledge of that name. If it works from descriptions alone, it's matching what
it sees against the question.

Both wordings were written once, before any results, and never tuned. Here
they are for the chest X-ray task:

```python
"concept": {"type": "bool",
            "instructions": "This is a pediatric chest X-ray. Does it show pneumonia?",
            "criteria": {"true": "Pneumonia is present.", "false": "Normal chest X-ray, no pneumonia."}},
"visual":  {"type": "bool",
            "instructions": "This is a chest X-ray. Are there hazy or patchy white areas clouding parts of the lungs?",
            "criteria": {"true": "Parts of the lung fields look cloudy, hazy, or patchy white.",
                         "false": "The lung fields look clear and dark, with sharp outlines."}},
```

The 7 skin-lesion and 8 blood-cell options are written out in
[`experiments/medmnist_eval.py`](../experiments/medmnist_eval.py).

## Results

All 6,050 test images, through both models with two questions each, took
**about 100 minutes**, roughly 0.35–0.55 s per image for both questions.

### The main table: AUROC

| | d1, concept | d1, visual | base, concept | base, visual | Trained baselines* |
|---|---|---|---|---|---|
| **DermaMNIST** (skin) | **0.815** | 0.798 | 0.703 | 0.695 | 0.912–0.920 |
| **PneumoniaMNIST** (X-ray) | 0.822 | **0.879** | 0.845 | 0.785 | 0.956–0.991 |
| **BloodMNIST** (microscopy) | 0.553 | **0.802** | 0.513 | 0.767 | 0.997–0.998 |

\* Published MedMNIST v2 test results for ResNet-18 (224), ResNet-50 (224)
and Google AutoML Vision, each trained on that dataset's training split
([medmnist.com](https://medmnist.com/)). Those "224" ResNets were trained on 28 px
images scaled up to 224, not on these native 224 px images, so the comparison
is approximate.

### Accuracy, which shows how far this is from usable

| | d1, concept | d1, visual | Majority-class accuracy | Trained ResNet-18 (224) |
|---|---|---|---|---|
| DermaMNIST: accuracy | 64.6% | 33.9% | 66.9% | 75.4% |
| DermaMNIST: balanced accuracy | 28.0% | 27.4% | 14.3% | n/a |
| PneumoniaMNIST: accuracy @0.5 | 74.5% | 70.0% | 62.5% | 86.4% |
| PneumoniaMNIST: best accuracy (tuned cutoff, optimistic) | 76.3% | 81.2% | n/a | n/a |
| BloodMNIST: accuracy | 16.3% | 29.5% | 19.5% | 96.3% |
| BloodMNIST: balanced accuracy | 12.4% | 28.8% | 12.5% | n/a |

For a balanced-accuracy column, the majority-class figure equals chance:
1 divided by the number of classes. On skin lesions, d1's plain accuracy is
*below* always answering "mole". Every zero-shot result here is far from the
trained models.

### What the controls show

**1. Medical concept knowledge comes from pretraining, not the finetune.** On
chest X-rays with the concept prompt, the base model *slightly beats* d1
(0.845 vs 0.822). Whatever LFM2.5-VL learned about "pneumonia on an X-ray" came
from pretraining, and d1's finetune didn't add to it.

**2. d1's finetune makes the model follow the question's descriptions.** The
clearest case is the X-ray visual prompt. Asked "are there hazy white areas in
the lungs?", the base model said **yes to every single X-ray**: 100% recall,
0% specificity, AUROC 0.785. With identical inputs, d1 gave a usable,
graded answer: AUROC 0.879, the best X-ray score in the table, with 94%
specificity. On skin lesions, d1 beats the base by about 0.1 AUROC with both
prompts. So the finetune doesn't teach medicine. It teaches the model to
**weigh the image against the described options** rather than drift toward
one answer.

**3. Where the concept isn't known, describing it helps a lot.** Blood-cell
types are the far end of the scale. With the cell names, both models fail:
AUROC 0.55 for d1 and 0.51 for the base, essentially chance. Both answered
"immature granulocyte" for nearly every cell (96% and 100%). **The same model,
asked with visual descriptions instead, reaches 0.80.** It can't recognise
"eosinophil" from the word, but it can tell "a cell packed with dark granules"
from "a cell with a lobed nucleus". The decision structure lets you write the
missing knowledge into the question itself.

**4. The gradient: distance from web photos hurts named concepts, not
descriptions.**

| | Skin (close) | X-ray (medium) | Blood cells (far) |
|---|---|---|---|
| d1, concept | 0.815 | 0.822 | **0.553** |
| d1, visual | 0.798 | 0.879 | **0.802** |

Named-concept performance collapses at the far end of the scale. Description
performance stays roughly flat around 0.8 across all three. The X-ray result
not following the expected order (medium beating close) suggests "distance"
is better measured by how familiar the *named concept* is than by how
photo-like the image looks.

### Why "immature granulocyte"? A bias check

Answering one class 96% of the time looks like a bias toward one option.
To find out which kind, we ran a quick diagnostic on 400 blood-cell images.
It's an explanation of the failure, not prompt tuning.

| Concept prompt variant | AUROC | Most common answer |
|---|---|---|
| As written | 0.540 | immature granulocyte, 94% |
| "Immature granulocyte" description shortened (the parenthetical list of three subtypes removed) | 0.550 | **lymphocyte, 60%** |
| Options in reverse order | 0.544 | immature granulocyte, 94% |

The position of the option makes no difference. Its *wording* does:
"Immature granulocyte (myelocyte, metamyelocyte, or promyelocyte)" is the
broadest-sounding option. Remove that and the model falls back on the next
most generic answer. The ranking stays at chance in every variant. When the
model doesn't recognise any of the named options, it picks the one that sounds
most like a catch-all.

That's a practical warning for anyone writing `choice` questions. **Watch the
predicted-class distribution, not just accuracy.** A long or generic option
description can quietly become the default answer.

## Conclusion

**So where does d1's generality come from? Both, in different roles:**

- **What it recognises** comes from pretraining. The base model already knows
  pneumonia on an X-ray about as well as d1. Nothing the finetune adds creates
  knowledge the backbone lacks. Blood-cell names stay at chance either way.
- **How reliably it applies that recognition to your question** comes from
  d1's decision finetune. With the same pixels and the same words, d1 gives
  graded, usable probabilities where the base model answers yes to everything
  or collapses onto one option.
- **The question can fill gaps in knowledge.** When the model doesn't know
  the concept but can see the features, describing the features rather than
  naming the class raised blood-cell AUROC from 0.55 to 0.80.

**For medical imaging specifically, fine-tuning is needed.** Even the best
zero-shot result (0.879 AUROC on X-rays) is well below a ResNet-18 trained on
these datasets (0.956). Blood-cell accuracy of 30% against the trained 96%
isn't close. This is the opposite of the hot dog result, and the reason is
clear: there the backbone knew the concept, and here mostly it doesn't.

### Lessons that transfer beyond medicine

1. **Test the base model with the same prompt** before crediting a finetune
   for a capability. Here it separated "knows the concept" from "answers
   reliably" cleanly.
2. **Describe the options, don't just name them**, whenever the domain is
   specialised. Descriptions were never much worse than names here, and in one case
   they were much better.
3. **Check the predicted-class distribution.** A model that's unsure picks
   the most catch-all-sounding option, and accuracy alone can hide that,
   especially with imbalanced classes.
4. **Zero-shot is a screening tool.** It tells you quickly whether a domain
   is within reach (hot dogs, surface defects, fights) or needs training data
   (blood cells, skin-lesion diagnosis).

### Caveats

- **Not for clinical use.** These are benchmark images, scored once, and
  far below trained models.
- **One wording per prompt type, never tuned.** Better-written descriptions,
  especially for the skin and blood classes, could score higher. The bias
  check shows how sensitive `choice` questions are to wording.
- **The base model is run through d1's prompt format.** That keeps the
  comparison controlled, but the base model wasn't trained for this answer
  slot and might do better asked in its own generative style. Read the base
  numbers as "same inputs, different weights", not as the best the base model
  can do.
- **Approximate comparison with the published baselines.** They were trained
  on 28 px images scaled up to 224, and we didn't re-run them.
- **The pneumonia best accuracy is optimistic**, since the cutoff was chosen
  on the same 624 images it was scored on.
- **The MedMNIST images are small, standardised crops.** Full-resolution
  clinical images would behave differently.

## Reproduce it

```bash
git clone https://github.com/changtimwu/d1-exp && cd d1-exp
uv sync
./experiments/fetch_medmnist.sh            # ~2.8 GB of MedMNIST+ 224 px + the 6.2 GB base model, md5-checked
./experiments/run_medmnist.sh              # d1 and base on all three test splits (~100 min), then the report
uv run experiments/medmnist_report.py      # reprint the tables from outputs/
uv run experiments/medmnist_blood_bias.py  # the 400-image option-bias diagnostic (~3 min)
```

Per-image probabilities are written to `outputs/medmnist_{d1,base}_{derma,pneumonia,blood}.json`.
