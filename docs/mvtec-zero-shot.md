# Factory Defect Detection with a Decision Model: d1 on MVTec AD

*Second experiment with LiquidAI's d1-3B on Apple Silicon via mlx-vlm — October 2026.
The first one is [Hot Dog or Not](hotdog-zero-shot.md).*

## The question

In the [hot dog experiment](hotdog-zero-shot.md), d1-3B told hot dogs from
everything else 98.4% of the time with no training at all. That's a concept any
vision model has seen millions of times.

Industrial defect detection is the opposite case. A scratch on a screw thread,
a crack in a capsule, or a wire in the wrong slot are rare, small, and specific
to one product. The usual approach is to train an anomaly-detection model per
product line on hundreds of images of good parts.

So the question is: **can a general decision model with vision be pointed at a
production line and asked "is this part defective?", or does this task need a
dataset and training?**

## Setup

### Hardware and software

Same as the hot dog run:

| | |
|---|---|
| Machine | Apple M1 Max, 32 GB unified memory |
| OS | macOS 26.5.1 |
| Runtime | Python 3.12, [uv](https://docs.astral.sh/uv/) |
| Inference | [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.10 (MLX 0.32.4) |
| Model | `LiquidAI/d1-3B`, bfloat16, as published (no quantization) |

### Dataset

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) is the
standard benchmark for industrial anomaly detection. It covers 15 product types:
5 textures (carpet, grid, leather, tile, wood) and 10 objects (bottle, cable,
capsule, hazelnut, metal nut, pill, screw, toothbrush, transistor, zipper). Each
type has between 1 and 8 defect classes, and there are 73 in total.

We used the full **test set: 1,725 images (467 good, 1,258 defective)**, taken
from the Hugging Face copy
[`TheoM55/mvtec_all_objects_split`](https://huggingface.co/datasets/TheoM55/mvtec_all_objects_split).
The images are 700–1024 px squares, shot under controlled lighting from a fixed
inspection camera.

MVTec AD is licensed CC BY-NC-SA 4.0, for non-commercial use only. The images
aren't redistributed with this repo.

### The question asked

One Boolean question per image, with the product name filled in, and short
descriptions of what "yes" and "no" mean:

```python
{
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
```

The model never sees a good example. It doesn't get a reference image,
per-product hints, or any training. As with the hot dogs, this was the first
and only wording tried.

### Metrics

- **Image-level AUROC per product type**, averaged over the 15 types. This is
  the standard MVTec AD classification metric, so results can be compared
  directly with published methods. It doesn't depend on choosing a cutoff.
- **Recall and specificity at a 0.5 cutoff.** Recall is the share of defective
  parts caught. Specificity is the share of good parts correctly passed. Together
  they show how the model behaves out of the box.

## Results

All 1,725 images took **66 minutes, about 2.3 seconds per image**. That's
roughly 10 times slower than the hot dogs, because these images are 30 to 40
times larger in pixel count.

### Per product

| Product | n | AUROC | Recall @0.5 | Specificity @0.5 | Best accuracy (cutoff) |
|---|---|---|---|---|---|
| carpet | 117 | **1.000** | 0.72 | 1.00 | 100% (0.05) |
| grid | 78 | **1.000** | 0.89 | 1.00 | 100% (0.06) |
| leather | 124 | **1.000** | 0.86 | 1.00 | 99.2% (0.10) |
| wood | 79 | **1.000** | 0.92 | 1.00 | 100% (0.12) |
| tile | 117 | 0.997 | 0.68 | 1.00 | 97.4% (0.10) |
| hazelnut | 110 | 0.983 | 0.80 | 1.00 | 95.5% (0.32) |
| toothbrush | 42 | 0.981 | 0.53 | 1.00 | 95.2% (0.25) |
| screw | 160 | 0.962 | 0.36 | 1.00 | 91.2% (0.18) |
| bottle | 83 | 0.953 | 0.68 | 0.95 | 91.6% (0.13) |
| capsule | 132 | 0.953 | 0.52 | 1.00 | 90.9% (0.05) |
| zipper | 151 | 0.930 | 0.46 | 1.00 | 87.4% (0.08) |
| cable | 150 | 0.894 | 0.09 | 1.00 | 81.3% (0.12) |
| metal nut | 115 | 0.879 | 0.70 | 1.00 | 84.3% (0.22) |
| pill | 167 | 0.840 | 0.57 | 0.96 | 85.0% (0.13) |
| transistor | 100 | 0.652 | 0.30 | 0.97 | 70.0% (0.50) |
| **Mean** | | **0.935** | | | |
| **All pooled** | 1725 | 0.919 | 0.59 | 0.99 | 83.9% (0.10–0.20) |

"Best accuracy" uses the cutoff that maximizes accuracy for that product alone.
It's chosen on the same images it's scored on, so it's optimistic. It shows
how good each product could get with a cutoff picked from a few labelled
examples.

### How that compares

| Method | Training | MVTec AD image AUROC |
|---|---|---|
| WinCLIP (CVPR 2023) | none (zero-shot CLIP) | 91.8 |
| AnomalyCLIP (ICLR 2024) | trained on a different dataset (VisA) | 91.5* |
| AdaCLIP (ECCV 2024) | trained on other datasets | 89.2* |
| **d1-3B, one question, this run** | **none** | **93.5** |

WinCLIP's figure is from its paper. Figures marked * are from comparison
tables in later papers ([PA-CLIP](https://arxiv.org/abs/2503.01292),
[FAPrompt](https://arxiv.org/abs/2410.10289)) and weren't checked against the
originals. Newer zero-shot methods report higher numbers. For example,
PA-CLIP lists MuSc at 97.8, though MuSc works by comparing many unlabelled test
images against each other, which is a different setup from asking about one
image at a time.

So a general-purpose 3B decision model, asked one plain-English question per
image, lands **in the range of dedicated zero-shot anomaly detectors** from
2023–24, without being built for this task. Treat the comparison as rough: our
prompt, image preprocessing, and single run differ from those papers' setups.

### Two problems: thresholds and "logical" defects

**1. The 0.5 cutoff is far too strict.** Out of the box, d1 is a cautious
inspector. It passed 99.1% of good parts, but caught only 59% of the defective
ones. The ranking is good; the scores are just low. Most products separate best
with a cutoff between 0.05 and 0.3.

No single cutoff works for every product, though. Here's the trade-off with one
cutoff for everything:

| Global cutoff | Recall (defects caught) | Specificity (good parts passed) |
|---|---|---|
| 0.50 | 59% | 99% |
| 0.30 | 73% | 94% |
| 0.20 | 85% | 79% |
| 0.10 | 93% | 59% |

The cutoff needs to be set per product line. That's realistic on a factory
floor, where you'd have some labelled parts for each line anyway.

**2. Surface defects are easy; logical defects are close to invisible.** Ranking
the 73 defect classes by AUROC shows a clear pattern:

| Easiest (AUROC 1.000) | | Hardest | AUROC |
|---|---|---|---|
| wood: hole, scratch, color, liquid | | transistor: misplaced | **0.24** |
| zipper: fabric border | | pill: wrong pill type | **0.28** |
| carpet, grid, leather: most classes | | metal nut: flipped upside down | **0.51** |
| | | cable: wires swapped | 0.72 |
| | | pill: color | 0.69 |

Anything that *looks damaged*, such as a hole, scratch, stain, cut, or tear, d1
catches reliably, even on busy textures like carpet. What it misses are defects
that only count as defects **if you know what the correct product looks like**:

- **`transistor/misplaced`, p = 0.08.** The transistor is *gone*, leaving a
  clean, undamaged circuit board. Nothing in the image looks broken. There's
  just a board, and only knowledge of the product says something should be on it.
- **`cable/cable_swap`, p = 0.15.** A cross-section with two green-yellow
  wires and one brown, where a good cable has green-yellow, blue, and brown.
  Every wire is intact; the colour combination is what's wrong.
- **`metal_nut/flip`**: the nut is upside down. **`pill/pill_type`**: a perfectly
  good pill of the wrong kind.

The AUROC below 0.5 for misplaced transistors and wrong pill types means d1
rates these defective parts as *less* suspicious than real good ones. They look
cleaner than normal.

This isn't really a model failure. The question gave d1 no way of knowing what
"correct" means for this product. A human inspector seeing a single photo would
miss these too, unless they'd been shown a good part first.

## Conclusion

**For surface defects, d1 works zero-shot. For logical defects, it needs to be
told what normal looks like.**

- On textures and visible damage (carpet, grid, leather, tile, wood, hazelnut)
  d1 reaches AUROC 0.98–1.00 with a single plain question. With a cutoff chosen
  for each product, several are at or near 100% accuracy.
- Averaged over all 15 products, its 0.935 AUROC is in the same range as
  dedicated zero-shot anomaly detectors such as WinCLIP (0.918).
- The out-of-the-box 0.5 cutoff isn't usable on a line: it misses 41% of
  defects. **Calibrating the cutoff per product line is required**, and only
  takes a few dozen labelled parts.
- Missing, swapped, flipped, or wrong-type parts are where d1 fails. That
  requires knowing the reference product, which a single image can't provide.

### Before fine-tuning, try these

In rough order of effort:

1. **Set a cutoff per product line** from a few dozen labelled parts. This is the
   biggest and cheapest win here.
2. **Show it a good part.** d1 accepts several images in one state, so you can
   pass a known-good reference image next to the test image and ask "Is the
   second part defective compared with the first?". This directly targets the
   logical-defect gap, and is the natural next experiment.
3. **Describe the product.** For example: "A good transistor has three straight
   leads and sits centred on the board." "A good cable has one green-yellow,
   one blue, and one brown wire." This costs nothing and encodes exactly the
   knowledge that's missing.
4. **Ask more specific questions.** Separate `bool` questions for "missing or
   misplaced part", "surface damage" and "wrong colour" can each be tuned
   independently. d1 reads the image once and answers them all from the same
   cached prefix.
5. **Fine-tune only if you still need more.** If a line needs more than 99%
   recall on subtle, product-specific defects, a model trained on that line's
   good parts, such as a classic anomaly detector like PatchCore, will still
   win. The mlx-vlm d1 package covers inference only.

### Caveats

- **One wording, one run.** Prompt variations, especially ones that describe
  each product, could change per-product results noticeably.
- **The best-accuracy figures are optimistic.** Cutoffs were chosen on the
  same images they were scored on. The out-of-the-box numbers are the 0.5
  rows.
- **Rough comparison to the literature.** We didn't re-run any baselines.
  Their numbers come from papers that use different preprocessing.
- **Clean lab images.** MVTec AD has controlled lighting and centred parts. A
  real line with glare, motion blur, or varying poses will be harder.
- **Speed.** At about 2.3 seconds per 1024 px image on an M1 Max in
  bfloat16, this is fine for spot checks, but slow for a high-speed line. Smaller
  input images or a 4-bit conversion would speed it up, with some accuracy
  cost that we didn't measure.

## Reproduce it

```bash
git clone https://github.com/changtimwu/d1-exp && cd d1-exp
uv sync
uv run experiments/fetch_mvtec.py    # ~1.7 GB of test parquet -> 1,725 PNGs in data/mvtec/
uv run experiments/mvtec_eval.py     # ~1 hour on an M1 Max; per-object AUROC table
uv run experiments/mvtec_eval.py 2   # quick check: 2 images per object
```

Per-image scores are written to `outputs/mvtec_eval.json`.
