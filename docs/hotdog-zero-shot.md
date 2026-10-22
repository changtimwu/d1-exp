# Hot Dog or Not: Does d1 Need Fine-Tuning for a Simple Vision Task?

*An experiment with LiquidAI's d1-3B on Apple Silicon via mlx-vlm — October 2026*

## The question

[d1-3B](https://huggingface.co/LiquidAI/d1-3B) is a 3-billion-parameter model from
LiquidAI that does one thing: it answers structured questions. You give it a
*state* (text, JSON, images, or a mix) and a set of typed questions — `bool`,
`choice`, or `score` — and it returns a probability for every option. It doesn't
generate any text. Each decision is a single forward pass that stops where the
answer would go, and the answer tokens' log-probabilities are read off as
probabilities.

d1 is a finetune of the LFM2.5-VL-3B vision-language model, so it can see. That
raises a practical question: **for a very simple visual task, can you use d1
as-is, or do you still need to collect a dataset and fine-tune it?**

The canonical "simple" vision task is the one made famous by HBO's *Silicon
Valley*: **hot dog or not hot dog**.

## Setup

### Hardware and software

| | |
|---|---|
| Machine | Apple M1 Max, 32 GB unified memory |
| OS | macOS 26.5.1 |
| Runtime | Python 3.12, [uv](https://docs.astral.sh/uv/) |
| Inference | [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.10 (MLX 0.32.4) |
| Model | `LiquidAI/d1-3B`, bfloat16, as published (no quantization) |

mlx-vlm 0.7.10 is the first PyPI release that includes the
[d1 model package](https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/d1),
so a plain `uv add mlx-vlm` is all it takes. The checkpoint is about 6 GB and
downloads on first use.

### Dataset

[`truepositive/hotdog_nothotdog`](https://huggingface.co/datasets/truepositive/hotdog_nothotdog)
on Hugging Face:

| Split | hotdog | not_hotdog |
|---|---|---|
| train | 100 | 100 |
| validation | 25 | 25 |
| **total** | **125** | **125** |

The images are small, mostly around 150×220 pixels. They're everyday photos:
people eating, food on plates, street scenes, and the "not hot dog" class
includes plenty of other food, which makes it a reasonable test rather than a
trivial one.

No training happens in this experiment, so the train/validation split doesn't
matter. All 250 images were used as test data.

### The question asked

One Boolean question, worded as plainly as possible:

```python
QUESTIONS = {
    "hotdog": {"type": "bool", "instructions": "Is there a hot dog in this image?"},
}

answer = predict(model, processor, None, QUESTIONS, images=[path])
p = answer["answers"]["hotdog"]["probability"]   # P(yes)
```

The state is `None` because the image is the whole input. d1 returns
`P(yes)`. An image is called a hot dog when `P(yes) ≥ 0.5`.

There was no prompt tuning, no few-shot examples, and no calibration. The
question above was the first and only wording tried.

## Results

All 250 images ran in about 50 seconds end to end, including loading the model,
which works out to well under 0.2 seconds per image.

### Headline numbers at the default 0.5 threshold

| | Predicted hot dog | Predicted not |
|---|---|---|
| **Actually hot dog** | 121 | 4 |
| **Actually not** | 0 | 125 |

- **Accuracy: 98.4%** (246 / 250)
- **Precision: 100%.** Every image it called a hot dog really was one.
- **Recall: 96.8%.** It missed 4 of 125 hot dogs.
- **ROC AUC: ≈1.000.** Apart from one tie at 0.06, every hot dog scored higher
  than every non-hot-dog.

### The scores separate cleanly

The highest score any non-hot-dog image received was **0.12**, from a photo
of someone photographing their restaurant meal on a phone. Every other
non-hot-dog scored 0.06 or below.

That means the default 0.5 threshold is too strict for this model on this task.
Sweeping the threshold:

| Threshold | Missed hot dogs | False alarms | Accuracy |
|---|---|---|---|
| 0.50 | 4 | 0 | 98.4% |
| 0.30 | 1 | 0 | **99.6%** |
| 0.15 | 1 | 0 | **99.6%** |
| 0.10 | 1 | 1 | 99.2% |

Anywhere from 0.15 to 0.30 gives **249 / 250**.

### Looking at the misses

The four hot dogs scored below 0.5:

| Image | P(hot dog) | What's in it |
|---|---|---|
| `hotdog/0077` | 0.06 | A dark, grainy photo of a person laughing while holding a half-eaten hot dog, which is mostly hidden by their hand. |
| `hotdog/0015` | 0.41 | Sausages on a grill, with no buns. |
| `hotdog/0000` | 0.44 | Two people at a table, one squeezing mustard onto a hot dog. There's a plate of something else in the foreground. |
| `hotdog/0064` | 0.47 | A person in a diner booth taking a bite of a hot dog, shot from a distance. |

Three of them are near-misses at 0.41–0.47, where the hot dog is small or
partly hidden in a wider scene. The model leaned yes but wasn't confident.

`0015` is arguably a labelling question rather than a model error. Whether
"sausages on a grill" counts as a hot dog depends on who's asking. d1 sitting
right on the fence at 0.41 is a fair reflection of that ambiguity.

`0077` is the only real miss: the hot dog is small, dark, and mostly hidden.
It's the one image a lower threshold can't rescue.

## Conclusion

**For a task this simple, d1 doesn't need fine-tuning.** With a one-line
question and no prompt engineering, it reached 98.4% accuracy, and 99.6% after
choosing a better threshold, with zero false positives. The scores it returns
are well separated, so the decision boundary is easy to pick from a handful of
labelled examples.

The general lessons:

1. **Try zero-shot first.** For concepts an ordinary vision model already
   knows, such as common objects, scenes, and people, d1 works out of the box.
   Measure on a few hundred labelled examples before investing in training.
2. **Tune the threshold, not just the model.** d1's probabilities are
   uncalibrated softmax scores. On this task the classes separate almost
   perfectly, but 0.5 isn't where the separation lies. Pick the threshold from
   your own labelled data.
3. **Tighten the question when the definition is fuzzy.** If you need a
   specific answer for edge cases like bun-less sausages, say so in the
   question. For example, "Is there a sausage in a bun?" or a `criteria` block
   with `true`/`false` descriptions. That costs nothing compared with
   fine-tuning.
4. **Reserve fine-tuning for when it's earned**:
   - Categories that are specific to your domain and that a general vision model
     wouldn't know, such as your product SKUs or defect types.
   - Images that look very different from everyday photos, such as medical
     scans, microscopy, or industrial cameras.
   - Accuracy targets beyond what the question wording and threshold choice
     can reach.

   Note that the mlx-vlm d1 package covers inference only.

### Caveats

- **Small, easy dataset.** 250 images with clear class boundaries is a
  sanity check, not a benchmark. Harder data, such as tiny hot dogs in crowded
  scenes, would lower the numbers.
- **Low-resolution inputs.** Most images were around 150×220 pixels, well below
  the 1024×1024 the model can take. Higher-resolution photos would likely help
  on the partly hidden cases.
- **The threshold was picked on the same data it was scored on.** The 99.6%
  figure is optimistic. The honest out-of-the-box number is 98.4%.
- **One run, one wording, bfloat16.** Per the mlx-vlm docs, bfloat16 shifts
  probabilities by up to a few hundredths compared with float32. That's
  irrelevant here, but worth knowing near a threshold.

## Reproduce it

```bash
git clone https://github.com/changtimwu/d1-exp && cd d1-exp
uv sync
uv run experiments/fetch_hotdog.py    # downloads 250 images to data/hotdog/
uv run experiments/hotdog_eval.py     # prints accuracy, AUC, confusion matrix, misses
```

Per-image scores are written to `outputs/hotdog_eval.json`.
