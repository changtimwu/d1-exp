# Flagging Violence in Surveillance Video with a Decision Model: d1 on RWF-2000

*Third experiment with LiquidAI's d1-3B on Apple Silicon via mlx-vlm — October 2026.
Earlier: [Hot Dog or Not](hotdog-zero-shot.md) · [Factory Defects on MVTec AD](mvtec-zero-shot.md).*

## The question

A video library, such as a surveillance archive, a user-upload platform, or a
media asset manager, often needs to **flag scenes before people see them**,
whether to blur a thumbnail, route a clip to review, or keep it out of search
results. The two classic categories are violence and NSFW content.

d1-3B is a decision model with vision. It answers typed questions about images
with a probability, in a single forward pass and without generating text. The
earlier experiments showed it works out of the box for simple concepts, and that
it's limited by what a single image can tell it. Violence in video tests both
points at once. A fight is a common concept, but it happens *over time*, and a
single still of people close together is ambiguous.

So the question is: **can d1 flag violent surveillance clips as it is, or
does it need fine-tuning? And does it help to show it several frames?**

## Setup

### Hardware and software

Same as before:

| | |
|---|---|
| Machine | Apple M1 Max, 32 GB unified memory |
| OS | macOS 26.5.1 |
| Runtime | Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg for frame extraction |
| Inference | [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.7.10 (MLX 0.32.4) |
| Model | `LiquidAI/d1-3B`, bfloat16, as published (no quantization) |

### Dataset

[RWF-2000](https://arxiv.org/abs/1911.05913) (Real World Fighting) contains
2,000 five-second clips from real surveillance cameras, half showing fights and
half not. The non-fight clips are deliberately hard: crowds, people hugging,
sports, and people moving fast in small spaces.

We used the **validation split, 400 clips (200 Fight, 200 NonFight)**, which is
the split the RWF-2000 paper reports test results on. The clips came from the
Hugging Face copy
[`A1mal/RWF-2000-Dataset`](https://huggingface.co/datasets/A1mal/RWF-2000-Dataset)
and are 1280×720 at 10 fps. Image quality varies widely: night footage, fisheye
lenses, re-encoded phone recordings of monitors, and burned-in watermarks.

From each clip, ffmpeg took **4 evenly spaced frames** at 0.6, 1.8, 3.0 and
4.2 seconds, scaled to 640 px wide.

### Three ways to turn a clip into one decision

d1 takes any number of images in one input, so there are several ways to
judge a clip:

| Method | What d1 sees | Model passes per clip |
|---|---|---|
| **single** | Only the middle frame (3.0 s) | 1 |
| **max** | Each of the 4 frames asked separately; the highest score wins | 4 |
| **multi** | All 4 frames together, introduced as "4 frames, in order, from a 5-second surveillance camera clip" | 1 (4 images) |

The question is a `bool`, with descriptions of what "yes" and "no" mean:

```python
CRITERIA = {
    "true": "Someone is fighting, hitting, punching, kicking, shoving, or attacking another person.",
    "false": "People are walking, standing, talking, or doing normal activities with no physical aggression.",
}
# single / max:  "This is a frame from a surveillance camera. Is there physical violence in it?"
# multi:         "Is there physical violence in this clip?"
```

As in the other experiments, nothing was trained or tuned, and the first
wording was the only one tried.

## Results

All 400 clips, running all three methods (5 model passes per clip), took **30
minutes, about 4.5 seconds per clip**.

### Headline

| Method | AUROC | Accuracy @0.5 | Recall @0.5 | Specificity @0.5 | Best accuracy (cutoff) |
|---|---|---|---|---|---|
| single | 0.921 | 79.0% | 64.0% | 94.0% | 84.5% (0.09) |
| **max** | 0.962 | **91.0%** | **91.0%** | 91.0% | 91.0% (0.50) |
| **multi** | **0.965** | 86.0% | 76.0% | **96.0%** | 90.2% (0.16) |

Recall is the share of fight clips flagged. Specificity is the share of
non-fight clips passed.

- **Using more frames clearly helps.** Moving from one frame to four raises
  AUROC from 0.921 to about 0.965 and cuts the error rate roughly in half.
  Averaging the 4 per-frame scores gives 0.957, between the two.
- **max is the best default.** At the out-of-the-box 0.5 cutoff it reaches 91%
  accuracy with balanced errors: 91% of fights caught and 91% of non-fights
  passed. That's the only setting in this series of experiments where 0.5
  happened to be the best cutoff.
- **multi is the most conservative.** It has the fewest false alarms (96%
  specificity at 0.5) and the best ranking overall. Lowering its cutoff
  trades a few false alarms for many more catches.

### For context

The RWF-2000 paper's own supervised model, a Flow Gated Network trained on the
1,600 training clips, reports **87.25% accuracy** on this split. Later
supervised methods report higher numbers, which we didn't check. Zero-shot d1 with 4
frames reaches **91.0% (max, at 0.5)**, so it's competitive with the
dataset's original trained baseline without seeing a single training clip.
This is a rough comparison: we use 4 still frames and a different copy of
the data, and we didn't re-run their model.

### Choosing the cutoff for the job

A video library rarely wants "best accuracy". It either wants to *catch
everything* and send it for review, or to *auto-flag only when sure*:

| Cutoff | max: recall / specificity | multi: recall / specificity |
|---|---|---|
| 0.50 | 91.0% / 91.0% | 76.0% / 96.0% |
| 0.30 | 93.0% / 86.0% | 83.5% / 92.5% |
| 0.20 | 97.5% / 79.0% | 88.0% / 88.5% |
| 0.10 | **99.0%** / 70.5% | 96.0% / 83.0% |
| 0.05 | 99.5% / 61.0% | 99.5% / 74.5% |

- **Review queue, where a missed fight is unacceptable:** use *max* at 0.10.
  It catches 99% of fights while sending about 30% of normal clips to review.
- **Auto-blur or auto-hide, where false alarms are costly:** use *multi* at
  0.5. It wrongly hides only 4% of normal clips and still catches three out of
  four fights.

### Looking at the errors

**Most confident false alarm, `NonFight/xRbCHA5YTi8_0`, scored 0.97.** Night
footage of two men grabbing each other in a parking area. Looking at the
frame, it's hard to argue this isn't physical aggression. It may be a scuffle
that the dataset labels as non-fight, or a mislabel. The second-highest,
`nVdn1krnTCQ_0` at 0.82, shows a man swinging his arm at another man in a fast-food
restaurant. Many of d1's "false alarms" are confrontational scenes that a human
reviewer would want to see anyway.

**Worst miss, `Fight/bW2vHhYbzHM_0`, scored 0.01 on every frame.** A wide night
shot of a full parking lot with a timestamp overlay. The people involved are a
handful of pixels at the right edge of the frame. At 640 px wide, there's
almost nothing for the model to see. That's a resolution and framing problem
rather than one of understanding. The next-lowest fight clips have frame
scores between 0.01 and 0.35. We didn't inspect each of them, so whether they
share the same cause, a **small, distant, or blurred fight**, is a likely
explanation rather than a confirmed one.

## What about NSFW?

NSFW wasn't tested. Measuring it properly means downloading a few hundred
explicit images, and public NSFW datasets are scraped from the web and can't be
guaranteed clean. We chose not to put that on a work machine. The method
carries over directly, though:

```python
NSFW_Q = {"nsfw": {
    "type": "choice",
    "instructions": "How should this video frame be rated for a general audience?",
    "criteria": {
        "safe": "No nudity or sexual content",
        "suggestive": "Revealing clothing, swimwear, or suggestive poses, but no nudity",
        "explicit": "Nudity or sexual activity",
    },
}}
```

Using a 3-way `choice` rather than a yes/no `bool` gives you a separate cutoff
for the grey zone, such as swimwear, art, or medical content, which is where
NSFW filters usually make mistakes. Evaluate it the same way as here: on a
**vetted, labelled internal set**, score each method, and pick cutoffs from the
recall and specificity table. Expect the same lessons: scan several frames per
scene and take the max, and calibrate the cutoff rather than trusting 0.5.

## Conclusion

**d1 can flag violent surveillance clips without fine-tuning, provided it sees
more than one frame.**

- A single frame gets 0.921 AUROC and misses a third of fights at 0.5. Four
  frames reach about 0.965 AUROC and 91% accuracy at the default cutoff, which
  beats the trained baseline published with the dataset (87.25%).
- Errors are mostly confrontational non-fights, which a reviewer would want to
  see anyway, and fights too small or far away for a 640 px frame.
- Cutoffs decide the use case: about 99% recall for a review queue, or about
  4% false alarms for auto-hiding.

### How to use it in a video library

1. **Sample frames, not clips.** For example, 1–2 frames per second, grouped
   into windows of a few seconds. Score each window by *max* over its frames,
   or pass the window's frames together (*multi*).
2. **Flag segments rather than whole videos.** Store per-window scores, so a
   reviewer can jump straight to the timestamp.
3. **Ask several questions at once.** d1 reads the frames once, and extra
   questions such as violence, weapons, blood, or nudity reuse that
   cached image prefix, so they cost much less than separate passes.
4. **Set cutoffs per action** on a labelled sample of *your* footage: a low
   cutoff for "send to review", a high one for "auto-hide".
5. **Fine-tune only if needed.** That would be the case if your cameras are
   very different (thermal, very wide angle, very low resolution), your
   definition of violence is narrower than "physical aggression", or a review
   queue at 30% of footage is too expensive. Higher resolution and better
   framing are worth trying first, since distant fights were the main miss.
   The mlx-vlm d1 package covers inference only.

### Caveats

- **Still frames, not video.** d1 sees 4 frames at most; it never sees motion
  directly. Fast events between sampled frames, such as a single punch, can be
  missed.
- **About 4.5 s per clip for all three methods together** on an M1 Max in bf16.
  We didn't time each method separately. For a large archive, sample sparsely
  first, then re-score the flagged windows densely.
- **One wording, one run, one split.** The best-accuracy and cutoff figures
  were chosen on the same 400 clips they were scored on. The 0.5 rows are the
  honest out-of-the-box numbers.
- **Rough comparison with the literature.** The published 87.25% is from a model
  that sees full video with optical flow. We didn't re-run it, and newer
  supervised methods score higher.
- **Label noise.** Some RWF-2000 "non-fights" look like scuffles, which caps
  measured specificity.

## Reproduce it

```bash
git clone https://github.com/changtimwu/d1-exp && cd d1-exp
uv sync
uv run experiments/fetch_rwf2000.py           # 400 validation clips (~400 MB) into data/rwf2000/
uv run experiments/extract_rwf2000_frames.py  # 4 frames per clip, 640 px wide
uv run experiments/rwf2000_eval.py            # ~30 min on an M1 Max; single / max / multi table
```

Per-clip scores, including each frame's score, are written to
`outputs/rwf2000_eval.json`.
