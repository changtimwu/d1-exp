# d1-exp

Experiments with [LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) on Apple
Silicon via [mlx-vlm's d1 model](https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/d1).

d1 answers structured questions (`bool`, `choice`, `score`) about a text/JSON state
and/or images in a single forward pass — no tokens are generated.

## Setup

```bash
uv sync
```

The first run downloads the checkpoint (~6 GB, bfloat16) to the Hugging Face cache.

## Run

```bash
uv run experiments/quickstart.py

# CLI
uv run python -m mlx_vlm decide --model LiquidAI/d1-3B \
  --state "I was charged twice, please refund one." \
  --questions '{"refund": {"type": "bool", "instructions": "Is the customer asking for a refund?"}}'

# images only (repeat --image for several)
uv run python -m mlx_vlm decide --model LiquidAI/d1-3B \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --questions '{"cats": {"type": "choice", "instructions": "How many cats?", "criteria": ["one", "two", "more"]}}'
```

## Layout

- `experiments/` — scripts
- `images/` — local test images
- `outputs/` — results (git-ignored)
