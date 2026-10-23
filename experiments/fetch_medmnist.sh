#!/bin/sh
# Download the MedMNIST+ 224 px subsets used in medmnist_eval.py, plus the base
# LFM2.5-VL-3B weights for the d1-vs-base comparison.
# Every file is checked against the md5 published on Zenodo (record 10519652).
# Pneumonia and Blood come from a Hugging Face mirror (Zenodo is slow and drops
# connections); its files are byte-identical. Derma differs there, so it comes from Zenodo.
# MedMNIST is CC BY 4.0, except DermaMNIST which is CC BY-NC 4.0; data stays out of git.
set -e
cd "$(dirname "$0")/../data/medmnist"
ZENODO="https://zenodo.org/records/10519652/files"
MIRROR="https://huggingface.co/datasets/convergedmachine/Enhanced-MedMNIST/resolve/main"

fetch() {  # name md5 base_url
  f="$1_224.npz"
  if [ -f "$f" ] && [ "$(md5 -q "$f")" = "$2" ]; then return; fi
  for i in 1 2 3 4 5; do
    curl -fL -C - -o "$f.part" "$3/$f" && break || sleep 5
  done
  [ "$(md5 -q "$f.part")" = "$2" ] || { echo "md5 mismatch for $f" >&2; exit 1; }
  mv "$f.part" "$f"
}

fetch pneumoniamnist d6a3c71de1b945ea11211b03746c1fe1 "$MIRROR"
fetch bloodmnist b718ff6835fcbdb22ba9eacccd7b2601 "$MIRROR"
fetch dermamnist 8974907d8e169bef5f5b96bc506ae45d "$ZENODO"
cd ../.. && uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('LiquidAI/LFM2.5-VL-3B')"
echo ALL DONE
