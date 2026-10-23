#!/bin/sh
# Full MedMNIST run: d1 and base on all three test splits (~100 min on an M1 Max).
# Derma runs last because its download is the slowest; the loop waits for it.
set -e
cd "$(dirname "$0")/.."
for job in "d1 blood" "d1 pneumonia" "base blood" "base pneumonia" "d1 derma" "base derma"; do
  set -- $job
  [ -f "outputs/medmnist_$1_$2.json" ] && [ -z "$FORCE" ] && continue
  until [ -f "data/medmnist/$2mnist_224.npz" ]; do sleep 30; done
  uv run experiments/medmnist_eval.py "$1" "$2"
done
uv run experiments/medmnist_report.py
