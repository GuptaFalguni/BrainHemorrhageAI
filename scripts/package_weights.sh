#!/usr/bin/env bash
# Package deployable model weights for GitHub Release (not committed to git).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REQUIRED=(
  checkpoints/experiment_best30h/best_model.pt
  checkpoints/nnunet_dataset501_fold0/checkpoint_best.pth
  checkpoints/nnunet_dataset501_fold0/nnUNetPlans.json
  checkpoints/nnunet_dataset501_fold0/dataset.json
)

for f in "${REQUIRED[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required file: $f" >&2
    exit 1
  fi
done

mkdir -p dist
OUT=dist/model-weights-v1.tar.gz
rm -f "$OUT"
tar -czf "$OUT" "${REQUIRED[@]}"
echo "Created $OUT ($(du -h "$OUT" | cut -f1))"
echo "Publish with:"
echo "  gh release create v1.0.0 $OUT --title \"v1.0.0\" --notes \"Deployable MONAI + nnU-Net checkpoints for Docker.\""
