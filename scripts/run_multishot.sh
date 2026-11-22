#!/usr/bin/env bash
# Run one nested-support experiment. Pass the CUDA device via CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
shot=${1:?Usage: bash scripts/run_multishot.sh SHOTS NEW_OUTPUT_DIRECTORY}
output=${2:?Expected a new experiment directory}
case "$shot" in 1|2|4|8|16) ;; *) echo "Unsupported shot count: $shot" >&2; exit 2;; esac
source configs/environment/fewshot_cuda.sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
cache=.cache/fewshot-rn101
mkdir -p "$(dirname -- "$output")"
mkdir "$output"
.venv-jittor/bin/python -u scripts/fit_fewshot.py \
    --train "$cache"/augmentations/train_aug_*.npz --shots "$shot" --seed 2 \
    --support-manifest manifests/modelnet40_16shot_seed2.json \
    --epochs 250 --augmentation-mode epoch --output "$output/training" > "$output/train.log" 2>&1
.venv-jittor/bin/python -u scripts/eval_fewshot.py test --features "$cache/test.npz" \
    --reference runs/pytorch/fewshot_reference/predictions.npz \
    --author weights/adapter/rn101_16shot.npz --trained "$output/training/adapter.npz" \
    --output "$output/eval" > "$output/eval.log" 2>&1
echo "Results: $output/eval/metrics.json"
