#!/usr/bin/env bash
# bash scripts/run_fewshot.sh chair 0       or: bash scripts/run_fewshot.sh all 0 1 2 3
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
stage=${1:-chair}
case "$stage" in chair|all) ;; *) echo 'Use chair or all' >&2; exit 2;; esac
if (( $# )); then shift; fi
gpus=("$@")
if (( ${#gpus[@]} == 0 )); then gpus=(0); fi
export CUDA_VISIBLE_DEVICES="${gpus[0]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
cache=.cache/fewshot-rn101
reference=runs/pytorch/fewshot_reference
output="${POINTCLIP_RUN_DIR:-runs/jittor/fewshot/$(date -u +%Y%m%d_%H%M%S)_s2_$$}"
author=weights/adapter/rn101_16shot.npz
mkdir -p "$cache" "$(dirname -- "$output")"
mkdir "$output"
echo "Results: $output"
jittor=.venv-jittor/bin/python
$jittor scripts/prepare_fewshot_cuda.py
.venv/bin/python scripts/prepare_fewshot.py
if [[ ! -f "$reference/predictions.npz" ]]; then
    if [[ ! -f runs/examples/chair_projection/reference.npz ]]; then
        .venv/bin/python scripts/show_chair.py
    fi
    flags=()
    if [[ "$stage" == chair ]]; then flags+=(--chair-only); fi
    .venv/bin/python scripts/torch_fs_reference.py --output "$reference" --batch-size 16 "${flags[@]}"
fi
source configs/environment/fewshot_cuda.sh
if [[ ! -f "$cache/chair.npz" ]]; then
    $jittor scripts/features_fs.py --split chair --batch-size 1 --output "$cache/chair.npz"
fi
$jittor scripts/eval_fewshot.py chair --features "$cache/chair.npz" \
    --reference "$reference/chair.npz" --author "$author" --output "$output/chair"
if [[ "$stage" == chair ]]; then exit 0; fi
if [[ ! -f "$cache/test.npz" ]]; then
    $jittor scripts/features_fs.py --split test --batch-size 16 --output "$cache/test.npz"
fi
test_args=(test --features "$cache/test.npz" --reference "$reference/predictions.npz" --author "$author")
$jittor scripts/eval_fewshot.py "${test_args[@]}" --output "$output/author_validation"
bash scripts/cache_fewshot_epochs.sh "${gpus[@]}"
$jittor scripts/fit_fewshot.py --train "$cache"/augmentations/train_aug_*.npz \
    --epochs 250 --augmentation-mode epoch --seed 2 --output "$output/training"
$jittor scripts/eval_fewshot.py "${test_args[@]}" --trained "$output/training/adapter.npz" --output "$output/eval"
echo "Measured results: $output/eval/metrics.json"
