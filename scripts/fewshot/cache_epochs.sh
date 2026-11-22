#!/usr/bin/env bash
# Encode each epoch's independently augmented point clouds once with frozen RN101.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
source configs/environment/fewshot_cuda.sh
devices=("$@")
if (( ${#devices[@]} == 0 )); then devices=(0); fi
if (( ${#devices[@]} > 250 )); then exit 2; fi
cache=.cache/fewshot-rn101
mkdir -p "$cache/augmentations"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
workers=()
for index in "${!devices[@]}"; do
    device=${devices[index]}
    [[ "$device" =~ ^[0-9]+$ ]] || { echo "Expected a numeric GPU ID" >&2; exit 2; }
    start=$((index * 250 / ${#devices[@]}))
    stop=$(((index + 1) * 250 / ${#devices[@]}))
    CUDA_VISIBLE_DEVICES="$device" .venv-jittor/bin/python -u -m scripts.fewshot.extract_features \
        --split train --train-list manifests/modelnet40_16shot_seed2.json \
        --augmentation-seed "$((1000 + start))" --augmentation-count "$((stop - start))" \
        --output-dir "$cache/augmentations" --batch-size 16 > "$cache/encode_${start}_${stop}.log" 2>&1 &
    workers+=("$!")
done
result=0
for pid in "${workers[@]}"; do wait "$pid" || result=1; done
exit "$result"
