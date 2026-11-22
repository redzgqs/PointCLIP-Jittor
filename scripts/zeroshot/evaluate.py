"""Run the native Jittor PointCLIP RN50 zero-shot benchmark on ModelNet40."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import h5py
import numpy as np

from scripts import ROOT
from scripts.modelnet40 import DATA, shard_index_files


def test_batches(batch_size, limit):
    count = 0
    for path, names in shard_index_files("test"):
        with h5py.File(path, "r") as shard:
            xyz, label = shard["data"], shard["label"]
            if len(names) != len(xyz) or len(label) != len(xyz):
                raise ValueError("The ModelNet40 test files disagree about sample count.")
            take = min(len(xyz), limit - count)
            for start in range(0, take, batch_size):
                end = min(take, start + batch_size)
                yield xyz[start:end, :1024, :], label[start:end].reshape(-1), names[start:end]
            count += take
        if count == limit:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("repo", "table1"), default="repo",
                        help="author repository defaults or fixed ModelNet40 parameters from paper Table 1")
    parser.add_argument("--batch-size", type=int, default=8, help="number of 3D objects per batch")
    parser.add_argument("--limit", type=int, default=2468, help="first N test objects for a quick check")
    parser.add_argument("--output", type=Path, help="new result directory (defaults to a timestamp)")
    args = parser.parse_args()
    if args.batch_size < 1 or not 1 <= args.limit <= 2468:
        parser.error("--batch-size must be positive and --limit must be between 1 and 2468")
    weights = ROOT / "weights/clip/RN50.npz"
    if not weights.is_file():
        parser.error("Convert RN50.pt first with .venv/bin/python -m scripts.setup.convert_rn50")
    # Run the CLIP convolutions with native Jittor/cuBLAS.
    for key in ("use_mkl", "use_cutt", "use_cutlass", "use_nccl"):
        os.environ.setdefault(key, "0")
    os.environ.setdefault("conv_opt", "1")
    os.environ.setdefault("cache_name", "pointclip_projection")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import jittor as jt
    sys.path.insert(0, str(ROOT / "src"))
    from pointclip_jittor.models.zeroshot import ZeroShotPointCLIP

    jt.flags.use_cuda = 1
    jt.flags.use_tensorcore = 1
    classes = (DATA / "shape_names.txt").read_text().splitlines()
    if len(set(classes)) != 40 or len(classes) != 40:
        raise ValueError("Expected the 40 ModelNet40 classes in original file order.")
    out = args.output or ROOT / "runs/jittor/zero" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    print("Output:", out, flush=True)
    clock = perf_counter()
    scores, labels, ids = [], [], []
    with jt.no_grad():
        model = ZeroShotPointCLIP(weights, classes, ROOT / "reference/upstream/PointCLIP", args.protocol)
        print("Loaded both CLIP towers; encoded 40 prompts.", flush=True)
        for batch, (xyz, label, names) in enumerate(test_batches(args.batch_size, args.limit)):
            logits, features = model(jt.array(xyz))
            values = logits.numpy().copy()
            if values.shape != (len(names), 40) or not np.isfinite(values).all():
                raise ValueError("Invalid zero-shot scores")
            scores.append(values)
            labels.append(label)
            ids.extend(names)
            if "chair/chair_0978.ply" in names:
                chair = names.index("chair/chair_0978.ply")
                np.savez(out / "chair.npz", features=features[chair].numpy().reshape(6, 1024),
                         logits=values[chair], text_raw=model.text_raw.numpy())
            if batch == 0 or (batch + 1) % 25 == 0 or len(ids) == args.limit:
                print(f"Scored {len(ids)}/{args.limit} objects in {perf_counter()-clock:.1f}s", flush=True)
        all_logits, truth = np.concatenate(scores), np.concatenate(labels).astype(np.int64)
        token_ids = model.tokens.numpy()
    predicted = all_logits.argmax(axis=1)
    if len(truth) != args.limit or len(set(ids)) != args.limit:
        raise ValueError("Unexpected sample coverage")
    correct = int((truth == predicted).sum())
    top5 = np.argsort(-all_logits, axis=1, kind="stable")[:, :5]
    report = dict(framework="Jittor", version=jt.__version__, model="PointCLIP V1 RN50", task="zero-shot",
                  protocol=args.protocol, resolution=model.resolution, camera_distance=1.6,
                  shots=0, views=6, view_weights=model.relative_weights, post_search=False, test_objects=len(truth),
                  correct=correct, accuracy_percent=100*correct/len(truth),
                  top5_accuracy_percent=100*float((top5 == truth[:, None]).any(axis=1).mean()),
                  batch_size=args.batch_size, elapsed_seconds=round(perf_counter()-clock, 2))
    np.savez_compressed(out / "predictions.npz", logits=all_logits, predictions=predicted, labels=truth,
                        sample_ids=np.array(ids), classnames=np.array(classes), tokens=token_ids,
                        protocol=np.array(args.protocol))
    with (out / "predictions.csv").open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["object", "true_class", "predicted_class"])
        writer.writerows((name, classes[y], classes[p]) for name, y, p in zip(ids, truth, predicted))
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Top-1: {correct}/{len(truth)} = {report['accuracy_percent']:.2f}% | "
          f"Top-5: {report['top5_accuracy_percent']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
