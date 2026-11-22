"""Extract native Jittor PointCLIP few-shot RN101 features in stable ModelNet40 order."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

import h5py
import numpy as np

from scripts import ROOT
from scripts.modelnet40 import CHAIR, DATA, dataset_index


def select_records(split, classes, train_list):
    records = dataset_index("train" if split == "train" else "test", classes)
    if split == "test":
        if len(records) != 2468:
            raise ValueError("Expected 2,468 test samples")
        return records
    if split == "chair":
        if records[3][0] != CHAIR:
            raise ValueError("Test row 3 no longer matches the chair used in the projection experiment")
        return [records[3]]
    manifest = json.loads(train_list.read_text())
    if isinstance(manifest, dict):
        manifest = manifest.get("sample_ids", manifest.get("samples"))
    if not isinstance(manifest, list):
        raise ValueError("Expected an ordered list of training IDs in the manifest")
    ids = [entry["sample_id"] if isinstance(entry, dict) else entry for entry in manifest]
    if len(ids) != 640 or len(set(ids)) != 640:
        raise ValueError("Expected 640 unique training IDs")
    by_name = {r[0]: r for r in records}
    selected = [by_name[name] for name in ids]
    counts = np.bincount([r[1] for r in selected], minlength=40)
    if not np.all(counts == 16):
        raise ValueError("Training manifest must contain precisely 16 objects from each class")
    return selected


def read_points(records):
    # Group disk reads by shard, then return each object in the manifest's order.
    by_shard = defaultdict(list)
    for dst, (_, _, shard, row) in enumerate(records):
        by_shard[shard].append((dst, row))
    points = np.empty((len(records), 1024, 3), dtype=np.float32)
    for shard, entries in by_shard.items():
        destinations, rows = np.asarray(entries).T
        first, last = int(rows.min()), int(rows.max()) + 1
        with h5py.File(shard, "r") as h5:
            block = h5["data"][first:last, :1024, :3]
        points[destinations] = block[rows - first]
    return points


def augment(pointclouds, seed, start):
    """Author's per-axis scale, translation and shuffle; deterministic across shard boundaries."""
    for offset, cloud in enumerate(pointclouds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, start + offset]))
        scale = rng.uniform(2.0 / 3, 1.5, 3)
        shift = rng.uniform(-0.2, 0.2, 3)
        cloud[:] = (cloud * scale + shift).astype(np.float32)
        rng.shuffle(cloud)


def cache_jobs(args, stop, total):
    """Contiguous seeds can be divided across GPUs without changing any realization."""
    if args.output is not None:
        return [(args.augmentation_seed, args.output)]
    suffix = "" if args.start == 0 and stop == total else f"_part{args.start:04d}_{stop:04d}"
    return [(seed, args.output_dir / f"train_aug_{seed:06d}{suffix}.npz")
            for seed in range(args.augmentation_seed, args.augmentation_seed + args.augmentation_count)]


def check_completed(path, seed, common):
    with np.load(path, allow_pickle=False) as cache:
        for key in ("sample_ids", "labels", "classnames", "split", "start", "stop"):
            np.testing.assert_array_equal(cache[key], common[key], err_msg=f"{path}: {key}")
        if cache["augmentation_seed"].item() != seed:
            raise ValueError(f"Augmentation seed disagrees with cache name: {path}")
        if cache["features"].shape != (len(common["labels"]), 10, 512) or cache["text"].shape != (40, 512):
            raise ValueError(f"Invalid completed feature cache: {path}")


def write_cache(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=str(path.parent), suffix=".npz", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        np.savez(temporary, **values)
        if path.exists():
            raise FileExistsError(path)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def encode_points(jt, visual, project_views, points, args, seed):
    if seed is not None:
        points = points.copy()
        augment(points, seed, args.start)
    features = np.empty((len(points), 10, 512), np.float32)
    output = {}
    started = perf_counter()
    for start in range(0, len(points), args.batch_size):
        end = min(len(points), start + args.batch_size)
        depth_batch = project_views(jt.array(points[start:end])).reshape((end - start) * 10, 224, 224)
        if args.split == "chair":
            output["points"] = points[0]
            output["depths"] = depth_batch.numpy().copy()
        images = depth_batch.unsqueeze(1).repeat(1, 3, 1, 1)
        features[start:end] = visual(images).numpy().reshape(end - start, 10, 512)
        if start == 0 or end % (25 * args.batch_size) == 0 or end == len(points):
            print(f"{args.split} rows {args.start}:{args.start+len(points)} seed {seed}: encoded "
                  f"{end}/{len(points)} in {perf_counter()-started:.1f}s", flush=True)
    if not np.isfinite(features).all() or (np.linalg.norm(features, axis=-1) == 0).any():
        raise ValueError("Invalid RN101 image embeddings")
    output.update(features=features, augmentation_seed=np.int64(-1 if seed is None else seed))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("chair", "train", "test"))
    parser.add_argument("--train-list", type=Path, help="ordered 16-shot sample ID manifest for train")
    parser.add_argument("--start", type=int, default=0, help="inclusive split index")
    parser.add_argument("--stop", type=int, help="exclusive split index")
    parser.add_argument("--augmentation-seed", type=int, help="seed for one realization of train augmentation")
    parser.add_argument("--augmentation-count", type=int, default=1,
                        help="number of consecutive seeds to extract when using --output-dir")
    parser.add_argument("--batch-size", type=int, default=4, help="point clouds per batch (10 images per object)")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="new .npz path for a single extraction")
    destination.add_argument("--output-dir", type=Path,
                             help="resumable directory with a separate .npz file for each augmentation seed")
    args = parser.parse_args()
    if args.batch_size < 1 or args.augmentation_count < 1:
        parser.error("Batch size and augmentation count must be positive")
    if args.output is not None and (args.output.suffix != ".npz" or args.output.exists() or args.augmentation_count != 1):
        parser.error("Use a new .npz path for one extraction, or --output-dir for a seed range")
    if args.output_dir is not None and args.augmentation_seed is None:
        parser.error("Set --augmentation-seed for the first seed in --output-dir")
    if (args.train_list is not None) != (args.split == "train"):
        parser.error("Supply --train-list when extracting the training split")
    if args.augmentation_seed is not None and (args.augmentation_seed < 0 or args.split != "train"):
        parser.error("--augmentation-seed accepts a nonnegative integer for the training split")
    classes = (DATA / "shape_names.txt").read_text().splitlines()
    if len(set(classes)) != 40 or len(classes) != 40:
        raise ValueError("Expected 40 distinct class names")
    records = select_records(args.split, classes, args.train_list)
    stop = len(records) if args.stop is None else args.stop
    if not 0 <= args.start < stop <= len(records):
        parser.error(f"Expected 0 <= start < stop <= {len(records)}")
    selected = records[args.start:stop]
    common = dict(sample_ids=np.asarray([r[0] for r in selected]),
                  labels=np.asarray([r[1] for r in selected], dtype=np.int64),
                  classnames=np.asarray(classes), split=np.asarray(args.split),
                  start=np.int64(args.start), stop=np.int64(stop))
    pending = []
    for seed, path in cache_jobs(args, stop, len(records)):
        if path.is_file() and args.output_dir is not None:
            check_completed(path, seed, common)
            print(f"Reusing completed cache {path}", flush=True)
        else:
            pending.append((seed, path))
    if not pending:
        return
    points = read_points(selected)

    for key in ("use_mkl", "use_cutt", "use_cutlass", "use_nccl"):
        os.environ.setdefault(key, "0")
    os.environ.setdefault("conv_opt", "1")
    os.environ.setdefault("cache_name", "pointclip_projection")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import jittor as jt
    sys.path.insert(0, str(ROOT / "src"))
    from pointclip_jittor.geometry.views import project_fewshot_views
    from pointclip_jittor.encoders.text import FEWSHOT_TEMPLATE, TextEncoder, tokenize_modelnet40
    from pointclip_jittor.encoders.visual import ModifiedResNet
    from pointclip_jittor.encoders.weights import load_tower

    jt.flags.use_cuda = 1
    jt.flags.use_tensorcore = 1
    with jt.no_grad():
        visual = ModifiedResNet("RN101")
        text = TextEncoder(embed_dim=512)
        with np.load(ROOT / "weights/clip/RN101.npz", allow_pickle=False) as weights:
            load_tower(visual, weights, "visual.")
            load_tower(text, weights, "")
            common["logit_scale"] = np.exp(weights["logit_scale"].astype(np.float32)).reshape(())
        tokens = tokenize_modelnet40(classes, ROOT / "reference/upstream/PointCLIP", FEWSHOT_TEMPLATE)
        common["text"] = text(tokens).numpy().copy()
        common["tokens"] = tokens.numpy().copy()
        if common["text"].shape != (40, 512) or not np.isfinite(common["text"]).all():
            raise ValueError("Invalid RN101 text embeddings")
        del text
        for seed, path in pending:
            values = encode_points(jt, visual, project_fewshot_views, points, args, seed)
            write_cache(path, {**common, **values})
            print(f"Saved {values['features'].shape} raw visual features to {path}", flush=True)
            del values


if __name__ == "__main__":
    main()
