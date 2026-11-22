"""Evaluate the five student-trained adapters using the original PyTorch XYZ-to-CLIP path."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

from torch_fs_reference import ROOT, build_reference, forward_with_control, normalize
from torch_zero_reference import test_index


@torch.no_grad()
def evaluate(args):
    upstream = ROOT / "reference/upstream/PointCLIP"
    sys.path[:0] = [str(upstream), str(upstream / "Dassl3D")]
    classes, ids, labels, shards = test_index(ROOT / "data/modelnet40_ply_hdf5_2048")
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / "predictions.npz"
    if dest.exists():
        raise FileExistsError(dest)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    model, tokens, text = build_reference(ROOT, classes)
    text = normalize(text)
    scale = model.logit_scale.exp()
    heads = {}
    for k in (1, 2, 4, 8, 16):
        head = deepcopy(model.adapter)
        path = args.trained_root / f"{k}shot" / "training/adapter.pt"
        head.load_state_dict(torch.load(str(path), map_location="cpu"), strict=True)
        heads[f"trained_{k}shot"] = head.eval()
    scores = {key: [] for key in ["no_adapter", "author", *heads]}
    count = 0
    for shard in shards:
        with h5py.File(str(shard), "r") as h5:
            for start in range(0, len(h5["data"]), args.batch_size):
                points = torch.from_numpy(h5["data"][start:start+args.batch_size, :1024, :]).float().cuda()
                author, bypass, views = forward_with_control(model, points, text)
                logits = {"no_adapter": bypass, "author": author}
                for name, head in heads.items():
                    logits[name] = (scale * normalize(head(views))) @ text.t()
                for key, value in logits.items():
                    scores[key].append(value.cpu().numpy())
                count += len(points)
                if start == 0 or count % (16*args.batch_size) == 0 or count == len(ids):
                    print(f"Raw RN101 projected and classified {count}/{len(ids)}", flush=True)
    packed = {}
    for key, batches in scores.items():
        array = np.concatenate(batches)
        if array.shape != (len(ids), 40) or not np.isfinite(array).all():
            raise ValueError(f"Invalid full scores for {key}")
        pred = array.argmax(1)
        packed[key+"_logits"], packed[key+"_predictions"] = array, pred
        print(key, int((pred == labels).sum()), "/", len(ids), flush=True)
    np.savez_compressed(dest, sample_ids=ids, labels=labels, classnames=np.array(classes), tokens=tokens,
                        batch_size=args.batch_size, image_dtype=str(model.dtype), **packed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    evaluate(args)
