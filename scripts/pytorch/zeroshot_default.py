"""Evaluate the upstream default PointCLIP RN50 directly on ModelNet40 test XYZ."""
import argparse
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from scripts import ROOT
from scripts.modelnet40 import test_index
from scripts.pytorch import RN50_CHANNELS, RN50_VIEWS


def evaluate(out, batch_size):
    upstream = ROOT / "reference/upstream/PointCLIP"
    sys.path[:0] = [str(upstream), str(upstream / "Dassl3D")]
    from clip import clip
    from trainers.mv_utils_zs import PCViews
    from trainers.zeroshot import CUSTOM_TEMPLATES, PointCLIP_ZS, Textual_Encoder

    out.mkdir(parents=True, exist_ok=True)
    dest = out / "predictions.npz"
    if dest.exists():
        raise FileExistsError(dest)
    torch.backends.cudnn.benchmark = True
    classes, ids, labels, shards = test_index(ROOT / "data/modelnet40_ply_hdf5_2048")
    state = torch.jit.load(str(ROOT / "weights/clip/RN50.pt"), map_location="cpu").state_dict()
    model = clip.build_model(state).cuda().eval()
    prompts = [CUSTOM_TEMPLATES["ModelNet40"].format(c.replace("_", " ")) for c in classes]
    tokens = torch.cat([clip.tokenize(p) for p in prompts]).numpy()
    cfg = SimpleNamespace(DATASET=SimpleNamespace(NAME="ModelNet40"),
                          MODEL=SimpleNamespace(PROJECT=SimpleNamespace(NUM_VIEWS=RN50_VIEWS)))
    projector = PCViews()
    ctx = SimpleNamespace(get_img=projector.get_img, visual_encoder=model.visual,
                          textual_encoder=Textual_Encoder(cfg, classes, model).eval(),
                          num_views=RN50_VIEWS, channel=RN50_CHANNELS, dtype=model.dtype,
                          logit_scale=model.logit_scale, feat_store=[], label_store=[])
    ctx.mv_proj = lambda pc: PointCLIP_ZS.mv_proj(ctx, pc)
    tic = perf_counter()
    parts, count = [], 0
    for shard in shards:
        with h5py.File(str(shard), "r") as h5:
            for start in range(0, len(h5["data"]), batch_size):
                xyz = torch.from_numpy(h5["data"][start:start+batch_size, :1024, :]).float().cuda()
                logits = PointCLIP_ZS.model_inference(ctx, xyz)
                parts.append(logits.detach().cpu().numpy())
                ctx.feat_store.clear()
                ctx.label_store.clear()
                count += len(xyz)
                if len(parts) == 1 or len(parts) % 16 == 0 or count == len(ids):
                    print("Raw XYZ scored {}/{} in {:.1f}s".format(count, len(ids), perf_counter()-tic), flush=True)
    scores = np.concatenate(parts)
    if scores.shape != (len(ids), len(classes)) or not np.isfinite(scores).all():
        raise ValueError("Incomplete or non-finite raw-point scores")
    pred = scores.argmax(1)
    np.savez_compressed(dest, sample_ids=ids, labels=labels, classnames=np.array(classes), tokens=tokens,
                        predictions=pred, logits=scores, protocol="repo", batch_size=batch_size,
                        seconds=perf_counter()-tic)
    print("Correct {}/{} | PyTorch {} | {} | {}".format((pred == labels).sum(), len(ids),
          torch.__version__, torch.cuda.get_device_name(), dest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    with torch.no_grad():
        evaluate(args.output, args.batch_size)
