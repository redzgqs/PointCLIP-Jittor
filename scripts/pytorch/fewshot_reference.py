"""Run PointCLIP's upstream RN101 adapter and a matched bypass on ModelNet40."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from scripts import ROOT
from scripts.modelnet40 import CHAIR, test_index
from scripts.pytorch import to_numpy

VIEWS = 10
CHANNELS = 512


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def build_reference(root, classes):
    import torch
    from clip import clip
    from dassl.config import get_cfg_default
    from trainers.fewshot import CUSTOM_TEMPLATES, PointCLIP_Model

    upstream = root / "reference/upstream/PointCLIP"
    config = get_cfg_default()
    config.merge_from_file(str(upstream / "configs/datasets/modelnet40.yaml"))
    config.merge_from_file(str(upstream / "configs/trainers/PointCLIP_FS/rn101.yaml"))
    config.freeze()
    checkpoint = torch.jit.load(str(root / "weights/clip/RN101.pt"), map_location="cpu")
    clip_model = clip.build_model(checkpoint.state_dict()).cuda().eval()
    del checkpoint
    model = PointCLIP_Model(config, classes, clip_model).cuda().eval()
    payload = torch.load(str(root / "weights/adapter/model-best.pth.tar"), map_location="cpu")
    model.adapter.load_state_dict(payload["state_dict"], strict=True)
    prompts = [CUSTOM_TEMPLATES["ModelNet40"].format(c.replace("_", " ")) for c in classes]
    tokens = torch.cat([clip.tokenize(p) for p in prompts])
    text_repeated = model.textual_encoder()
    return model, to_numpy(tokens), text_repeated


def forward_with_control(model, points, flat_text):
    # Capture the exact view features the original forward passes to its adapter.
    # The control applies the same subsequent normalization to an identity adapter.
    captured = []

    def capture(module, args, result):
        captured.append(result)

    hook = model.visual_encoder.register_forward_hook(capture)
    try:
        logits = model(points)
    finally:
        hook.remove()
    raw = captured[0].reshape(len(points), VIEWS, CHANNELS)
    bypass = normalize(raw.reshape(len(points), VIEWS * CHANNELS))
    control = (model.logit_scale.exp() * bypass) @ flat_text.t()
    return logits, control, raw


def run(args):
    import h5py
    import torch

    upstream = args.root / "reference/upstream/PointCLIP"
    sys.path[:0] = [str(upstream), str(upstream / "Dassl3D")]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    classes, ids, labels, shards = test_index(args.root / "data/modelnet40_ply_hdf5_2048")
    model, tokens, repeated = build_reference(args.root, classes)
    flat_text = normalize(repeated)
    out = args.root / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "predictions.npz").exists():
        raise FileExistsError("A full evaluation already exists at " + str(out))
    if ids[3] != CHAIR:
        raise ValueError("ModelNet40 test ordering changed")
    with h5py.File(str(shards[0]), "r") as h5:
        xyz = h5["data"][3, :1024, :].astype(np.float32)
    with np.load(args.root / "runs/examples/chair_projection/reference.npz", allow_pickle=False) as old:
        np.testing.assert_array_equal(xyz, old["points"])
    chair = torch.from_numpy(xyz).unsqueeze(0).cuda()
    logits, control, visual = forward_with_control(model, chair, flat_text)
    depth = model.get_img(chair)
    np.savez_compressed(out / "chair.npz", sample_id=np.array(CHAIR), label=labels[3],
                        points=xyz, depth=to_numpy(depth), view_features=to_numpy(visual[0]),
                        text_features=to_numpy(repeated[:, :CHANNELS]), tokens=tokens,
                        logits=to_numpy(logits[0]), control_logits=to_numpy(control[0]),
                        logit_scale=to_numpy(model.logit_scale.exp()), classnames=np.array(classes))
    print("Chair reference:", out / "chair.npz", flush=True)
    if args.chair_only:
        return

    author_rows, control_rows = [], []
    count = 0
    clock = perf_counter()
    for path in shards:
        with h5py.File(str(path), "r") as h5:
            data = h5["data"]
            for start in range(0, len(data), args.batch_size):
                batch = torch.from_numpy(data[start:start + args.batch_size, :1024, :]).cuda()
                logits, control, _ = forward_with_control(model, batch, flat_text)
                author_rows.append(to_numpy(logits))
                control_rows.append(to_numpy(control))
                count += len(batch)
                if start == 0 or len(author_rows) % 20 == 0 or count == len(ids):
                    print("Scored {}/{}".format(count, len(ids)), flush=True)
    logits, control = np.concatenate(author_rows), np.concatenate(control_rows)
    if logits.shape != (len(ids), len(classes)) or control.shape != logits.shape:
        raise ValueError("Unexpected output counts")
    if not np.isfinite(logits).all() or not np.isfinite(control).all():
        raise ValueError("Non-finite scores in one or both classifiers")
    predictions, bypass = logits.argmax(1), control.argmax(1)
    np.savez_compressed(out / "predictions.npz", logits=logits, control_logits=control,
                        predictions=predictions, control_predictions=bypass, labels=labels,
                        sample_ids=ids, classnames=np.array(classes), tokens=tokens)
    correct = int((predictions == labels).sum())
    control_correct = int((bypass == labels).sum())
    metrics = dict(framework="PyTorch", model="PointCLIP V1 RN101",
                   views=VIEWS, resolution=224, test_objects=count, correct=correct,
                   accuracy_percent=100 * correct / count, control_correct=control_correct,
                   control_accuracy_percent=100 * control_correct / count,
                   batch_size=args.batch_size, elapsed_seconds=round(perf_counter() - clock, 2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("runs/pytorch/fewshot_reference"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chair-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    import torch
    with torch.no_grad():
        run(args)


if __name__ == "__main__":
    main()
