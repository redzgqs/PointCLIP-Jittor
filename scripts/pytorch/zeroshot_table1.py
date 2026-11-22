"""Run the fixed RN50 settings from PointCLIP Table 1 on the complete ModelNet40 test split."""

import argparse
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from scripts import ROOT
from scripts.modelnet40 import test_index
from scripts.pytorch import RN50_CHANNELS, RN50_VIEWS, to_numpy


VIEW_WEIGHTS = (3, 9, 5, 4, 5, 4)
RESOLUTION = 121
DISTANCE = 1.6


def run(args):
    import h5py
    import torch

    upstream = args.root / 'reference/upstream/PointCLIP'
    sys.path[:0] = [str(upstream), str(upstream / 'Dassl3D')]
    from clip import clip
    from trainers import mv_utils_zs
    from trainers.zeroshot import CUSTOM_TEMPLATES, PointCLIP_ZS, Textual_Encoder

    classes, ids, labels, shards = test_index(args.root / 'data/modelnet40_ply_hdf5_2048')
    out = args.root / args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    mv_utils_zs.RESOLUTION = RESOLUTION
    mv_utils_zs.TRANS = -DISTANCE  # PCViews subtracts the stored Z translation.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    projector = mv_utils_zs.PCViews()
    state = torch.jit.load(str(args.root / 'weights/clip/RN50.pt'), map_location='cpu').state_dict()
    model = clip.build_model(state).cuda().eval()
    del state

    cfg = SimpleNamespace(DATASET=SimpleNamespace(NAME='ModelNet40'),
                          MODEL=SimpleNamespace(PROJECT=SimpleNamespace(NUM_VIEWS=RN50_VIEWS)))
    prompts = [CUSTOM_TEMPLATES['ModelNet40'].format(c.replace('_', ' ')) for c in classes]
    tokens = torch.cat([clip.tokenize(p) for p in prompts])
    repeated = Textual_Encoder(cfg, classes, model).eval()()
    flat_text = repeated / repeated.norm(dim=-1, keepdim=True)
    scale = model.logit_scale.exp()
    weights = torch.tensor(VIEW_WEIGHTS, dtype=torch.float32, device='cuda').reshape(1, RN50_VIEWS, 1)
    common = dict(protocol=np.array('table1'), resolution=RESOLUTION, camera_distance=DISTANCE,
                  view_weights=np.array(VIEW_WEIGHTS), tokens=to_numpy(tokens), logit_scale=to_numpy(scale),
                  text_features=to_numpy(repeated[:, :RN50_CHANNELS]), text_features_repeated=to_numpy(flat_text))

    scores, count, step = [], 0, 0
    clock = perf_counter()
    print('Scoring {} objects with {}px depth maps and weights {}'.format(len(ids), RESOLUTION, VIEW_WEIGHTS), flush=True)
    for path in shards:
        with h5py.File(str(path), 'r') as h5:
            xyz = h5['data']
            for start in range(0, len(xyz), args.batch_size):
                points = torch.from_numpy(xyz[start:start + args.batch_size, :1024, :].astype(np.float32)).cuda()
                images = PointCLIP_ZS.mv_proj(projector, points).to(model.dtype)
                raw = model.visual(images)
                unit = (raw / raw.norm(dim=-1, keepdim=True)).reshape(-1, RN50_VIEWS, RN50_CHANNELS)
                # Follow the author's search_weights_zs weighting arithmetic with the single published ratio.
                weighted = (unit * weights).reshape(len(points), -1).to(model.dtype)
                logits = (scale * weighted) @ flat_text.t()
                values = to_numpy(logits)
                if values.shape != (len(points), len(classes)) or not np.isfinite(values).all():
                    raise ValueError('Invalid logits at test index {}'.format(count))
                scores.append(values)
                end = count + len(points)
                if args.chair and 'chair/chair_0978.ply' in ids[count:end]:
                    chair = int(np.flatnonzero(ids[count:end] == 'chair/chair_0978.ply')[0])
                    np.savez(out / 'chair.npz', **common, view_features_normalized=to_numpy(unit[chair]),
                             view_features=to_numpy(raw.reshape(-1, RN50_VIEWS, RN50_CHANNELS)[chair]),
                             logits=values[chair:chair + 1], predictions=values[chair:chair + 1].argmax(1),
                             sample_ids=ids[count + chair:count + chair + 1], labels=labels[count + chair:count + chair + 1])
                count, step = end, step + 1
                if step == 1 or step % 25 == 0 or count == len(ids):
                    print('Scored {}/{} in {:.1f}s'.format(count, len(ids), perf_counter() - clock), flush=True)

    if count != len(ids):
        raise ValueError('Test sample coverage differs from the ModelNet40 index')
    logits = np.concatenate(scores)
    predictions = logits.argmax(axis=1)
    np.savez(out / 'predictions.npz', **common, logits=logits, predictions=predictions, labels=labels,
             sample_ids=ids, classnames=np.array(classes))
    correct = int((predictions == labels).sum())
    print('Table 1 PyTorch top-1: {}/{} = {:.2f}%. {}'.format(correct, count, 100 * correct / count, out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--batch-size', type=int, default=8, help='Number of point clouds per batch')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/pytorch/zero_table1_reference'))
    parser.add_argument('--chair', action='store_true', help='Save the chair_0978 intermediate embeddings')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    import torch
    with torch.no_grad():
        run(args)


if __name__ == '__main__':
    main()
