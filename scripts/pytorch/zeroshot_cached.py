"""Recover author-run logits from its image cache and optionally rerun one chair end to end."""

import argparse
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np

from scripts import ROOT
from scripts.modelnet40 import test_index
from scripts.pytorch import RN50_CHANNELS, RN50_VIEWS, to_numpy


def cache_tensor(path):
    import torch

    value = torch.load(str(path), map_location='cpu')
    return torch.cat(value) if isinstance(value, (list, tuple)) else value


def log_count(path, name):
    pattern = r'^\s*\*?\s*' + name + r'\s*:\s*([\d,]+)\s*$'
    values = re.findall(pattern, path.read_text(), flags=re.MULTILINE | re.IGNORECASE)
    if not values:
        raise ValueError('Missing evaluation count {!r} in {}'.format(name, path))
    return int(values[-1].replace(',', ''))


def chair_reference(torch, trainer, text_encoder, clip_model, flat_text, meta, root, shards, out):
    import h5py
    from trainers.mv_utils_zs import PCViews

    row = 3
    with h5py.File(str(shards[0]), 'r') as h5:
        points = np.asarray(h5['data'][row, :1024], dtype=np.float32)
    if meta['sample_ids'][row] != 'chair/chair_0978.ply':
        raise ValueError('The first shard no longer contains the reference chair at row 3')

    # Call the actual upstream model_inference + mv_proj methods. Capture the values they use.
    captured = {}
    projector = PCViews()

    def get_img(pc):
        depth = projector.get_img(pc)
        captured['depth'] = to_numpy(depth)
        return depth

    def mv_proj(pc):
        rgb = trainer.mv_proj(context, pc)
        captured['resized_depth'] = to_numpy(rgb[:, 0])
        return rgb

    def capture_visual(module, inputs, output):
        captured['view_features'] = to_numpy(output)

    context = SimpleNamespace(get_img=get_img, mv_proj=mv_proj, visual_encoder=clip_model.visual,
                              textual_encoder=text_encoder, num_views=RN50_VIEWS, channel=RN50_CHANNELS,
                              dtype=clip_model.dtype, logit_scale=clip_model.logit_scale,
                              feat_store=[], label_store=[])
    hook = clip_model.visual.register_forward_hook(capture_visual)
    try:
        logits = trainer.model_inference(context, torch.from_numpy(points).unsqueeze(0).cuda())
    finally:
        hook.remove()
    normalized = context.feat_store[0]
    replay = (clip_model.logit_scale.exp() * normalized) @ flat_text.t()
    if not torch.equal(logits, replay):
        raise ValueError('Single-object replay differs from upstream model_inference')

    chair_file = root / 'runs/examples/chair_projection/reference.npz'
    with np.load(chair_file, allow_pickle=False) as previous:
        np.testing.assert_array_equal(previous['points'], points)
        np.testing.assert_allclose(previous['depth'], captured['depth'], atol=1e-6, rtol=0)
    cached = meta['chair_cache'].reshape(RN50_VIEWS, RN50_CHANNELS)
    fresh = to_numpy(normalized.reshape(RN50_VIEWS, RN50_CHANNELS))
    cosine = np.sum(fresh.astype(np.float32) * cached.astype(np.float32), axis=1)
    cosine /= np.linalg.norm(fresh.astype(np.float32), axis=1) * np.linalg.norm(cached.astype(np.float32), axis=1)
    if np.min(cosine) < 0.99:
        raise ValueError('Single-object visual features disagree with the cached author run: ' + str(cosine))
    ref_logits = to_numpy(logits)
    cache_logits = meta['logits'][row:row + 1]
    np.savez(out / 'chair.npz', points=points, **captured, view_features_normalized=fresh,
             cached_view_features_normalized=cached, cached_logits=cache_logits, logits=ref_logits,
             predictions=to_numpy(logits.argmax(1)), labels=meta['labels'][row:row + 1],
             sample_ids=meta['sample_ids'][row:row + 1], text_features=meta['text_features'],
             text_features_repeated=meta['text_features_repeated'], tokens=meta['tokens'],
             logit_scale=meta['logit_scale'])
    print('chair_0978 fresh / cached predictions: {} / {}, minimum view cosine {:.8f}, max logit delta {:.6f}'.format(
        int(ref_logits.argmax(1)[0]), int(cache_logits.argmax(1)[0]), float(cosine.min()),
        float(np.abs(ref_logits.astype(np.float32) - cache_logits.astype(np.float32)).max())))


def run(args):
    import torch

    upstream = args.root / 'reference/upstream/PointCLIP'
    sys.path[:0] = [str(upstream), str(upstream / 'Dassl3D')]
    from clip import clip
    from trainers.zeroshot import CUSTOM_TEMPLATES, PointCLIP_ZS, Textual_Encoder

    cache = args.root / args.cache
    out = args.root / args.output_dir
    classes, ids, h5_labels, shards = test_index(args.root / 'data/modelnet40_ply_hdf5_2048')
    labels = cache_tensor(cache / 'labels.pt').view(-1)
    np.testing.assert_array_equal(to_numpy(labels), h5_labels)
    image = cache_tensor(cache / 'features.pt')
    if image.shape != (len(ids), RN50_VIEWS * RN50_CHANNELS):
        raise ValueError('Unexpected zero-shot feature shape ' + str(tuple(image.shape)))

    state = torch.jit.load(str(args.root / 'weights/clip/RN50.pt'), map_location='cpu').state_dict()
    model = clip.build_model(state).cuda().eval()
    del state
    if image.dtype != model.dtype:
        raise ValueError('Cached image dtype differs from the original encoder output dtype')
    torch.backends.cudnn.benchmark = True  # Upstream train.py, including the evaluation path.
    cfg = SimpleNamespace(DATASET=SimpleNamespace(NAME='ModelNet40'),
                          MODEL=SimpleNamespace(PROJECT=SimpleNamespace(NUM_VIEWS=RN50_VIEWS)))
    text_encoder = Textual_Encoder(cfg, classes, model).eval()
    prompts = [CUSTOM_TEMPLATES['ModelNet40'].format(c.replace('_', ' ')) for c in classes]
    tokens = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    repeated = text_encoder()
    raw_text = repeated[:, :RN50_CHANNELS]
    flat_text = repeated / repeated.norm(dim=-1, keepdim=True)
    scale = model.logit_scale.exp()
    # Match the unweighted RN50 config's 100-object batches and original half-precision operation order.
    logits = torch.cat([((scale * batch.cuda()) @ flat_text.t()).cpu() for batch in image.split(100)])
    preds = logits.argmax(dim=1)
    correct = int((preds == labels).sum().item())
    log = cache / 'log.txt'
    if (len(ids), correct) != (log_count(log, 'total'), log_count(log, 'correct')):
        raise ValueError('Cached replay {}/{} does not reproduce {}'.format(correct, len(ids), log))
    out.mkdir(parents=True, exist_ok=False)
    result = dict(logits=to_numpy(logits), predictions=to_numpy(preds), labels=to_numpy(labels), sample_ids=ids,
                  classnames=np.array(classes), tokens=to_numpy(tokens), text_features=to_numpy(raw_text),
                  text_features_repeated=to_numpy(flat_text), logit_scale=to_numpy(scale))
    np.savez(out / 'predictions.npz', **result)
    print('Matched original evaluation: {}/{} ({:.2f}%). {}'.format(correct, len(ids), 100 * correct / len(ids), out))
    if args.chair:
        result['chair_cache'] = to_numpy(image[3])
        chair_reference(torch, PointCLIP_ZS, text_encoder, model, flat_text, result, args.root, shards, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='PointCLIP project root')
    parser.add_argument('--cache', type=Path, required=True, help='Existing author-run features.pt, labels.pt and log.txt directory')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/pytorch/zero_cache_replay'))
    parser.add_argument('--chair', action='store_true', help='Also run the upstream projector, image and text encoder for chair_0978')
    args = parser.parse_args()
    import torch
    with torch.no_grad():
        run(args)


if __name__ == '__main__':
    main()
