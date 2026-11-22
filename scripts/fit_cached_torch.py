"""Independently train the upstream PyTorch PointCLIP adapter on shared RN101 features.

Run from the server project root, with this script alongside fit_fewshot.py:
    .venv/bin/python scripts/fit_cached_torch.py --shots 16 --output RUN/16shot
Only the adapter is trained. The held-out test cache is first opened after saving
the final checkpoint. Old runs are never overwritten.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from fit_fewshot import compare_text, learning_rate, ordered_epoch_caches, support_rows, training_data
from eval_fewshot import load_cache, performance, rank, write_csv


ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 32
EPOCHS = 250
SEED = 2


def upstream_model():
    upstream = ROOT / "reference/upstream/PointCLIP"
    sys.path[:0] = [str(upstream), str(upstream / "Dassl3D")]
    from dassl.config import get_cfg_default
    from trainers.fewshot import Adapter, smooth_loss

    cfg = get_cfg_default()
    cfg.merge_from_file(str(upstream / "configs/datasets/modelnet40.yaml"))
    cfg.merge_from_file(str(upstream / "configs/trainers/PointCLIP_FS/rn101.yaml"))
    cfg.freeze()
    adapter = Adapter(cfg).float()
    if (adapter.num_views, adapter.in_features) != (10, 512):
        raise ValueError("Unexpected upstream view or channel configuration")
    np.testing.assert_allclose([adapter.adapter_ratio, adapter.fusion_init, adapter.dropout],
                               [0.6, 0.5, 0.075])
    return adapter.cuda(), smooth_loss


def normalized_rows(x):
    return x / x.norm(dim=-1, keepdim=True)


def text_weights(text):
    return normalized_rows(torch.from_numpy(text).cuda().float().repeat(1, 10))


def classify(views, weights, scale, model):
    image = views.flatten(1) if model is None else model(views)
    return (normalized_rows(image) * scale) @ weights.t()


def json_file(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(array):
    return hashlib.sha256("\n".join(array).encode("utf-8")).hexdigest()


def train(args, model, criterion):
    output = args.output / "training"
    output.mkdir()
    manifest = ROOT / "manifests/modelnet40_16shot_seed2.json"
    cache_files = list((args.cache / "augmentations").glob("train_aug_*.npz"))
    caches = ordered_epoch_caches(cache_files, EPOCHS)
    seeds = [seed for seed, path in caches]
    if seeds != list(range(1000, 1250)):
        raise ValueError("Training requires all 250 archived augmentation seeds 1000..1249")
    rows, pool_seed = support_rows(caches[0][1], manifest, args.shots)
    first_views, labels, ids, text, scale = training_data([caches[0][1]], rows)
    weights = text_weights(text)
    targets = torch.as_tensor(labels, device="cuda", dtype=torch.long)
    n = len(ids)
    np.testing.assert_array_equal(np.bincount(labels, minlength=40), np.full(40, args.shots))
    write_csv(output / "training_objects.csv", ["sample_id", "label"], zip(ids, labels))

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
    # This stream is reserved for batch shuffling. Dropout uses PyTorch's CUDA RNG.
    rng = np.random.RandomState(SEED)
    order_hash = hashlib.sha256()
    started = perf_counter()
    with (output / "history.csv").open("w", newline="") as stream:
        history = csv.writer(stream)
        history.writerow(("epoch", "lr", "training_loss", "training_top1_percent", "elapsed_seconds",
                          "augmentation_seed"))
        for epoch, (augmentation_seed, path) in enumerate(caches, 1):
            if epoch == 1:
                views = first_views
            else:
                views, current_labels, current_ids, current_text, current_scale = training_data([path], rows)
                np.testing.assert_array_equal(current_labels, labels)
                np.testing.assert_array_equal(current_ids, ids)
                np.testing.assert_allclose(current_scale, scale)
                compare_text(text, current_text)
            if views.shape != (1, n, 10, 512):
                raise ValueError("Expected one independently augmented view tensor per epoch")
            images = torch.as_tensor(views[0], device="cuda", dtype=torch.float32)
            lr = learning_rate(epoch, EPOCHS)
            for group in optimizer.param_groups:
                group["lr"] = lr
            order = rng.permutation(n)
            order_hash.update(order.astype("<i8").tobytes())
            permutation = torch.as_tensor(order, device="cuda", dtype=torch.long)
            total_loss, correct = 0.0, 0
            for idx in permutation.split(BATCH_SIZE):
                expected = targets[idx]
                logits = classify(images[idx], weights, scale, model)
                loss = criterion(logits, expected)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * idx.numel()
                correct += int((logits.detach().argmax(1) == expected).sum().item())
            mean_loss = total_loss / n
            if not math.isfinite(mean_loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            history.writerow((epoch, lr, mean_loss, 100 * correct / n,
                              round(perf_counter() - started, 2), augmentation_seed))
            stream.flush()
            if epoch in (1, 2) or epoch % 10 == 0:
                print(f"Epoch {epoch}/{EPOCHS} loss={mean_loss:.6f} train_top1={100*correct/n:.2f}%", flush=True)

    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise FloatingPointError("Non-finite final checkpoint")
    partial = output / "adapter.partial.pt"
    torch.save(state, str(partial))
    partial.replace(output / "adapter.pt")
    json_file(output / "training.json", dict(framework="PyTorch", version=torch.__version__, seed=SEED,
              training_objects=n, shots_per_class=args.shots, cached_variants=len(caches),
              optimizer_steps=EPOCHS * math.ceil(n / BATCH_SIZE), epochs=EPOCHS, final_epoch=EPOCHS,
              batch_size=BATCH_SIZE, support_pool_seed=pool_seed, support_manifest=str(manifest),
              selected_ids_sha256=digest(ids), epoch_batch_order_sha256=order_hash.hexdigest(),
              initialization="native PyTorch author Adapter, seeded CPU initialization",
              dropout_prob=0.075, dropout_rng="PyTorch CUDA", dtype="float32", matmul_allow_tf32=True,
              optimizer="SGD", momentum=0.9, weight_decay=5e-4,
              loss="upstream smooth_loss: 0.8 target, 0.2/39 for each non-target class",
              selection="fixed final epoch; no validation or test inputs",
              augmentation_mode="epoch", epoch_augmentation_seeds=seeds,
              feature_caches=[str(path) for _, path in caches],
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
              gpu=torch.cuda.get_device_name(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
              elapsed_seconds=round(perf_counter() - started, 2)))
    return set(ids)


@torch.no_grad()
def evaluate(args, model, support):
    path = args.cache / "test.npz"
    reference_path = ROOT / "runs/pytorch/fewshot_reference/predictions.npz"
    with np.load(reference_path, allow_pickle=False) as reference:
        raw, ids, labels, text, scale = load_cache([path], reference, "test")
        classes, tokens = reference["classnames"], reference["tokens"]
    if support.intersection(ids):
        raise ValueError("Support and evaluation IDs overlap")
    features = torch.as_tensor(raw, dtype=torch.float32, device="cuda")
    weights = text_weights(text)
    published = ROOT / "weights/adapter/model-best.pth.tar"
    trained = args.output / "training/adapter.pt"
    model.eval()
    scores = {}
    for name in ("no_adapter", "author", "trained"):
        if name == "author":
            model.load_state_dict(torch.load(str(published), map_location="cpu")["state_dict"], strict=True)
        if name == "trained":
            model.load_state_dict(torch.load(str(trained), map_location="cpu"), strict=True)
        head = None if name == "no_adapter" else model
        scores[name] = torch.cat([classify(batch, weights, scale, head)
                                  for batch in features.split(BATCH_SIZE)]).cpu().numpy()
        if scores[name].shape != (2468, 40) or not np.isfinite(scores[name]).all():
            raise ValueError("Non-finite or incomplete test output")
    output = args.output / "eval"
    output.mkdir()
    heads = {name: performance(logits, labels) for name, logits in scores.items()}
    pred = {name: rank(logits)[:, 0] for name, logits in scores.items()}
    np.savez_compressed(output / "predictions.npz", sample_ids=ids, labels=labels,
                        classnames=classes, tokens=tokens,
                        **{name + "_logits": logits for name, logits in scores.items()},
                        **{name + "_predictions": p for name, p in pred.items()})
    write_csv(output / "predictions.csv", ["sample_id", "label", *scores],
              zip(ids, classes[labels], *[classes[p] for p in pred.values()]))
    per_class = []
    for i, c in enumerate(classes):
        mask = labels == i
        per_class.append([c, int(mask.sum()), *[int((mask & (p == i)).sum()) for p in pred.values()]])
    write_csv(output / "per_class.csv", ["class", "objects", *[name + "_correct" for name in scores]], per_class)
    chair_row = np.flatnonzero(ids == "chair/chair_0978.ply").item()
    write_csv(output / "chair_scores.csv", ["class", *scores],
              zip(classes, *[logits[chair_row] for logits in scores.values()]))
    report = dict(framework="PyTorch", version=torch.__version__, model="PointCLIP V1 RN101",
                  test_objects=len(ids), views=10, post_search=False, heads=heads,
                  frozen_test_cache=str(path), training_test_overlap=0)
    json_file(output / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--cache", type=Path, default=Path(".cache/fewshot-rn101"))
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.output.exists():
        parser.error("Please choose a new output path")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, criterion = upstream_model()
    args.output.mkdir(parents=True)
    support = train(args, model, criterion)
    evaluate(args, model, support)


if __name__ == "__main__":
    main()
