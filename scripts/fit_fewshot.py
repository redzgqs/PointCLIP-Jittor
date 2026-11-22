"""Train a Jittor PointCLIP V1 adapter on nested ModelNet40 support sets."""
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


ROOT = Path(__file__).resolve().parents[1]


def compare_text(left, right):
    # Independent GPU runs can round nearly zero components differently.
    right = right.astype(np.float64)
    denom = np.linalg.norm(right, axis=1) * np.linalg.norm(left, axis=1)
    cosine = (right * left).sum(axis=1) / denom
    if not np.all(cosine > 0.9999):
        raise ValueError("Prompt embeddings disagree between training caches")


def ordered_epoch_caches(paths, epochs):
    """Read only small metadata now; feature matrices stream in one per epoch."""
    if len(paths) != epochs:
        raise ValueError("Epoch augmentation mode requires one feature file per training epoch")
    indexed = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            seed = int(data["augmentation_seed"].item())
            if seed < 0 or data["split"].item() != "train":
                raise ValueError(f"{path}: each epoch requires one augmented training split")
            if not indexed:
                tokens, classes = data["tokens"], data["classnames"]
                ids, labels = data["sample_ids"], data["labels"]
            else:
                for key, expected in (("sample_ids", ids), ("labels", labels),
                                      ("tokens", tokens), ("classnames", classes)):
                    np.testing.assert_array_equal(data[key], expected)
            indexed.append((seed, path))
    if len({seed for seed, path in indexed}) != epochs:
        raise ValueError("Each epoch must have a unique augmentation seed")
    # Use increasing augmentation seed, including when a shell expands 1, 10, ..., 2.
    return sorted(indexed, key=lambda item: item[0])


def support_rows(path, manifest_path, shots):
    """Index the first k rows of each class in the fixed 16-shot reference order."""
    manifest = json.loads(manifest_path.read_text())
    expected_ids = manifest["sample_ids"]
    classes = manifest["classes"]
    if len(expected_ids) != 640 or len(set(expected_ids)) != 640 or len(set(classes)) != 40:
        raise ValueError("Reference manifest must contain 640 distinct objects for 40 classes")
    with np.load(path, allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache["sample_ids"].reshape(-1), expected_ids)
        np.testing.assert_array_equal(cache["classnames"], classes)
        labels = cache["labels"].astype(np.int64).reshape(-1)
    np.testing.assert_array_equal(labels, [classes.index(name.split("/")[0]) for name in expected_ids])
    np.testing.assert_array_equal(np.bincount(labels, minlength=40), np.full(40, 16))
    selected = np.sort(np.concatenate([np.flatnonzero(labels == c)[:shots] for c in range(40)]))
    return selected, manifest["seed"]


def training_data(paths, selected=None):
    """Cache shape is [variant, object, view, channel], with identities shared across variants."""
    variants = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "split" in data and data["split"].item() != "train":
                raise ValueError(f"{path}: training input must be from ModelNet40 train")
            features = data["features"]
            if features.ndim == 3:
                features = features[None]
            if features.ndim != 4 or features.shape[1:] != (640, 10, 512):
                raise ValueError(f"{path}: expected one or more cached variants of [640,10,512]")
            if not np.isfinite(features).all():
                raise ValueError(f"{path}: non-finite training features")
            if not variants:
                labels = data["labels"].astype(np.int32).reshape(-1)
                ids = data["sample_ids"].astype(str).reshape(-1)
                text = data["text"].astype(np.float32)
                scale = float(data["logit_scale"].item())
                tokens = data["tokens"] if "tokens" in data else None
                classes = data["classnames"] if "classnames" in data else None
            else:
                np.testing.assert_array_equal(data["labels"].reshape(-1), labels)
                np.testing.assert_array_equal(data["sample_ids"].astype(str).reshape(-1), ids)
                if tokens is not None:
                    np.testing.assert_array_equal(data["tokens"], tokens)
                if classes is not None:
                    np.testing.assert_array_equal(data["classnames"], classes)
                compare_text(text, data["text"])
                np.testing.assert_allclose(data["logit_scale"].item(), scale)
            variants.append(features.astype(np.float32, copy=False))
    if labels.shape != (640,) or len(set(ids)) != 640 or len(ids) != 640:
        raise ValueError("Expected 640 distinct labeled training objects")
    np.testing.assert_array_equal(np.bincount(labels, minlength=40), np.full(40, 16))
    if text.shape != (40, 512) or not np.isfinite(text).all() or not (0 < scale < np.inf):
        raise ValueError("Expected finite 40-class RN101 prompt embeddings and positive CLIP scale")
    views = variants[0] if len(variants) == 1 else np.concatenate(variants)
    if selected is not None and len(selected) < len(ids):
        views, labels, ids = views[:, selected], labels[selected], ids[selected]
    return views, labels, ids, text, scale


def learning_rate(epoch, total_epochs):
    # Dassl constant warm-up for epoch 1, then cosine annealing with WARMUP_RECOUNT.
    if epoch == 1:
        return 1e-5
    return 0.01 * (1 + math.cos(math.pi * (epoch - 2) / total_epochs)) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", type=Path, required=True,
                        help="train feature NPZ(s), with the same 640 identities in the same order")
    parser.add_argument("--output", type=Path, required=True, help="new directory for this run")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--shots", type=int, default=16, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--support-manifest", type=Path,
                        help="ordered 16-shot source manifest; required for nested support sets")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--augmentation-mode", choices=("sample", "epoch"), default="sample",
                        help="sample a variant per object, or use a fresh cache in increasing seed order each epoch")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    for key in ("use_mkl", "use_cutt", "use_cutlass", "use_nccl"):
        os.environ.setdefault(key, "0")
    os.environ.setdefault("conv_opt", "1")
    os.environ.setdefault("cache_name", "pointclip_projection")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import jittor as jt
    from jittor import nn
    sys.path.insert(0, str(ROOT / "src"))
    from pointclip_jittor.adapter import InterViewAdapter, class_weights, score_views

    jt.flags.use_cuda = 1
    jt.flags.use_tensorcore = 1
    jt.set_global_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    epoch_caches = ordered_epoch_caches(args.train, args.epochs) if args.augmentation_mode == "epoch" else []
    inputs = [epoch_caches[0][1]] if epoch_caches else args.train
    manifest = args.support_manifest
    if args.shots < 16 and manifest is None:
        manifest = ROOT / "manifests/modelnet40_16shot_seed2.json"
    selected, support_seed = support_rows(inputs[0], manifest, args.shots) if manifest else (None, None)
    views, labels, ids, text, scale = training_data(inputs, selected)
    np.testing.assert_array_equal(np.bincount(labels, minlength=40), np.full(40, args.shots))
    if epoch_caches and len(views) != 1:
        raise ValueError("Each epoch cache must contain exactly one variant per object")
    variants = len(epoch_caches) if epoch_caches else len(views)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "training_objects.csv").open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(("sample_id", "label"))
        writer.writerows(zip(ids, labels))
    model = InterViewAdapter()
    model.train()
    text_weights = class_weights(jt.array(text)).stop_grad().sync()
    lr = jt.array([1e-5]).float32().stop_grad()
    params = [p for p in model.parameters() if not p.is_stop_grad()]
    optimizer = jt.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    batch_size, n = 32, len(labels)
    started = perf_counter()
    print(f"Training {args.epochs} epochs on {n} objects with {variants} cached variants "
          f"(mode={args.augmentation_mode})", flush=True)
    with (args.output / "history.csv").open("w", newline="") as fp:
        history = csv.writer(fp)
        history.writerow(("epoch", "lr", "training_loss", "training_top1_percent", "elapsed_seconds",
                          "augmentation_seed"))
        for epoch in range(1, args.epochs + 1):
            if epoch_caches and epoch > 1:
                del views
                views, current_labels, current_ids, current_text, current_scale = training_data(
                    [epoch_caches[epoch - 1][1]], selected)
                if len(views) != 1:
                    raise ValueError("Each epoch cache must contain exactly one variant per object")
                np.testing.assert_array_equal(current_labels, labels)
                np.testing.assert_array_equal(current_ids, ids)
                np.testing.assert_allclose(current_scale, scale)
                compare_text(text, current_text)
            rate = learning_rate(epoch, args.epochs)
            lr.update(jt.array([rate]).float32())
            order = rng.permutation(n)
            variant = np.zeros(n, dtype=np.int32) if epoch_caches else rng.randint(len(views), size=n)
            total_loss = 0.0
            total_correct = 0
            for start in range(0, n, batch_size):
                batch = order[start:start + batch_size]
                image = jt.array(views[variant[batch], batch]).stop_grad()
                target = jt.array(labels[batch])
                logits = score_views(image, text_weights, scale, adapter=model)
                log_probs = nn.log_softmax(logits, dim=1)
                positive = log_probs.gather(1, target.reshape(-1, 1)).reshape(-1)
                # Target gets 0.8; each other class gets 0.2/39 (authors' smooth_loss).
                loss = -(0.8 * positive + (0.2 / 39) * (log_probs.sum(1) - positive)).mean()
                correct = (jt.argmax(logits, dim=1)[0] == target).sum()
                optimizer.step(loss)
                total_loss += loss.item() * len(batch)
                total_correct += int(correct.item())
            mean_loss = total_loss / n
            if not np.isfinite(mean_loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            elapsed = perf_counter() - started
            augmentation_seed = epoch_caches[epoch - 1][0] if epoch_caches else ""
            history.writerow((epoch, rate, mean_loss, 100 * total_correct / n, round(elapsed, 2),
                              augmentation_seed))
            fp.flush()
            if epoch in (1, 2, args.epochs) or epoch % 10 == 0:
                print(f"Epoch {epoch}/{args.epochs} loss={mean_loss:.4f} "
                      f"train_top1={100*total_correct/n:.2f}% time={elapsed:.1f}s", flush=True)
    model.eval()
    state = {name: value.numpy() for name, value in model.state_dict().items()}
    if not all(np.isfinite(value).all() for value in state.values()):
        raise FloatingPointError("Non-finite values in trained adapter")
    temporary = args.output / "adapter.partial.npz"
    np.savez(temporary, **state)
    temporary.replace(args.output / "adapter.npz")
    details = dict(framework="Jittor", version=jt.__version__, seed=args.seed,
                   training_objects=n, shots_per_class=args.shots, cached_variants=variants,
                   optimizer_steps=args.epochs * math.ceil(n / batch_size),
                   support_pool_seed=support_seed,
                   support_manifest=manifest.name if manifest is not None else None,
                   epochs=args.epochs, final_epoch=args.epochs, batch_size=batch_size,
                   initialization="random", optimizer="SGD", momentum=0.9, weight_decay=5e-4,
                   selection="fixed final epoch; no validation or test inputs",
                   augmentation_mode=args.augmentation_mode,
                   epoch_augmentation_seeds=[seed for seed, p in epoch_caches],
                   feature_caches=[str(p) for p in ([p for _, p in epoch_caches] if epoch_caches else args.train)],
                   selected_ids_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                   trainable_parameters=sum(int(p.numel()) for p in params),
                   elapsed_seconds=round(perf_counter() - started, 2))
    (args.output / "training.json").write_text(json.dumps(details, indent=2) + "\n")
    print(f"Saved final-epoch checkpoint to {args.output / 'adapter.npz'}", flush=True)


if __name__ == "__main__":
    main()
