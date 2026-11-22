"""Shared ModelNet40 feature-cache, learning-rate and evaluation protocols."""
import csv
import json
import math

import numpy as np

from scripts.modelnet40 import CHAIR


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


def finite(values, shape, description):
    values = np.asarray(values)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"Invalid {description}: shape {values.shape}, expected {shape} and finite values")
    return values


def cosines(left, right):
    a, b = left.astype(np.float64), right.astype(np.float64)
    lengths = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    if not np.all(lengths > 0):
        raise ValueError("Cannot compare a zero-length embedding or score vector")
    return np.clip(np.sum(a * b, axis=-1) / lengths, -1, 1)


def same_metadata(cache, reference):
    for key, shape in (("classnames", (40,)), ("tokens", (40, 77))):
        if cache[key].shape != shape:
            raise ValueError(f"Unexpected {key} shape {cache[key].shape}")
        np.testing.assert_array_equal(cache[key], reference[key])
    if len(np.unique(cache["classnames"])) != 40:
        raise ValueError("ModelNet40 class names must be unique")


def load_cache(paths, reference, stage):
    chunks = []
    for path in paths:
        with np.load(path, allow_pickle=False) as cache:
            same_metadata(cache, reference)
            if cache["split"].item() != stage or int(cache["augmentation_seed"]) != -1:
                raise ValueError(f"{path}: expected unaugmented {stage} inputs")
            ids, labels = cache["sample_ids"], cache["labels"]
            start, stop = int(cache["start"]), int(cache["stop"])
            if stop - start != len(ids) or len(labels) != len(ids) or len(ids) == 0:
                raise ValueError(f"{path}: inconsistent chunk length")
            raw = finite(cache["features"], (len(ids), 10, 512), "view features")
            text = finite(cache["text"], (40, 512), "text features")
            scale = float(cache["logit_scale"].item())
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Expected a positive finite CLIP logit scale")
            chunks.append((start, stop, raw, ids, labels, text, scale))
    chunks.sort(key=lambda row: row[0])
    end = 0
    text, scale = chunks[0][5:]
    for start, stop, raw, ids, labels, other_text, other_scale in chunks:
        if start != end:
            raise ValueError(f"Non-contiguous test chunks at {end}")
        if not np.all(cosines(text, other_text) > 0.9999):
            raise ValueError("Text embeddings disagree across evaluated caches")
        np.testing.assert_allclose(other_scale, scale)
        end = stop
    if end != (1 if stage == "chair" else 2468):
        raise ValueError("Incomplete split coverage")
    views = np.concatenate([chunk[2] for chunk in chunks])
    ids = np.concatenate([chunk[3] for chunk in chunks]).astype(str)
    labels = np.concatenate([chunk[4] for chunk in chunks]).astype(np.int64)
    if len(set(ids)) != len(ids) or not np.all((0 <= labels) & (labels < 40)):
        raise ValueError("Invalid object IDs or class indices")
    if stage == "test":
        np.testing.assert_array_equal(ids, reference["sample_ids"])
        np.testing.assert_array_equal(labels, reference["labels"].ravel())
    else:
        if ids.tolist() != [CHAIR] or reference["sample_id"].item() != CHAIR:
            raise ValueError("Reference chair ID mismatch")
        np.testing.assert_array_equal(labels, [reference["label"].item()])
    return views, ids, labels, text, scale


def rank(scores):
    return np.argsort(-scores.astype(np.float32), axis=-1, kind="stable")


def performance(scores, labels):
    ordered = rank(scores)
    top1 = int(np.sum(ordered[:, 0] == labels))
    top5 = int(np.sum(np.any(ordered[:, :5] == labels[:, None], axis=1)))
    return dict(correct=top1, accuracy_percent=100 * top1 / len(labels), top5_correct=top5,
                top5_percent=100 * top5 / len(labels))


def write_csv(path, columns, rows):
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(columns)
        writer.writerows(rows)
