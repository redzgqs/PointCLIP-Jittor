"""Audit the reference chair and compare three RN101 heads on the full test split."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

from scripts import ROOT
from scripts.modelnet40 import CHAIR
from scripts.fewshot.protocol import cosines, finite, load_cache, performance, rank, write_csv


def score_heads(views, text, scale, checkpoints):
    import jittor as jt
    from pointclip_jittor.models.adapter import InterViewAdapter, class_weights, load_adapter_npz, score_views

    scores = {}
    with jt.no_grad():
        prompts = class_weights(jt.array(text.astype(np.float32))).sync()
        for name, path in checkpoints.items():
            adapter = None
            if path is not None:
                adapter = load_adapter_npz(InterViewAdapter(), path)
                adapter.eval()
            rows = []
            for start in range(0, len(views), 32):
                features = jt.array(views[start:start + 32].astype(np.float32, copy=False))
                rows.append(score_views(features, prompts, scale, adapter).numpy().copy())
            scores[name] = finite(np.concatenate(rows), (len(views), 40), f"{name} logits")
    return scores


def audit_chair(output, path, reference, views, text, labels, classes, scores):
    with np.load(path, allow_pickle=False) as cache:
        np.testing.assert_array_equal(finite(cache["points"], (1024, 3), "chair points"), reference["points"])
        depth = finite(cache["depths"], (10, 224, 224), "Jittor depths")
    original = finite(reference["depth"], (10, 224, 224), "PyTorch depths")
    pixel_difference = np.abs(depth.astype(np.float64) - original)
    occupied = depth != 0
    reference_mask = original != 0
    per_view_masks = np.sum(occupied != reference_mask, axis=(1, 2))
    expected_features = finite(reference["view_features"], (10, 512), "PyTorch visual embeddings")
    view_similarity = cosines(views[0], expected_features)
    prompt_similarity = cosines(text, finite(reference["text_features"], (40, 512), "PyTorch text embeddings"))
    rows = zip(range(1, 11), occupied.sum(axis=(1, 2)), reference_mask.sum(axis=(1, 2)),
               per_view_masks, pixel_difference.max(axis=(1, 2)), pixel_difference.mean(axis=(1, 2)),
               view_similarity)
    write_csv(output / "per_view.csv", ("view","jittor_occupied","pytorch_occupied","mask_differences",
                                        "max_abs_depth_error","mean_abs_depth_error","feature_cosine"), rows)
    report = dict(sample_id=CHAIR, true_class=str(classes[labels[0]]), points_identical=True,
                  views=10, resolution=224, input_points=1024, mask_difference_pixels=int(per_view_masks.sum()),
                  max_abs_depth_error=float(pixel_difference.max()), min_view_cosine=float(view_similarity.min()),
                  mean_view_cosine=float(view_similarity.mean()), minimum_prompt_cosine=float(prompt_similarity.min()),
                  head={})
    columns = ["class"]
    values = [classes]
    for name, key in (("author", "logits"), ("no_adapter", "control_logits")):
        our = scores[name][0]
        ref = finite(reference[key].ravel(), (40,), f"PyTorch {name} scores")
        jtop, ptop = rank(our)[:5], rank(ref)[:5]
        report["head"][name] = dict(jittor_top5=classes[jtop].tolist(), pytorch_top5=classes[ptop].tolist(),
                                    matching_top1=bool(jtop[0] == ptop[0]), score_cosine=float(cosines(our, ref)),
                                    max_abs_logit_error=float(np.abs(our.astype(np.float64) - ref).max()))
        columns.extend((f"jittor_{name}", f"pytorch_{name}"))
        values.extend((our, ref))
    write_csv(output / "scores.csv", columns, zip(*values))
    return report


def audit_test(output, reference, ids, labels, classes, scores):
    names = list(scores)
    predicted = {name: rank(scores[name])[:, 0] for name in names}
    report = dict(framework="Jittor", model="PointCLIP V1 RN101", test_objects=len(ids), views=10,
                  post_search=False, heads={name: performance(scores[name], labels) for name in names})
    ref_predictions = {}
    for name, key, pred_key in (("author", "logits", "predictions"),
                                ("no_adapter", "control_logits", "control_predictions")):
        ref = finite(reference[key], (len(ids), 40), "PyTorch " + name)
        ref_pred = rank(ref)[:, 0]
        np.testing.assert_array_equal(ref_pred, reference[pred_key])
        ref_predictions[name] = ref_pred
        same = int(np.sum(predicted[name] == ref_pred))
        delta = np.abs(scores[name].astype(np.float64) - ref)
        report["heads"][name]["pytorch"] = performance(ref, labels)
        report["heads"][name]["agreement"] = dict(matching_top1=same, matching_percent=100 * same / len(ids),
                                                  mean_score_cosine=float(cosines(scores[name], ref).mean()),
                                                  max_abs_logit_error=float(delta.max()),
                                                  mean_abs_logit_error=float(delta.mean()))
    base = report["heads"]["no_adapter"]["accuracy_percent"]
    for name in names:
        report["heads"][name]["gain_vs_no_adapter_pp"] = report["heads"][name]["accuracy_percent"] - base
    sample_columns = ["sample_id", "label"] + names + ["pytorch_author", "pytorch_no_adapter"]
    values = [ids, classes[labels], *[classes[predicted[n]] for n in names],
              classes[ref_predictions["author"]], classes[ref_predictions["no_adapter"]]]
    write_csv(output / "predictions.csv", sample_columns, zip(*values))
    chair_row = np.flatnonzero(ids == CHAIR).item()
    write_csv(output / "chair_scores.csv", ["class"] + names,
              zip(classes, *[scores[name][chair_row] for name in names]))
    class_columns = ["class", "objects"] + [n + "_correct" for n in names]
    class_rows = []
    for idx, name in enumerate(classes):
        mask = labels == idx
        class_rows.append([name, int(mask.sum()), *[int(np.sum(mask & (predicted[n] == idx))) for n in names]])
    write_csv(output / "per_class.csv", class_columns, class_rows)
    np.savez_compressed(output / "predictions.npz", sample_ids=ids, labels=labels, classnames=classes,
                        tokens=reference["tokens"], **{n + "_logits": scores[n] for n in names},
                        **{n + "_predictions": predicted[n] for n in names})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("chair", "test"))
    parser.add_argument("--features", type=Path, nargs="+", action="extend", required=True)
    parser.add_argument("--reference", type=Path, required=True, help="matching upstream PyTorch NPZ")
    parser.add_argument("--author", type=Path, required=True, help="converted published adapter NPZ")
    parser.add_argument("--trained", type=Path, help="independently trained Jittor adapter NPZ for test")
    parser.add_argument("--output", type=Path, required=True, help="new result directory")
    args = parser.parse_args()
    if args.stage == "chair" and (len(args.features) != 1 or args.trained):
        parser.error("Chair comparison expects a single feature file and the published weights")
    if args.output.exists():
        parser.error("Choose a new output directory")
    for key in ("use_mkl", "use_cutt", "use_cutlass", "use_nccl"):
        os.environ.setdefault(key, "0")
    os.environ.setdefault("conv_opt", "1")
    os.environ.setdefault("cache_name", "pointclip_projection")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import jittor as jt
    sys.path.insert(0, str(ROOT / "src"))
    jt.flags.use_cuda = 1
    jt.flags.use_tensorcore = 1
    checkpoints = {"no_adapter": None, "author": args.author}
    if args.trained:
        checkpoints["trained"] = args.trained
    with np.load(args.reference, allow_pickle=False) as reference:
        views, ids, labels, text, scale = load_cache(args.features, reference, args.stage)
        classes = reference["classnames"].astype(str)
        scores = score_heads(views, text, scale, checkpoints)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=str(args.output.parent)) as temp:
            folder = Path(temp) / "results"
            folder.mkdir()
            if args.stage == "chair":
                report = audit_chair(folder, args.features[0], reference, views, text, labels, classes, scores)
            else:
                report = audit_test(folder, reference, ids, labels, classes, scores)
            (folder / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            if args.output.exists():
                raise FileExistsError(args.output)
            folder.rename(args.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
