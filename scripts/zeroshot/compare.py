"""Audit two saved zero-shot evaluations without using either model during the comparison."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from scripts import ROOT


def cosines(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Jittor result directory")
    parser.add_argument("--reference", type=Path, help="optional override for matching PyTorch run")
    parser.add_argument("--output", type=Path, help="new comparison directory (defaults to a timestamp under the Jittor run)")
    args = parser.parse_args()
    with np.load(args.run / "predictions.npz") as result:
        protocol = str(result["protocol"].item()) if "protocol" in result else "repo"
    if args.reference is None:
        directory = {"repo": "zero_reference", "table1": "zero_table1_reference"}[protocol]
        args.reference = ROOT / "runs/pytorch" / directory
    with np.load(args.run / "predictions.npz") as result, np.load(args.reference / "predictions.npz") as ref:
        reference_protocol = str(ref["protocol"].item()) if "protocol" in ref else "repo"
        if protocol != reference_protocol:
            raise ValueError("The two runs use different projection or weighting protocols")
        classes = ref["classnames"]
        ids, labels, jt_logits = result["sample_ids"], result["labels"], result["logits"].astype(np.float32)
        n = len(ids)
        for name in ("classnames", "tokens"):
            np.testing.assert_array_equal(result[name], ref[name])
        for name in ("sample_ids", "labels"):
            np.testing.assert_array_equal(result[name], ref[name][:n])
        ref_logits = ref["logits"][:n].astype(np.float32)
        ours, theirs = jt_logits.argmax(1), ref_logits.argmax(1)
        np.testing.assert_array_equal(ours, result["predictions"])
        np.testing.assert_array_equal(theirs, ref["predictions"][:n])
        if jt_logits.shape != (n, 40) or ref_logits.shape != jt_logits.shape:
            raise ValueError("Expected 40 scores per object")
        mismatches = np.flatnonzero(ours != theirs)
        both = int(np.sum((ours == labels) & (theirs == labels)))
        jc, pc = int(np.sum(ours == labels)), int(np.sum(theirs == labels))
        report = dict(samples=n, matching_top1=int(n-len(mismatches)), matching_percent=100*(n-len(mismatches))/n,
                      jittor_correct=jc, pytorch_correct=pc, both_correct=both,
                      only_jittor_correct=jc-both, only_pytorch_correct=pc-both,
                      mean_abs_logit_error=float(np.abs(jt_logits-ref_logits).mean()),
                      max_abs_logit_error=float(np.abs(jt_logits-ref_logits).max()),
                      mean_score_cosine=float(cosines(jt_logits, ref_logits).mean()))
    output = args.output or args.run / "comparisons" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "disagreements.csv").open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["object","label","jittor","pytorch","pytorch_top1_margin","jittor_top1_margin"])
        for i in mismatches:
            pm, jm = np.sort(ref_logits[i]), np.sort(jt_logits[i])
            w.writerow([ids[i],classes[labels[i]],classes[ours[i]],classes[theirs[i]],pm[-1]-pm[-2],jm[-1]-jm[-2]])
    with (output / "per_class.csv").open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["class","test_objects","jittor_correct","pytorch_correct","same_prediction"])
        for i, name in enumerate(classes):
            selected = labels == i
            w.writerow([name,int(selected.sum()),int((ours[selected]==i).sum()),int((theirs[selected]==i).sum()),
                        int((ours[selected]==theirs[selected]).sum())])
    chair = np.flatnonzero(ids == "chair/chair_0978.ply")
    if len(chair) and (args.run / "chair.npz").is_file() and (args.reference / "chair.npz").is_file():
        i = chair.item()
        with np.load(args.run / "chair.npz") as a, np.load(args.reference / "chair.npz") as b:
            report["chair"] = dict(truth=str(classes[labels[i]]), jittor_top1=str(classes[ours[i]]),
                                   pytorch_top1=str(classes[theirs[i]]),
                                   minimum_view_cosine=float(cosines(a["features"], b["view_features_normalized"]).min()),
                                   minimum_prompt_cosine=float(cosines(a["text_raw"], b["text_features"]).min()))
    (output / "agreement.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Comparison:", output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
