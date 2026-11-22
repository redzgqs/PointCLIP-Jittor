"""Recalculate the five shared-feature experiments from their published test rankings."""
import csv
from collections import defaultdict
from math import isclose
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "results"


def read_rows(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    predictions = defaultdict(list)
    for row in read_rows("predictions.csv"):
        predictions[int(row["shot"])].append(row)
    summary = {int(row["shots"]): row for row in read_rows("shot_comparison.csv")}
    shots = (1, 2, 4, 8, 16)
    if set(summary) != set(shots) or set(predictions) != set(shots):
        raise ValueError("Expected the five ModelNet40 experiments")
    reference_ids = None
    print("shot    objects    Jittor correct    PyTorch correct    same top-1")
    for k in shots:
        rows, expected = predictions[k], summary[k]
        ids = {r["sample_id"]: r["truth"] for r in rows}
        if len(rows) != 2468 or len(ids) != len(rows) or (reference_ids is not None and ids != reference_ids):
            raise ValueError(f"Inconsistent test set at {k}-shot")
        reference_ids = ids
        hits = {}
        for framework in ("jittor", "pytorch"):
            first = sum(r[f"{framework}_rank1"] == r["truth"] for r in rows)
            five = sum(r["truth"] in [r[f"{framework}_rank{i}"] for i in range(1, 6)] for r in rows)
            if (first != int(expected[f"{framework}_correct"]) or
                    not isclose(first * 100 / len(rows), float(expected[f"{framework}_top1"])) or
                    not isclose(five * 100 / len(rows), float(expected[f"{framework}_top5"]))):
                raise ValueError(f"{framework} accuracy mismatch at {k}-shot")
            hits[framework] = first
        same = sum(r["jittor_rank1"] == r["pytorch_rank1"] for r in rows)
        if same != int(expected["same_top1"]):
            raise ValueError(f"Top-1 agreement mismatch at {k}-shot")
        print(f"{k:4} {len(rows):10} {hits['jittor']:17} {hits['pytorch']:18} {same:13}")


if __name__ == "__main__":
    main()
