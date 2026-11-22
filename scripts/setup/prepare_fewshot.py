"""Prepare public RN101/adapter weights and the audited 16-shot train manifest."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from scripts import ROOT
from scripts.modelnet40 import DATA, dataset_index


def export(path, output, expected_digest, scripted):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
        raise ValueError(f"Unexpected source weights: {path}")
    if scripted:
        state = torch.jit.load(str(path), map_location="cpu").state_dict()
    else:
        state = torch.load(str(path), map_location="cpu")["state_dict"]
    arrays = {name: value.detach().cpu().numpy() for name, value in state.items()}
    with NamedTemporaryFile(dir=str(output.parent), suffix=".npz", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        if not output.exists():
            np.savez(tmp, **arrays)
        source = output if output.exists() else tmp
        with np.load(source, allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            for name, array in arrays.items():
                assert array.dtype == saved[name].dtype
                np.testing.assert_array_equal(saved[name], array)
        if not output.exists():
            tmp.replace(output)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"{output.name}: {len(arrays)} arrays verified", flush=True)


def main():
    weights = ROOT / "weights"
    export(weights / "clip/RN101.pt", weights / "clip/RN101.npz",
           "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599", True)
    export(weights / "adapter/model-best.pth.tar", weights / "adapter/rn101_16shot.npz",
           "f21942b9fe5b51075f6fd853549e3f8cbcd973f4ffcbd11f23a6f347f0edc6dc", False)
    classes = (DATA / "shape_names.txt").read_text().splitlines()
    groups = defaultdict(list)
    for name, y, _, _ in dataset_index("train", classes):
        groups[y].append(name)
    assert len(groups) == 40 and sum(map(len, groups.values())) == 9840
    rng = random.Random(2)
    ids = [name for group in groups.values() for name in rng.sample(group, 16)]
    test_ids = {r[0] for r in dataset_index("test", classes)}
    assert len(set(ids)) == 640 and not set(ids) & test_ids
    path = ROOT / "manifests/modelnet40_16shot_seed2.json"
    manifest = dict(seed=2, shots=16, classes=classes, sample_ids=ids)
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Different 16-shot manifest already present")
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"16 training samples per class, zero test overlap: {path}", flush=True)


if __name__ == "__main__":
    main()
