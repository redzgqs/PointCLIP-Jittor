"""ModelNet40 HDF5 shard and sample identity checks shared by both frameworks."""
import json
from pathlib import Path
import re

import numpy as np

from scripts import ROOT

DATA = ROOT / "data/modelnet40_ply_hdf5_2048"
CHAIR = "chair/chair_0978.ply"


def shard_index_files(split, directory=DATA):
    """Yield each HDF5 path and the sample IDs in its companion JSON file."""
    for entry in (directory / f"{split}_files.txt").read_text().splitlines():
        if not entry.strip():
            continue
        shard = directory / Path(entry.strip()).name
        match = re.fullmatch(rf"ply_data_{split}(\d+)\.h5", shard.name)
        if not match:
            raise ValueError(f"Unexpected ModelNet40 shard {shard.name}")
        index = directory / f"ply_data_{split}_{match.group(1)}_id2file.json"
        yield shard, json.loads(index.read_text())


def indexed_shards(split, classes, directory):
    import h5py

    for shard, names in shard_index_files(split, directory):
        with h5py.File(shard, "r") as h5:
            labels = h5["label"][:].reshape(-1)
            if len(h5["data"]) != len(names) or len(names) != len(labels):
                raise ValueError(f"HDF5 and sample IDs disagree: {shard.name}")
        for name, label in zip(names, labels):
            if not 0 <= label < len(classes) or Path(name).parent.name != classes[label]:
                raise ValueError(f"Sample name and class label disagree: {name}, {label}")
        yield shard, names, labels


def dataset_index(split, classes, directory=DATA):
    """Return (sample ID, class index, HDF5 path, row) in the original shard order."""
    records = [(name, int(label), shard, row)
               for shard, names, labels in indexed_shards(split, classes, directory)
               for row, (name, label) in enumerate(zip(names, labels))]
    if len({name for name, _, _, _ in records}) != len(records):
        raise ValueError(f"Duplicate {split} sample IDs")
    return records


def test_index(directory):
    classes = (directory / "shape_names.txt").read_text().splitlines()
    ids, labels, shards = [], [], []
    for shard, names, truth in indexed_shards("test", classes, directory):
        ids.extend(names)
        labels.append(truth)
        shards.append(shard)
    if len(classes) != 40 or len(set(ids)) != 2468 or len(ids) != 2468:
        raise ValueError("Expected the complete 40-class ModelNet40 test split")
    return classes, np.asarray(ids), np.concatenate(labels), shards
