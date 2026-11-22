"""Display one ModelNet40 test chair and the six raw PointCLIP zero-shot views."""
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")  # The SSH server can write a figure without a desktop.
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts import ROOT
from scripts.modelnet40 import DATA

UPSTREAM = ROOT / "reference/upstream/PointCLIP"
OUTPUT = ROOT / "runs/examples/chair_projection"
sys.path[:0] = [str(UPSTREAM), str(UPSTREAM / "Dassl3D")]
from trainers.mv_utils_zs import PCViews


def load_chair():
    # Row 3 (zero-based) of the first test shard is chair_0978.
    row = 3
    with h5py.File(DATA / "ply_data_test0.h5", "r") as shard:
        stored = shard["data"][row]
        label = int(shard["label"][row, 0])
    classes = (DATA / "shape_names.txt").read_text().splitlines()
    ids = json.loads((DATA / "ply_data_test_0_id2file.json").read_text())
    if classes[label] != "chair" or ids[row] != "chair/chair_0978.ply":
        raise ValueError("The dataset does not match the expected test chair.")
    # Match the author's evaluation loader: take the first 1,024 points in order.
    return stored[:1024], Path(ids[row]).stem


def draw(points, depth, sample):
    fig = plt.figure(figsize=(12, 5.7), facecolor="white")
    grid = fig.add_gridspec(2, 4, width_ratios=[1.45, 1, 1, 1],
                            left=.025, right=.985, top=.83, bottom=.11,
                            wspace=.14, hspace=.30)
    fig.text(.035, .94, f"ModelNet40  /  {sample}", fontsize=20, weight="bold")
    fig.text(.035, .885, "PointCLIP zero-shot projection", fontsize=12, color="#42556a")
    ax = fig.add_subplot(grid[:, 0], projection="3d")
    # Display the chair upright: ModelNet's Y axis becomes the plot's vertical axis.
    # The model and reference.npz both receive the original XYZ coordinates.
    display = np.column_stack((points[:, 0], -points[:, 2], points[:, 1]))
    ax.scatter(*display.T, s=5, color="#276888", alpha=.72, linewidths=0)
    center = (display.min(axis=0) + display.max(axis=0)) / 2
    radius = np.ptp(display, axis=0).max() * .52
    for axis, mid in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        axis(mid - radius, mid + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-55)
    ax.set_axis_off()
    ax.set_title("Input  /  1,024 XYZ points", fontsize=11)
    maximum = depth.max()
    for i, view in enumerate(depth):
        ax = fig.add_subplot(grid[i // 3, 1 + i % 3])
        ax.imshow(view, cmap="gray", vmin=0, vmax=maximum, interpolation="nearest")
        ax.set_title(f"View {i + 1}", loc="left", fontsize=10)
        ax.set_axis_off()
    fig.text(.035, .045, "Six raw 128 x 128 maps in model order.  Black = empty pixel.  "
             "All views use the same depth scale.", fontsize=10, color="#42556a")
    fig.savefig(OUTPUT / "chair_views.png", dpi=150)
    plt.close(fig)


def main():
    points, sample = load_chair()
    # One object per batch. PCViews is the unmodified projector from the paper's repo.
    batch = torch.from_numpy(points).unsqueeze(0).cuda()
    with torch.no_grad():
        depth = PCViews().get_img(batch).cpu().numpy()
    if depth.shape != (6, 128, 128) or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("Unexpected depth maps")
    occupied = (depth > 0).sum(axis=(1, 2))
    if not occupied.all():
        raise ValueError("One or more views are empty.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "reference.npz", points=points, depth=depth)
    draw(points, depth, sample)
    print(f"{sample}: input {tuple(batch.shape)} -> depth {depth.shape}")
    print("Occupied pixels by view:", occupied.tolist())
    print(OUTPUT / "chair_views.png")


if __name__ == "__main__":
    main()
