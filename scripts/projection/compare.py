"""Compare Jittor's six chair projections to the saved author-code output."""
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts import ROOT

REFERENCE = ROOT / "runs/examples/chair_projection/reference.npz"
OUTPUT = ROOT / "runs/jittor/chair_projection"
TOLERANCE = 1e-6


def load_reference():
    with np.load(REFERENCE, allow_pickle=False) as saved:
        points, depth = saved["points"], saved["depth"]
    data_dir = ROOT / "data/modelnet40_ply_hdf5_2048"
    ids = json.loads((data_dir / "ply_data_test_0_id2file.json").read_text())
    with h5py.File(data_dir / "ply_data_test0.h5", "r") as shard:
        original = shard["data"][3, :1024, :]
    if ids[3] != "chair/chair_0978.ply" or not np.array_equal(points, original):
        raise ValueError("Expected the original 1,024 XYZ rows for chair_0978")
    if depth.shape != (6, 128, 128) or depth.dtype != np.float32 or not np.isfinite(depth).all():
        raise ValueError("Invalid PyTorch baseline")
    return points, depth


def jittor_depth(points):
    # The projection uses native Jittor CUDA operations.
    for key in ("use_mkl", "use_cutt", "use_cutlass", "use_nccl"):
        os.environ.setdefault(key, "0")
    os.environ.setdefault("conv_opt", "1")
    os.environ.setdefault("cache_name", "pointclip_projection")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import jittor as jt
    sys.path.insert(0, str(ROOT / "src"))
    from pointclip_jittor.geometry.views import project_views

    jt.flags.use_cuda = 1
    # PyTorch 1.11 used TF32 for the original chair's batched rotations on RTX 3090.
    jt.flags.use_tensorcore = 1
    with jt.no_grad():
        actual = project_views(jt.array(points).unsqueeze(0)).numpy()
    return actual


def draw_comparison(expected, actual, error):
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.7))
    fig.subplots_adjust(left=.12, right=.98, top=.86, bottom=.12, wspace=.07, hspace=.10)
    fig.text(.04, .94, "PointCLIP  /  chair_0978", size=20, weight="bold")
    depth_max = max(expected.max(), actual.max())
    error_max = max(float(error.max()), TOLERANCE)
    for row, (values, title) in enumerate(((expected, "PyTorch"), (actual, "Jittor"), (error, "Abs. error"))):
        for col, view in enumerate(values):
            ax = axes[row, col]
            ax.imshow(view, cmap="magma" if row == 2 else "gray", interpolation="nearest",
                      vmin=0, vmax=error_max if row == 2 else depth_max)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"View {col + 1}", size=10)
            if col == 0:
                ax.set_ylabel(title, rotation=0, ha="right", va="center", labelpad=12)
    fig.text(.12, .05, f"Depth scale 0 to {depth_max:.3f}   |   Error scale 0 to {error_max:.2e}"
             f"   |   Maximum observed error {error.max():.2e}", fontsize=10)
    fig.savefig(OUTPUT / "comparison.png", dpi=150)
    plt.close(fig)


def main():
    points, expected = load_reference()
    actual = jittor_depth(points)
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.isfinite(actual).all():
        raise ValueError("Jittor returned invalid depth maps")
    error = np.abs(expected - actual)
    views = []
    print("View  Foreground  Mask differences  Max absolute error")
    for i, (a, b, e) in enumerate(zip(expected, actual, error)):
        mask_diff = int(np.count_nonzero((a != 0) != (b != 0)))
        row = dict(view=i + 1, occupied_pixels=int(np.count_nonzero(a)), mask_differences=mask_diff,
                   max_abs_error=float(e.max()), different_values=int(np.count_nonzero(e)))
        views.append(row)
        print(f"{i+1:>4}  {row['occupied_pixels']:>10}  {mask_diff:>16}  {e.max():>18.9g}")
    passed = all(v["mask_differences"] == 0 for v in views) and float(error.max()) <= TOLERANCE
    report = dict(sample="chair_0978", backend="Jittor CUDA", matrix_precision="TF32",
                  input_shape=list(points.shape),
                  output_shape=list(actual.shape), dtype=str(actual.dtype), tolerance=TOLERANCE,
                  max_abs_error=float(error.max()), exactly_equal=bool(np.array_equal(expected, actual)),
                  within_tolerance=bool(passed), per_view=views)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT / "depth.npy", actual)
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    draw_comparison(expected, actual, error)
    print(f"Exactly equal: {report['exactly_equal']} | Max absolute error: {error.max():.9g}")
    print(f"{'PASS' if passed else 'FAIL'}  (identical masks and absolute error <= {TOLERANCE:g})")
    print(OUTPUT)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
