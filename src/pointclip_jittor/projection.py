"""PointCLIP V1 zero-shot projection in Jittor. Output [B*6, resolution, resolution]."""
from math import pi

import jittor as jt


RESOLUTION = 128


def view_rotations():
    """Six fixed cameras, in the same order as the author's PCViews."""
    # Each triple is (x, y, z) in quarter turns; R = Rx @ Ry @ Rz.
    turns = [(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1), (0, -1, 1), (0, 1, 1)]
    angles = jt.array(turns).float32() * (pi / 2)
    x, y, z = [angles[:, axis] for axis in range(3)]
    c, s = jt.cos, jt.sin
    one, zero = jt.ones_like(x), jt.zeros_like(x)

    def matrices(entries):
        return jt.stack(entries, dim=1).reshape(6, 3, 3)

    rx = matrices([one, zero, zero, zero, c(x), -s(x), zero, s(x), c(x)])
    ry = matrices([c(y), zero, s(y), zero, one, zero, -s(y), zero, c(y)])
    rz = matrices([c(z), -s(z), zero, s(z), c(z), zero, zero, zero, one])
    return rx @ ry @ rz


def project_camera_points(camera, resolution=RESOLUTION):
    """Rasterize one image per camera-coordinate point cloud [K, N, 3]."""
    z = camera[:, :, 2]
    # The author uses X for image rows and Y for columns. The 1x1 footprint
    # samples ceil(coord - 0.5), including the half-pixel tie convention.
    row = jt.ceil((camera[:, :, 0] / (z + 1e-12) + 1) * (resolution / 2) - 0.5).int32()
    col = jt.ceil((camera[:, :, 1] / (z + 1e-12) + 1) * (resolution / 2) - 0.5).int32()
    valid = (z >= 0) & (row >= 0) & (row < resolution) & (col >= 0) & (col < resolution)
    pixels = jt.where(valid, row * resolution + col, 0).int32()
    weights = jt.where(valid, 1 / (z + 1e-12), 0.0)

    def pixel_sum(values):
        # Source [view, point] contributes to destination [view, pixel_id].
        return values.reindex_reduce("add", [camera.shape[0], resolution**2],
                                     ["i0", "@e0(i0,i1)"], extras=[pixels])

    sum_weights = pixel_sum(weights)
    sum_depths = pixel_sum(z * weights)
    depth = sum_depths / jt.where(sum_weights == 0, 1.0, sum_weights)
    return depth.reshape(-1, resolution, resolution)


def project_views(points, resolution=RESOLUTION):
    """Project float32 XYZ rows for each object; all its six views stay together."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("Expected points with shape [batch, points, 3]")
    batch, count, _ = points.shape
    rotations = view_rotations().permute(0, 2, 1)
    # A rank-3 batched matmul matches the author's CUDA path; keep object-major order.
    repeated = points.float32().unsqueeze(1).repeat(1, 6, 1, 1).reshape(-1, count, 3)
    camera = repeated @ rotations.repeat(batch, 1, 1)
    camera = camera + jt.array([0.0, 0.0, 1.6]).float32()
    return project_camera_points(camera, resolution)
