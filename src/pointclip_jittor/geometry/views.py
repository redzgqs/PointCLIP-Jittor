"""PointCLIP camera geometry for six-view zero-shot and ten-view few-shot classification."""
from math import pi

import jittor as jt

from .rasterize import RESOLUTION, project_camera_points


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


def fewshot_rotations():
    # Use the exact camera order of upstream trainers/mv_utils_fs.py. Multiples
    # of pi/4 are first formed in double precision, as in its NumPy constants.
    eighth_turns = ((0, 0, 2), (2, 0, 2), (4, 0, 2), (6, 0, 2),
                    (5, -1, 2), (5, 1, 2), (7, -1, 2), (7, 1, 2),
                    (0, -2, 2), (0, 2, 2))
    angles = jt.array([[v * pi / 4 for v in xyz] for xyz in eighth_turns]).float32()
    x, y, z = [angles[:, axis] for axis in range(3)]
    c, s = jt.cos, jt.sin
    one, zero = jt.ones_like(x), jt.zeros_like(x)

    def matrices(entries):
        return jt.stack(entries, dim=1).reshape(10, 3, 3)

    rx = matrices([one, zero, zero, zero, c(x), -s(x), zero, s(x), c(x)])
    ry = matrices([c(y), zero, s(y), zero, one, zero, -s(y), zero, c(y)])
    rz = matrices([c(z), -s(z), zero, s(z), c(z), zero, zero, zero, one])
    return rx @ ry @ rz


def project_fewshot_views(points):
    """Project [B,N,3] points; the ten views of each object are consecutive."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("Expected points of shape [batch, points, 3]")
    batch, count, _ = points.shape
    rotations = fewshot_rotations().permute(0, 2, 1)
    repeated = points.float32().unsqueeze(1).repeat(1, 10, 1, 1).reshape(-1, count, 3)
    camera = repeated @ rotations.repeat(batch, 1, 1)
    camera = camera + jt.array([0.0, 0.0, 1.4]).float32()
    # PCViews explicitly sets size_x=size_y=1, just as the six-view renderer does.
    return project_camera_points(camera, resolution=224)
