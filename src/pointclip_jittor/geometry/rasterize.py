"""PointCLIP inverse-depth rasterization for camera-space point clouds."""
import jittor as jt


RESOLUTION = 128


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
