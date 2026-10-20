"""Mip-Splatting's 3D smoothing filter size, ported from `GaussianModel.compute_3D_filter`
(autonomousvision/mip-splatting, scene/gaussian_model.py:142-191, commit dda02ab).

Mip-Splatting writes the maximal sampling rate as ν̂. Here it is f̂ = focal / depth, so it never
collides with SSS's tail parameter ν. The filter's standard deviation is sqrt(filter_variance) / f̂.
"""
import math
from dataclasses import dataclass

import torch


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


@dataclass
class PinholeCamera:
    """The camera fields compute_3D_filter reads. R is stored transposed, as in 3DGS."""
    R: torch.Tensor
    T: torch.Tensor
    focal_x: float
    focal_y: float
    image_width: int
    image_height: int


@torch.no_grad()
def max_sampling_rate(xyz, cameras, near=0.2, margin=0.15):
    """f̂ per point: max focal_x over all cameras / min depth over the cameras that see the point.

    A camera sees a point if its camera-space z > near and its projection lands inside the image
    widened by `margin` on every side (the principal point is assumed centred, as upstream).
    Points no camera sees get the largest depth among the seen points. As upstream, the focal
    length is the maximum over all cameras, not only the ones that see the point.
    """
    distance = torch.full((xyz.shape[0],), 100000.0, dtype=xyz.dtype, device=xyz.device)
    seen = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    focal = 0.0
    for cam in cameras:
        R = torch.as_tensor(cam.R, dtype=xyz.dtype, device=xyz.device)
        T = torch.as_tensor(cam.T, dtype=xyz.dtype, device=xyz.device)
        xyz_cam = xyz @ R + T[None, :]
        valid_depth = xyz_cam[:, 2] > near
        z = torch.clamp(xyz_cam[:, 2], min=0.001)
        x = xyz_cam[:, 0] / z * cam.focal_x + cam.image_width / 2.0
        y = xyz_cam[:, 1] / z * cam.focal_y + cam.image_height / 2.0
        in_screen = ((x >= -margin * cam.image_width) & (x <= (1 + margin) * cam.image_width)
                     & (y >= -margin * cam.image_height) & (y <= (1 + margin) * cam.image_height))
        valid = valid_depth & in_screen
        distance[valid] = torch.minimum(distance[valid], z[valid])
        seen |= valid
        focal = max(focal, cam.focal_x)
    distance[~seen] = distance[seen].max()
    return focal / distance


def filter_3d_std(xyz, cameras, filter_variance=0.2, **kwargs):
    """Standard deviation σ3 of the 3D smoothing filter; Mip-Splatting's `filter_3D`."""
    return math.sqrt(filter_variance) / max_sampling_rate(xyz, cameras, **kwargs)
