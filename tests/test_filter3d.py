"""Port of Mip-Splatting's compute_3D_filter (README M3.1)."""
import math

import torch

from mip_sss.filter3d import PinholeCamera, filter_3d_std, fov2focal, max_sampling_rate


def camera(R_w2c=None, t=(0.0, 0.0, 0.0), focal=100.0, size=(200, 200)):
    R_w2c = torch.eye(3) if R_w2c is None else R_w2c
    return PinholeCamera(R=R_w2c.T, T=torch.tensor(t), focal_x=focal, focal_y=focal,
                         image_width=size[0], image_height=size[1])


def test_single_camera():
    xyz = torch.tensor([[0.0, 0.0, 5.0]])
    std = filter_3d_std(xyz, [camera()])
    assert torch.allclose(std, torch.tensor([math.sqrt(0.2) * 5 / 100]))


def test_min_depth_and_max_focal_over_cameras():
    xyz = torch.tensor([[0.0, 0.0, 5.0]])
    near_cam = camera(t=(0.0, 0.0, -2.0), focal=50.0)   # sees the point at depth 3
    far_cam = camera(focal=80.0)                          # depth 5
    blind_cam = camera(t=(0.0, 0.0, -10.0), focal=400.0)  # point behind it, but its focal counts
    f_hat = max_sampling_rate(xyz, [near_cam, far_cam, blind_cam])
    assert torch.allclose(f_hat, torch.tensor([400.0 / 3.0]))


def test_screen_margin_and_unseen_points():
    # a camera with focal 100 and width 200: x_pix = x/z*100 + 100
    xyz = torch.tensor([[-1.1 * 5, 0.0, 5.0],    # x_pix = -10  → inside the 15% margin (≥ -30)
                        [-1.4 * 5, 0.0, 5.0],    # x_pix = -40  → outside
                        [0.0, 0.0, 7.0],         # centre, depth 7
                        [0.0, 0.0, 0.1]])        # closer than the 0.2 near plane → unseen
    f_hat = max_sampling_rate(xyz, [camera()])
    # unseen points take the largest depth among seen points (7)
    assert torch.allclose(f_hat, torch.tensor([100 / 5.0, 100 / 7.0, 100 / 7.0, 100 / 7.0]))


def test_rotated_camera_uses_transposed_R():
    # camera looking along world +x: world x maps to camera z
    R_w2c = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    xyz = torch.tensor([[4.0, 0.0, 0.0]])
    f_hat = max_sampling_rate(xyz, [camera(R_w2c=R_w2c)])
    assert torch.allclose(f_hat, torch.tensor([100 / 4.0]))


def test_fov2focal():
    assert math.isclose(fov2focal(2 * math.atan(0.5), 200), 200.0)
