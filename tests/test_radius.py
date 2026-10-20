"""Culling radius (README M4.4): |alpha| < 1/255 outside it, for both branches of the design."""
import math

import pytest
import torch

from mip_sss import design
from mip_sss.kernels import sss_radius
from mip_sss.testing import COV_CASES, S_MIP, rot_cov


@pytest.mark.parametrize("nu", [1.0, 1.5, 4.0, 30.0, 100.0, 300.0, 1e4])
def test_alpha_below_threshold_outside_radius(nu):
    cov = torch.stack([rot_cov(*c) for c in COV_CASES] + [rot_cov((200.0, 900.0), 0.5)])
    n = cov.shape[0]
    nut, o = torch.full((n,), nu), torch.full((n,), -1.0)
    r = design.radius(cov, nut, o, S_MIP)
    th = torch.linspace(0, 2 * math.pi, 64)
    dirs = torch.stack([torch.cos(th), torch.sin(th)], -1)
    for i in range(n):
        for scale in (1.0 + 1e-9, 1.5, 4.0):
            alpha = o[i] * design.kernel(dirs * r[i] * scale, torch.zeros(1, 2), cov[i:i + 1],
                                         nut[i:i + 1], S_MIP)
            assert alpha.abs().max() < 1 / 255


def test_sss_radius_matches_cuda_table():
    """The port reproduces forward.cu, including its reversed interpolation within [n, n+1)."""
    # diag(4, 1): mid² − det = 2.25 > 0.1, so the 0.1 clamp is inactive and λ_max = 4 exactly
    cov = torch.diag(torch.tensor([4.0, 1.0])).expand(6, 2, 2)
    nu = torch.tensor([1.0, 1.5, 1.999, 2.0, 12.5, 13.0])
    expect = torch.ceil(torch.tensor([9.925, 0.5 * (63.657 - 9.925) + 9.925,
                                      0.999 * (63.657 - 9.925) + 9.925, 5.841,
                                      0.5 * (3.055 - 3.012) + 3.012, 3.0]) * 2)
    assert torch.equal(sss_radius(cov, nu), expect)
