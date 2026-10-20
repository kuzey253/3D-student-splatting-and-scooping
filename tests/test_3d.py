"""3D smoothing filter + 2D filter (README "Adding the 3D smoothing filter"), for the design."""
import pytest
import torch

from mip_sss import design, mip_gauss_3d, option_b_3d, sss_kernel
from mip_sss.kernels import project
from mip_sss.reference import lambda_integral_3d
from mip_sss.testing import S_MIP

TOL = 1e-3


def scene(seed=5, flat=False):
    """Six components: world-space scales around 0.03 (about 2 px at 60 px/unit), random 3D filters.

    With flat=True one axis of every component is 1e-4 world units: a thin splat.
    """
    g = torch.Generator().manual_seed(seed)
    n = 6
    q, _ = torch.linalg.qr(torch.randn(n, 3, 3, generator=g))
    scales = torch.exp(torch.randn(n, 3, generator=g) * 1.0 - 3.5)
    if flat:
        scales[:, 0] = 1e-4
    cov3 = q @ torch.diag_embed(scales ** 2) @ q.transpose(-1, -2)
    M = torch.randn(n, 2, 3, generator=g) * 6 + torch.tensor([[60.0, 0, 0], [0, 60.0, 0]])
    mu2 = torch.randn(n, 2, generator=g)
    sigma3_sq = (torch.rand(n, generator=g) * 0.02) ** 2
    x = torch.linspace(-6, 6, 49)
    X, Y = torch.meshgrid(x, x, indexing="xy")
    return torch.stack([X.ravel(), Y.ravel()], -1), mu2, cov3, M, sigma3_sq


@pytest.mark.parametrize("nu", [1.0, 4.0, 30.0, 1e3])
def test_reduces_to_sss_without_filters(nu):
    u, mu2, cov3, M, _ = scene()
    nut, zero = torch.full((mu2.shape[0],), nu), torch.zeros(mu2.shape[0])
    expect = sss_kernel(u, mu2, project(cov3, M), nut)
    assert (design.kernel_3d(u, mu2, cov3, M, nut, zero, 0.0) - expect).abs().max() <= 1e-12
    assert (option_b_3d(u, mu2, cov3, M, nut, zero, 0.0, design.QUAD) - expect).abs().max() <= 1e-12


@pytest.mark.parametrize("flat", [False, True], ids=["round", "flat"])
def test_reduces_to_mip_splatting(flat):
    u, mu2, cov3, M, sigma3_sq = scene(flat=flat)
    nut = torch.full((mu2.shape[0],), 1e4)
    expect = mip_gauss_3d(u, mu2, cov3, M, sigma3_sq, S_MIP)
    assert (design.kernel_3d(u, mu2, cov3, M, nut, sigma3_sq, S_MIP) - expect).abs().max() <= 1e-4
    assert (option_b_3d(u, mu2, cov3, M, nut, sigma3_sq, S_MIP, design.QUAD)
            - expect).abs().max() <= 1e-4


@pytest.mark.parametrize("nu", [1.0, 2.0, 4.0, 8.0, 30.0, 100.0, 1e3])
@pytest.mark.parametrize("flat", [False, True], ids=["round", "flat"])
@pytest.mark.parametrize("seed", [5, 6])
def test_accuracy_vs_lambda_integral(seed, flat, nu):
    u, mu2, cov3, M, sigma3_sq = scene(seed, flat)
    nut = torch.full((mu2.shape[0],), nu)
    ref = lambda_integral_3d(u, mu2, cov3, M, nut, sigma3_sq, S_MIP)
    assert (design.kernel_3d(u, mu2, cov3, M, nut, sigma3_sq, S_MIP) - ref).abs().max() <= TOL
    assert (option_b_3d(u, mu2, cov3, M, nut, sigma3_sq, S_MIP, design.QUAD) - ref).abs().max() <= TOL


def test_3d_filter_factor_is_mip_formula_at_large_nu():
    """With s = 0 and ν → ∞ the peak is Mip-Splatting's 3D det ratio sqrt(|Σ|/|Σ + σ²I|)."""
    _, mu2, cov3, M, sigma3_sq = scene()
    nut = torch.full((mu2.shape[0],), 1e4)
    peak = torch.stack([design.kernel_3d(mu2[i:i + 1], mu2[i:i + 1], cov3[i:i + 1], M[i:i + 1],
                                         nut[i:i + 1], sigma3_sq[i:i + 1], 0.0)[0, 0]
                        for i in range(mu2.shape[0])])
    expect = torch.sqrt(torch.det(cov3) / torch.det(cov3 + sigma3_sq[:, None, None] * torch.eye(3)))
    assert torch.allclose(peak, expect, rtol=1e-12)
