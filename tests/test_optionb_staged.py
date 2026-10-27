"""The hand-derived Option B backward (mip_sss/optionb_staged.py, transliterated into CUDA for M4)
against autograd of a direct implementation of the same definitions."""
import math

import pytest
import torch

from mip_sss import design, optionb_staged as ob
from mip_sss.kernels import project


def sym6(M):
    """Gradient w.r.t. a symmetric matrix's 6 free entries (off-diagonals counted once)."""
    i, j = zip(*[(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)])
    S = M + M.T - torch.diag(torch.diag(M))
    return S[list(i), list(j)]


def case(seed, sigma):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    S3 = q @ torch.diag(torch.exp(torch.randn(3, generator=g) - 7)) @ q.T  # ~0.03 world units
    T = torch.randn(2, 3, generator=g) * 6 + torch.tensor([[40.0, 0, 0], [0, 40.0, 0]])
    d = torch.randn(40, 2, generator=g) * 1.5
    dLdT = torch.randn(40, generator=g)
    return S3, torch.tensor(sigma), T, d, dLdT


@pytest.mark.parametrize("nu", [1.0, 1.5, 3.0, 30.0, 600.0])
@pytest.mark.parametrize("sigma", [0.0, 0.004])
@pytest.mark.parametrize("seed", [0, 1])
def test_manual_backward_matches_autograd(seed, sigma, nu):
    S3, sig, T, d, dLdT = case(seed, sigma)
    s = 0.1
    nut = torch.tensor(nu)
    pre = ob.preprocess(S3, sig, T, nut, s)
    val, acc = ob.pixel(d, pre, nut, dLdT)
    dS3_m, dT_m = ob.chain(S3, sig, T, acc)

    leaves = [x.clone().requires_grad_(True) for x in (S3, T, nut, d)]
    ref = ob.direct(leaves[0], sig, leaves[1], leaves[2], s, leaves[3])
    (ref * dLdT).sum().backward()
    S3g, Tg, nug, dg = (x.grad for x in leaves)

    assert torch.allclose(val, ref.detach(), rtol=1e-12, atol=1e-14)
    assert torch.allclose(acc["dd"], dg, rtol=1e-9, atol=1e-12)
    assert torch.allclose(acc["dnu"], nug, rtol=1e-9, atol=1e-14)
    assert torch.allclose(dT_m, Tg, rtol=1e-9, atol=1e-12)
    assert torch.allclose(sym6(dS3_m), sym6(S3g), rtol=1e-8, atol=1e-6 * float(sym6(S3g).abs().max()))
    if nu < 400:
        assert not pre["fast"], "small ν should take Option B here"


@pytest.mark.parametrize("sigma", [0.0, 0.004])
@pytest.mark.parametrize("nu", [1.0, 3.0, 30.0, 600.0])
def test_staged_forward_is_the_design_kernel(sigma, nu):
    """The staged T equals mip_sss.design.kernel_3d (the M1/M3 reference) at the same pixels."""
    S3, sig, T, d, _ = case(3, sigma)
    s = 0.1
    nut = torch.tensor(nu)
    staged = ob.pixel(d, ob.preprocess(S3, sig, T, nut, s), nut)
    ref = design.kernel_3d(d, torch.zeros(1, 2), S3[None], T[None], nut[None],
                           torch.tensor([sigma ** 2]), s)[0]
    # design.kernel_3d works with offsets u − μ; staged uses d = μ − u: the kernel is even in d
    assert torch.allclose(staged, ref, rtol=1e-10, atol=1e-13)


def test_nodes_match_design_quadrature():
    from mip_sss.quadrature import nodes
    for nu in (1.0, 2.5, 40.0):
        nd = ob.nodes_with_derivs(torch.tensor(nu))
        z, logw = nodes(torch.tensor([nu]), design.QUAD)
        assert torch.allclose(nd["z"], z[0]) and torch.allclose(nd["logw"], logw[0])


def test_projection_term_is_the_3d_filter():
    """P = σ²TTᵀ is the projection of σ²I, so Σ2 + P = T(Σ3 + σ²I)Tᵀ exactly."""
    S3, sig, T, _, _ = case(4, 0.01)
    lhs = T @ S3 @ T.T + sig ** 2 * T @ T.T
    rhs = project((S3 + sig ** 2 * torch.eye(3))[None], T[None])[0]
    assert torch.allclose(lhs, rhs)
    assert math.isfinite(float(lhs.sum()))
