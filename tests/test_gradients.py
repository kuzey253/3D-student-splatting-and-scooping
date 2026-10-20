"""README test: gradcheck for μ, Σ, ν, o. Plus SSS's ν gradient, pinned."""
import pytest
import torch

from mip_sss import design, option_a, option_b
from mip_sss.testing import S_MIP


def inputs():
    g = torch.Generator().manual_seed(1)
    mu = torch.randn(3, 2, generator=g) * 0.3
    # (a, b, c) of [[a, b], [b, c]]: sub-pixel (Option B), pixel-sized (Option B), large (Option A)
    cov_params = torch.tensor([[0.04, 0.01, 0.09], [1.5, -0.4, 0.8], [60.0, 10.0, 45.0]])
    nu = torch.tensor([2.3, 45.0, 400.0])
    o = torch.tensor([0.7, -0.4, 0.9])
    for t in (mu, cov_params, nu, o):
        t.requires_grad_(True)
    u = torch.randn(9, 2, generator=g) * 2
    return u, (mu, cov_params, nu, o)


def cov_of(p):
    return torch.stack([torch.stack([p[:, 0], p[:, 1]], -1), torch.stack([p[:, 1], p[:, 2]], -1)], -2)


def test_inputs_cover_both_branches():
    _, (_, p, nu, _) = inputs()
    assert design.use_fast_path(cov_of(p), nu, S_MIP).tolist() == [False, False, True]


def test_gradcheck_design():
    u, args = inputs()
    assert torch.autograd.gradcheck(
        lambda mu, p, nu, o: o[:, None] * design.kernel(u, mu, cov_of(p), nu, S_MIP), args)


def test_gradcheck_option_b():
    """Option B alone; ν moves the adaptive nodes, so this also checks ∂/∂ν through them."""
    u, args = inputs()
    assert torch.autograd.gradcheck(
        lambda mu, p, nu, o: o[:, None] * option_b(u, mu, cov_of(p), nu, S_MIP, design.QUAD), args)


def test_gradcheck_option_a():
    u, args = inputs()
    assert torch.autograd.gradcheck(
        lambda mu, p, nu, o: o[:, None] * option_a(u, mu, cov_of(p), nu, S_MIP), args)


def test_sss_nu_gradient_drops_the_log_term():
    """Pins finding 2 in PLAN.md: SSS's backward.cu:592 omits −½·ln(1 + b/ν)·G from ∂G/∂ν."""
    b = torch.tensor([0.1, 1.0, 9.0, 100.0])
    nu = torch.tensor([1.5, 4.0, 30.0, 100.0], requires_grad=True)
    G = torch.pow(1 + b / nu, -(nu + 2) / 2)
    exact = torch.autograd.grad(G.sum(), nu)[0]
    nu = nu.detach()
    G = G.detach()
    sss = (nu + 2) / 2 * (b / nu ** 2) * torch.pow(1 + b / nu, -(nu + 4) / 2)
    assert not torch.allclose(sss, exact, rtol=1e-3)
    assert torch.allclose(sss - 0.5 * torch.log1p(b / nu) * G, exact, rtol=1e-12)


@pytest.mark.parametrize("nu", [1.0, 30.0, 300.0, 1e4])
def test_nu_gradient_is_nonzero_everywhere(nu):
    """No branch freezes ν (the README's Gaussian fallback would give ∂/∂ν = 0 above ν = 30)."""
    cov = torch.tensor([[[0.3, 0.05], [0.05, 0.5]], [[50.0, 5.0], [5.0, 30.0]]])
    nut = torch.full((2,), nu, requires_grad=True)
    u = torch.tensor([[0.8, -0.5], [6.0, 4.0]])
    k = design.kernel(u, torch.zeros(2, 2), cov, nut, S_MIP)
    grad = torch.autograd.grad(k.sum(), nut)[0]
    assert (grad.abs() > 0).all()
