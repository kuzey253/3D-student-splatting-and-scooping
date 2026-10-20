"""README accuracy tests, with the gate criterion: max alpha error ≤ 1e-3 against brute force."""
import math
from functools import lru_cache

import pytest
import torch

from mip_sss import design, option_a, option_b
from mip_sss.reference import lambda_integral
from mip_sss.testing import COV_CASES, NUS_EXTENDED, S_MIP, fft_reference, ray_points, rot_cov

TOL = 1e-3


@lru_cache(maxsize=None)
def fft_ref(case, nu):
    return fft_reference(*COV_CASES[case], nu, S_MIP)


@pytest.mark.parametrize("nu", NUS_EXTENDED)
@pytest.mark.parametrize("case", range(len(COV_CASES)))
def test_design_vs_fft(case, nu):
    cov, u, ref = fft_ref(case, nu)
    k = design.kernel(u, torch.zeros(1, 2), cov[None], torch.tensor([nu]), S_MIP)[0]
    assert (k - ref).abs().max() <= TOL


@pytest.mark.parametrize("nu", NUS_EXTENDED)
@pytest.mark.parametrize("case", range(len(COV_CASES)))
def test_option_b_vs_fft(case, nu):
    """Option B alone meets the criterion everywhere, including where the design takes Option A."""
    cov, u, ref = fft_ref(case, nu)
    k = option_b(u, torch.zeros(1, 2), cov[None], torch.tensor([nu]), S_MIP, design.QUAD)[0]
    assert (k - ref).abs().max() <= TOL


@pytest.mark.parametrize("nu", [1.0, 2.0, 4.0, 8.0])
@pytest.mark.parametrize("eigs", [(0.01, 0.04), (0.003, 0.2)], ids=["subpixel", "needle"])
def test_option_b_deep_tails_vs_lambda_integral(eigs, nu):
    """Beyond the FFT domain: the whole ray out to T ≈ 1e-9 (λ-integral reference)."""
    cov = rot_cov(eigs, 0.8)[None]
    u, mu, nut = ray_points(cov[0], nu, S_MIP, t_min=1e-9), torch.zeros(1, 2), torch.tensor([nu])
    ref = lambda_integral(u, mu, cov, nut, S_MIP)[0]
    assert (option_b(u, mu, cov, nut, S_MIP, design.QUAD)[0] - ref).abs().max() <= TOL


def random_fast_components(n, seed):
    """Components the fast path accepts: log-uniform ν, shapes up to 1:64, and sizes from the
    threshold up to 1000x above it (log-uniform; from ρ = 1e-3 where the threshold is 0)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    while len(out) < n:
        nu = float(10 ** (torch.rand(1, generator=g) * 4))
        rho = float(design.fast_threshold(torch.tensor([nu]))[0])
        if math.isinf(rho):
            continue
        lam = max(rho, 1e-3) * S_MIP * float(1000 ** torch.rand(1, generator=g))
        aspect = float(64 ** torch.rand(1, generator=g))
        th = float(torch.rand(1, generator=g) * math.pi)
        out.append((nu, rot_cov((lam, aspect * lam), th)))
    return out


@pytest.mark.parametrize("seed", range(6))
def test_fast_path_rule_holds(seed):
    for nu, cov in random_fast_components(5, seed):
        cov, nut, mu = cov[None], torch.tensor([nu]), torch.zeros(1, 2)
        assert bool(design.use_fast_path(cov, nut, S_MIP)[0])
        u = ray_points(cov[0], nu, S_MIP)
        ref = lambda_integral(u, mu, cov, nut, S_MIP)[0]
        assert (design.kernel(u, mu, cov, nut, S_MIP)[0] - ref).abs().max() <= TOL, (nu, cov)


@pytest.mark.parametrize("nu", [2.0, 4.0, 8.0, 30.0, 100.0])
def test_switch_jump_is_small(nu):
    """At the threshold both branches are within TOL of exact, so they differ by at most 2·TOL."""
    rho = float(design.fast_threshold(torch.tensor([nu]))[0])
    cov = rot_cov((rho * S_MIP, 4 * rho * S_MIP), 0.3)[None]
    u, mu, nut = ray_points(cov[0], nu, S_MIP), torch.zeros(1, 2), torch.tensor([nu])
    a = option_a(u, mu, cov, nut, S_MIP)[0]
    b = option_b(u, mu, cov, nut, S_MIP, design.QUAD)[0]
    assert (a - b).abs().max() <= 2 * TOL


@pytest.mark.parametrize("nu", [1.0, 2.0, 4.0, 8.0])
def test_option_a_tails_too_heavy_for_subpixel(nu):
    """README: Option A's far tail exceeds the true convolution for sub-pixel components."""
    cov = rot_cov(*COV_CASES[0])[None]
    u, mu, nut = ray_points(cov[0], nu, S_MIP, t_min=1e-9), torch.zeros(1, 2), torch.tensor([nu])
    ref = lambda_integral(u, mu, cov, nut, S_MIP)[0]
    ratio = option_a(u, mu, cov, nut, S_MIP)[0] / ref
    far = (ref < 1e-6) & (ref > 1e-9)
    assert far.any() and (ratio[far] > 1).all()
