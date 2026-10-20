"""README tests, gate criterion: exact limits. s = 0 gives SSS, ν → ∞ gives Mip-Splatting, peak 1."""
import math

import pytest
import torch

from mip_sss import design, gauss_mip, option_b, sss_kernel
from mip_sss.testing import COV_CASES, NUS_EXTENDED, S_MIP, ray_points, rot_cov

# The design takes Option A at s = 0, so Option B's own limit is checked separately.
BRANCHES = {
    "design": design.kernel,
    "option_b": lambda u, mu, cov, nu, s: option_b(u, mu, cov, nu, s, design.QUAD),
}


def grid_and_rays(cov, nu, s):
    """A 2D pixel grid around the component plus long rays for the tails."""
    half = 4 * math.sqrt(float(torch.linalg.eigvalsh(cov).max())) + 2
    x = torch.linspace(-half, half, 61)
    X, Y = torch.meshgrid(x, x, indexing="xy")
    return torch.cat([torch.stack([X.ravel(), Y.ravel()], -1), ray_points(cov, nu, s)])


@pytest.mark.parametrize("branch", BRANCHES)
@pytest.mark.parametrize("nu", NUS_EXTENDED)
@pytest.mark.parametrize("case", range(len(COV_CASES)))
def test_reduces_to_sss_exactly(branch, nu, case):
    cov = rot_cov(*COV_CASES[case])[None]
    u, mu, nut = grid_and_rays(cov[0], nu, 0.0), torch.zeros(1, 2), torch.tensor([nu])
    err = (BRANCHES[branch](u, mu, cov, nut, 0.0) - sss_kernel(u, mu, cov, nut)).abs().max()
    assert err <= 1e-12


@pytest.mark.parametrize("case", range(len(COV_CASES)))
def test_reduces_to_mip_splatting(case):
    cov = rot_cov(*COV_CASES[case])[None]
    u, mu, nut = grid_and_rays(cov[0], 1e4, S_MIP), torch.zeros(1, 2), torch.tensor([1e4])
    err = (design.kernel(u, mu, cov, nut, S_MIP) - gauss_mip(u, mu, cov, S_MIP)).abs().max()
    assert err <= 1e-4


@pytest.mark.parametrize("branch", BRANCHES)
def test_converges_to_mip_splatting(branch):
    """The gap to G_mip shrinks like 1/ν: ν → ∞ is an exact limit, not just a small error."""
    cov = rot_cov(*COV_CASES[2])[None]
    mu = torch.zeros(1, 2)
    errs = []
    for nu in (1e3, 1e4, 1e5):
        u = grid_and_rays(cov[0], nu, S_MIP)
        k = BRANCHES[branch](u, mu, cov, torch.tensor([nu]), S_MIP)
        errs.append(float((k - gauss_mip(u, mu, cov, S_MIP)).abs().max()))
    assert errs[1] < errs[0] / 5 and errs[2] < errs[1] / 5


@pytest.mark.parametrize("branch", BRANCHES)
@pytest.mark.parametrize("nu", NUS_EXTENDED)
def test_unit_peak(branch, nu):
    cov = torch.stack([rot_cov(*c) for c in COV_CASES])
    mu = torch.randn(len(COV_CASES), 2, generator=torch.Generator().manual_seed(0))
    nut = torch.full((len(COV_CASES),), nu)
    peak = torch.stack([BRANCHES[branch](mu[i:i + 1], mu[i:i + 1], cov[i:i + 1], nut[i:i + 1], 0.0)[0, 0]
                        for i in range(len(COV_CASES))])
    assert (peak - 1).abs().max() <= 1e-12
