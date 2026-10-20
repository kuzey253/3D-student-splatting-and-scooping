"""README test, gate criterion: the 2D filter preserves mass exactly. ∫T du = 2π|Σ|^½ for every ν.

Both branches are exact by construction (Option A by its det-ratio factor, Option B because every
control-variate residual node integrates to zero). These tests check that numerically.
"""
import math

import pytest
import torch

from mip_sss import design, mass_option_b, option_b
from mip_sss.testing import S_MIP, rot_cov


def grid_integral(fn, eigs, nu):
    """Riemann sum over ±100 σ_max, with spacing a quarter of the narrowest core scale.

    The integrands are smooth on that scale, so the sum is spectrally accurate; ±100 σ_max leaves
    out < 1e-6 of the mass for ν ≥ 4.
    """
    half = 100 * math.sqrt(max(eigs) + S_MIP)
    h = 0.25 * math.sqrt((min(eigs) + S_MIP) * nu / (nu + 2))
    x = torch.arange(-half, half + h / 2, h)
    total = 0.0
    for xs in torch.split(x, 200):
        X, Y = torch.meshgrid(xs, x, indexing="ij")
        total += float(fn(torch.stack([X.ravel(), Y.ravel()], -1)).sum()) * h * h
    return total


@pytest.mark.parametrize("nu", [4.0, 8.0, 30.0])
@pytest.mark.parametrize("eigs", [(0.02, 0.08), (0.6, 2.5), (40.0, 90.0)],
                         ids=["subpixel", "pixel", "large"])
def test_design_mass_numeric(eigs, nu):
    cov, nut = rot_cov(eigs, 0.4)[None], torch.tensor([nu])
    total = grid_integral(lambda u: design.kernel(u, torch.zeros(1, 2), cov, nut, S_MIP), eigs, nu)
    assert abs(total / float(design.mass(cov, nut)) - 1) < 1e-6


@pytest.mark.parametrize("nu", [4.0, 8.0])
@pytest.mark.parametrize("eigs", [(0.02, 0.08), (0.6, 2.5)], ids=["subpixel", "pixel"])
def test_option_b_residual_has_zero_mass(eigs, nu):
    """Option B alone (even where the design would take Option A) integrates to 2π|Σ|^½."""
    cov, nut = rot_cov(eigs, 0.4)[None], torch.tensor([nu])
    total = grid_integral(lambda u: option_b(u, torch.zeros(1, 2), cov, nut, S_MIP, design.QUAD),
                          eigs, nu)
    assert abs(total / float(mass_option_b(cov, nut, design.QUAD)) - 1) < 1e-6
