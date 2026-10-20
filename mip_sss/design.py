"""The Mip-SSS design chosen at the M1 gate (2026-10-20).

- Criterion: max alpha error ≤ 1e-3 against the exact filtered kernel, with exact limits
  (s = 0 gives SSS, ν → ∞ gives Mip-Splatting) and exact mass for the 2D filter.
- Option B: adaptive_tail grid (τ = 1/255, extent ±3 widths), K = 12 nodes, with the Option A
  control variate.
- Fast path: Option A wherever the ν-aware table (fastpath_table.py, ε = 1e-3) allows it. This
  covers ν ≥ 300 at every size, so there is no Gaussian fallback and ν always has a gradient.

These functions are the reference the CUDA kernels must match.
"""
import math

import torch

from .fastpath_table import TABLE
from .kernels import (det2, eig2_max, eig2_min, eye_like, option_a, option_a_3d, option_b,
                      option_b_3d, project, radius_option_b)
from .quadrature import QuadConfig

QUAD = QuadConfig("adaptive_tail", 12, left=3.0, right=3.0, tail_tau=1 / 255, control="A")
_NU = torch.tensor([nu for nu, _ in TABLE], dtype=torch.float64)
_RHO = torch.tensor([rho for _, rho in TABLE], dtype=torch.float64)


def fast_threshold(nu):
    """ρ(ν): Option A is used when λ_min/s ≥ ρ(ν). Step lookup on the largest table ν_i ≤ ν."""
    idx = torch.searchsorted(_NU.to(nu.device), nu.detach().double(), right=True) - 1
    return _RHO.to(nu.device)[idx.clamp(min=0)].to(nu.dtype)


def use_fast_path(cov, nu, s):
    """Per component: True where Option A is accurate enough. cov is the unfiltered 2D covariance."""
    if s == 0:
        return torch.ones_like(nu, dtype=torch.bool)  # Option A is exactly SSS at s = 0
    return eig2_min(cov) >= fast_threshold(nu) * s


def _merge(fast, n_pix, like, fast_fn, slow_fn):
    """Evaluate each branch on its own components only and scatter the rows back."""
    out = like.new_zeros(fast.shape[0], n_pix)
    for mask, fn in ((fast, fast_fn), (~fast, slow_fn)):
        idx = torch.nonzero(mask).squeeze(1)
        if idx.numel():
            out = out.index_put((idx,), fn(mask))
    return out


def kernel(u, mu, cov, nu, s):
    """The 2D-filtered kernel, shape (N, P); alpha = opacity[:, None] * kernel."""
    return _merge(use_fast_path(cov, nu, s), u.shape[0], cov,
                  lambda m: option_a(u, mu[m], cov[m], nu[m], s),
                  lambda m: option_b(u, mu[m], cov[m], nu[m], s, QUAD))


def mass(cov, nu):
    """∫ kernel du. Exact for both branches: the unfiltered SSS mass 2π|Σ|^½, for any s."""
    return 2 * math.pi * torch.sqrt(det2(cov))


def radius(cov, nu, opacity, s, alpha_min=1 / 255):
    """Radius beyond which |alpha| < alpha_min, for whichever branch the component takes."""
    fast = use_fast_path(cov, nu, s)
    covp = cov + s * eye_like(cov)
    peak = torch.sqrt(det2(cov) / det2(covp))
    ratio = (opacity.abs() * peak / alpha_min).clamp(min=1)
    out = torch.sqrt(nu * eig2_max(covp) * (ratio ** (2 / (nu + 2)) - 1))
    slow = torch.nonzero(~fast).squeeze(1)
    if slow.numel():
        out = out.index_put((slow,), radius_option_b(cov[slow], nu[slow], opacity[slow], s, QUAD,
                                                     alpha_min))
    return out


def kernel_3d(u, mu2, cov3, M, nu, sigma3_sq, s):
    """Both filters. Option A needs both the 3D and the projected 2D sizes past ρ(ν)."""
    cov2f = project(cov3 + sigma3_sq[:, None, None] * eye_like(cov3, 3), M)
    rho = fast_threshold(nu)
    lam3 = torch.linalg.eigvalsh(cov3)[:, 0]
    ok3 = (sigma3_sq == 0) | (lam3 >= rho * sigma3_sq)
    ok2 = torch.ones_like(ok3) if s == 0 else eig2_min(cov2f) >= rho * s
    return _merge(ok3 & ok2, u.shape[0], cov3,
                  lambda m: option_a_3d(u, mu2[m], cov3[m], M[m], nu[m], sigma3_sq[m], s),
                  lambda m: option_b_3d(u, mu2[m], cov3[m], M[m], nu[m], sigma3_sq[m], s, QUAD))
