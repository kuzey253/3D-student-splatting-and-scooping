"""Quadrature rules for the Student-t scale mixture behind Option B.

SSS's peak-1 kernel in d dimensions is a mixture of peak-1 Gaussians:

    T(u) = E_{λ ~ Gamma((ν+d)/2, rate ν/2)} [ exp(-½ λ dᵀΣ⁻¹d) ].

In z = log λ the node weights therefore follow the log-Gamma density

    w(z) ∝ exp(a z − b e^z),   a = (ν+d)/2,   b = ν/2,

which is the README's π_k λ_k^(d/2) with π_k = GammaPDF(λ_k; ν/2, ν/2) · λ_k · Δz once the
ν-only factors cancel in the normalization.

Every rule returns nodes z and normalized log-weights, both of shape (N, K).
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy.special import digamma, polygamma

RULES = ("fixed", "adaptive", "adaptive_tail", "gauss")
CLOSED_FORMS = ("none", "gaussian", "A", "A_prime")


@dataclass(frozen=True)
class QuadConfig:
    rule: str = "fixed"
    K: int = 8
    # fixed: shared grid in z
    z_min: float = -12.0
    z_max: float = 3.0
    # adaptive rules: grid extent left/right of the centre, in units of the log-Gamma width.
    # gauss: with gauss_tail_nodes > 0, the bulk/tail split sits at centre − left·width.
    left: float = 6.0
    right: float = 3.0
    # adaptive_tail: smallest unfiltered kernel value the left end of the grid must resolve
    tail_tau: float = 1e-4
    # gauss: nodes given to a separate Gauss rule on the left tail (0 = one Gauss rule)
    gauss_tail_nodes: int = 0
    # Above nu_fallback the closed form `fallback` replaces the quadrature
    nu_fallback: float = math.inf
    fallback: str = "gaussian"
    # Control variate: evaluate this closed form exactly, use the quadrature only for the residual
    control: str = "none"

    def __post_init__(self):
        assert self.rule in RULES, self.rule
        assert self.fallback in CLOSED_FORMS[1:], self.fallback
        assert self.control in ("none", "A", "A_prime"), self.control
        assert self.K >= 1 and 0 <= self.gauss_tail_nodes < self.K


def log_density(z, nu, d=2):
    """Unnormalized log-Gamma((ν+d)/2, ν/2) log-density in z = log λ."""
    return (nu + d) / 2 * z - nu / 2 * torch.exp(z)


def centre_width(nu, shape):
    """Mean and std of log λ for λ ~ Gamma(shape, rate ν/2)."""
    return torch.digamma(shape) - torch.log(nu / 2), torch.sqrt(torch.polygamma(1, shape))


def nodes(nu, cfg, d=2):
    """Nodes z (N, K) and normalized log-weights (N, K) for components with tail parameter nu (N,)."""
    nu = nu.reshape(-1, 1)
    if cfg.rule == "gauss":
        return _gauss_nodes(nu, cfg, d)
    t = torch.linspace(0, 1, cfg.K, dtype=nu.dtype, device=nu.device)
    if cfg.rule == "fixed":
        z = (cfg.z_min + (cfg.z_max - cfg.z_min) * t).expand(nu.shape[0], -1)
    elif cfg.rule == "adaptive":
        # README: centre and width from Gamma(ν/2, ν/2), the unweighted mixing distribution
        c, w = centre_width(nu, nu / 2)
        z = c + w * (-cfg.left + (cfg.left + cfg.right) * t)
    else:
        # centre and width from the weight distribution itself, extended left far enough that
        # the unfiltered kernel is resolved down to tail_tau (its integrand shifts by -log(τ)/a)
        a = (nu + d) / 2
        c, w = centre_width(nu, a)
        lo = c - cfg.left * w + math.log(cfg.tail_tau) / a
        hi = c + cfg.right * w
        z = lo + (hi - lo) * t
    logw = log_density(z, nu, d)
    return z, logw - torch.logsumexp(logw, dim=-1, keepdim=True)


def _gauss_nodes(nu, cfg, d):
    rules = [_gauss_rule(float(v), cfg.K, cfg.gauss_tail_nodes, cfg.left, d)
             for v in nu.detach().cpu().double().flatten()]
    z = torch.tensor(np.stack([r[0] for r in rules]), dtype=nu.dtype, device=nu.device)
    logw = torch.tensor(np.stack([r[1] for r in rules]), dtype=nu.dtype, device=nu.device)
    return z, logw


@lru_cache(maxsize=4096)
def _gauss_rule(nu, K, tail_nodes, left, d):
    """Gauss rule for the log-Gamma weight in z.

    With tail_nodes > 0 the rule is composite: Gauss on the bulk [c − left·w, ∞) and a separate
    Gauss rule on the whole left tail (−∞, c − left·w). The tail rule puts nodes where the
    kernel's far field comes from, and both parts stay exact for low-order moments, so mass is
    kept. Nodes are computed numerically (no gradient through ν); this rule is for the M1 study.
    """
    a, b = (nu + d) / 2, nu / 2
    c, w = digamma(a) - math.log(b), math.sqrt(polygamma(1, a))
    log_norm = math.lgamma(a) - a * math.log(b)
    far_left = c - 60 * w + math.log(1e-30) / a
    split = c - left * w if tail_nodes else far_left
    parts = [(split, c + 40 * w, K - tail_nodes)]
    if tail_nodes:
        parts.append((far_left, split, tail_nodes))
    zs, ws = [], []
    for lo, hi, n in parts:
        zz = np.linspace(lo, hi, 40001)
        dens = np.exp(a * zz - b * np.exp(zz) - log_norm) * (zz[1] - zz[0])
        zg, wg = _gauss_from_discrete(zz, dens, n)
        zs.append(zg)
        ws.append(wg)
    z, wt = np.concatenate(zs[::-1]), np.concatenate(ws[::-1])
    return z, np.log(wt) - np.log(wt.sum())


def _gauss_from_discrete(x, wts, n):
    """n-point Gauss rule for a discrete measure, by Lanczos with full reorthogonalization."""
    scale = x.std() or 1.0
    xs = (x - x.mean()) / scale
    v = np.sqrt(wts / wts.sum())
    basis, alpha, beta = [v], [], []
    for k in range(n):
        r = xs * basis[-1]
        alpha.append(r @ basis[-1])
        for q in basis:
            r -= (r @ q) * q
        for q in basis:
            r -= (r @ q) * q
        if k < n - 1:
            beta.append(np.linalg.norm(r))
            basis.append(r / beta[-1])
    jac = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    evals, evecs = np.linalg.eigh(jac)
    return evals * scale + x.mean(), wts.sum() * evecs[0] ** 2
