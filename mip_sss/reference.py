"""Brute-force references for M1, by two independent routes.

1. lambda_integral: the scale-mixture integral over λ, on a very fine trapezoid grid in z = log λ.
   The integrand is smooth and decays fast at both ends, so the trapezoid rule is spectrally
   accurate. This route shares the scale-mixture derivation with Option B.
2. fft_convolution: T ⊛ N(0, sI) sampled on a fine Cartesian grid and convolved by FFT.
   This route does not use the mixture at all.

The tests require the two to agree before either is used as ground truth.
"""
import math

import numpy as np
import torch
from scipy.signal import fftconvolve
from scipy.special import digamma, gammaln, polygamma

from .kernels import det2, eye_like, maha2, project


def z_grid(nu, d=2, per_width=40, left=30.0, right=20.0, tau=1e-14):
    """Fine z grid and log trapezoid weights (incl. dz) for the normalized log-Gamma weight."""
    a, b = (nu + d) / 2, nu / 2
    c, w = digamma(a) - math.log(b), math.sqrt(polygamma(1, a))
    lo, hi = c - left * w + math.log(tau) / a, c + right * w
    n = int(math.ceil((hi - lo) * per_width / w)) + 1
    z = np.linspace(lo, hi, n)
    log_w = a * z - b * np.exp(z) - (gammaln(a) - a * math.log(b)) + math.log(z[1] - z[0])
    return torch.tensor(z), torch.tensor(log_w)


def _integrate(d, log_w, log_peak, filt_cov, chunk):
    out = []
    for dc in torch.split(d, chunk):
        q = maha2(dc[None], filt_cov[:, None])
        out.append(torch.exp(torch.logsumexp((log_w + log_peak)[:, None] - 0.5 * q, dim=0)))
    return torch.cat(out)


def lambda_integral(u, mu, cov, nu, s, chunk=1024, **grid):
    """Exact 2D-filtered SSS kernel, shape (N, P)."""
    out = []
    for i in range(mu.shape[0]):
        z, log_w = z_grid(float(nu[i]), **grid)
        lam = torch.exp(z).to(cov.dtype)
        node = cov[i] / lam[:, None, None]
        filt = node + s * eye_like(cov)
        log_peak = 0.5 * (torch.log(det2(node)) - torch.log(det2(filt)))
        out.append(_integrate(u - mu[i], log_w, log_peak, filt, chunk))
    return torch.stack(out)


def lambda_integral_3d(u, mu2, cov3, M, nu, sigma3_sq, s, chunk=1024, **grid):
    """Exact kernel with the per-node 3D filter, projection and 2D filter, shape (N, P)."""
    out = []
    for i in range(mu2.shape[0]):
        z, log_w = z_grid(float(nu[i]), **grid)
        lam = torch.exp(z).to(cov3.dtype)
        node3 = cov3[i] / lam[:, None, None]
        filt3 = node3 + sigma3_sq[i] * eye_like(cov3, 3)
        cov2 = project(filt3, M[i])
        filt2 = cov2 + s * eye_like(cov2)
        log_peak = 0.5 * (torch.logdet(node3) - torch.logdet(filt3)
                          + torch.log(det2(cov2)) - torch.log(det2(filt2)))
        out.append(_integrate(u - mu2[i], log_w, log_peak, filt2, chunk))
    return torch.stack(out)


def fft_convolution(cov, nu, s, h, half_width, kernel="t"):
    """T ⊛ N(0, sI) for one component at the origin, by FFT on a grid of spacing h.

    cov is a 2x2 numpy array; kernel is "t" (SSS, peak 1) or "gaussian" (peak 1, for validation).
    Returns (x, values, valid): values[i, j] is the result at (x[j], x[i]), and points with
    max(|x|, |y|) <= valid are unaffected by the grid boundary.
    """
    n = 2 * int(round(half_width / h)) + 1
    x = (np.arange(n) - n // 2) * h
    X, Y = np.meshgrid(x, x)
    a, b, e = cov[0, 0], cov[0, 1], cov[1, 1]
    q = (e * X * X - 2 * b * X * Y + a * Y * Y) / (a * e - b * b)
    T = np.exp(-(nu + 2) / 2 * np.log1p(q / nu)) if kernel == "t" else np.exp(-0.5 * q)
    kr = int(math.ceil(9 * math.sqrt(s) / h))
    xk = np.arange(-kr, kr + 1) * h
    gk = np.exp(-xk ** 2 / (2 * s)) / math.sqrt(2 * math.pi * s) * h
    # the Gaussian is separable: convolve rows, then columns
    out = fftconvolve(fftconvolve(T, gk[None, :], mode="same"), gk[:, None], mode="same")
    return x, out, x[-1] - kr * h
