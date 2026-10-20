"""Shared test cases and error metrics for the M1 tests and study scripts."""
import math

import numpy as np
import torch

from .kernels import eig2_max, eig2_min
from .reference import fft_convolution

NUS_README = (1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 30.0)
NUS_EXTENDED = NUS_README + (50.0, 100.0, 300.0, 1000.0, 1e4)
S_MIP = 0.1  # Mip-Splatting's 2D filter variance, px² (kernel_size)

# (eigenvalues in px², rotation): sub-pixel, mixed, pixel-sized, large, very large
COV_CASES = (
    ((0.01, 0.04), 0.3),
    ((0.05, 0.5), 1.0),
    ((0.3, 1.2), -0.7),
    ((1.0, 4.0), 0.2),
    ((10.0, 30.0), 2.1),
)
CORE, TAIL = 1e-3, 1e-6


def rot_cov(eigs, theta):
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    return R @ torch.diag(torch.tensor(eigs, dtype=torch.float64)) @ R.T


def ray_points(cov, nu, s, n_dirs=8, n_r=300, t_min=1e-7):
    """Points on rays from the origin, dense near the centre, out to where the kernel is ~t_min."""
    delta = nu * (t_min ** (-2 / (nu + 2)) - 1)
    r_max = math.sqrt(delta * float(eig2_max(cov))) + 8 * math.sqrt(s)
    r_min = 1e-3 * math.sqrt(float(eig2_min(cov)))
    r = torch.cat([torch.zeros(1, dtype=torch.float64),
                   torch.logspace(math.log10(r_min), math.log10(r_max), n_r, dtype=torch.float64)])
    th = np.linspace(0, math.pi, n_dirs, endpoint=False)
    dirs = torch.tensor(np.stack([np.cos(th), np.sin(th)], -1))
    return (r[:, None, None] * dirs[None]).reshape(-1, 2)


def fft_points(x, valid):
    """FFT-grid indices on the axes and diagonals through the centre, plus a coarse 2D subsample."""
    c0, n = len(x) // 2, len(x)
    sel = np.where(np.abs(x) <= valid)[0]
    idx = set()
    for i in sel:
        idx |= {(c0, i), (i, c0), (i, i), (i, n - 1 - i)}
    sub = sel[:: max(1, len(sel) // 60)]
    idx |= {(i, j) for i in sub for j in sub}
    ii = np.array(sorted(idx))
    return ii, torch.tensor(np.stack([x[ii[:, 1]], x[ii[:, 0]]], -1))


def fft_reference(eigs, theta, nu, s, kernel="t"):
    """Brute-force T ⊛ N(0, sI) by FFT for a component at the origin: (cov, points, values)."""
    cov = rot_cov(eigs, theta)
    shrink = nu / (nu + 2) if kernel == "t" else 1.0  # the t's core is narrower than Σ
    h = min(math.sqrt(min(eigs) * shrink), math.sqrt(s)) / 8
    x, val, valid = fft_convolution(cov.numpy(), nu, s, h, min(60.0, 15 * math.sqrt(max(eigs)) + 4),
                                    kernel)
    ii, u = fft_points(x, valid)
    return cov, u, torch.tensor(val[ii[:, 0], ii[:, 1]])


def errors(approx, ref, core=CORE, tail=TAIL):
    """Max relative error in the core (ref >= core), in the tail (tail <= ref < core), max abs error."""
    rel = (approx - ref).abs() / ref
    core_m = ref >= core
    tail_m = (ref >= tail) & ~core_m
    return {
        "core_rel": float(rel[core_m].max()) if core_m.any() else 0.0,
        "tail_rel": float(rel[tail_m].max()) if tail_m.any() else 0.0,
        "max_abs": float((approx - ref).abs().max()),
    }
