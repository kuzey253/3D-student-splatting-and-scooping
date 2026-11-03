"""SSS relocation and its 3D-filtered variant (README M5.1).

`relocation_student_t` ports SSS's `compute_relocation_student_t` (diff-t-rasterization
cuda_rasterizer/utils.cu, Eq. 12 of the SSS paper) to torch, at any dtype, for tests and for
measuring how far relocation drifts from the filtered kernel. It keeps the kernel's arithmetic,
including scale_new = coeff² · scale_old.

With the 3D filter on, the rendered component has the effective opacity o · coef(s, σ3) (Option A
in 3D, `NTModel.get_opacity_with_3D_filter`). README M5.1 runs relocation on that opacity and writes
back the raw opacity that gives the relocated effective opacity under the new scale.
"""
import math

import torch

N_MAX = 51  # utils/reloc_utils.py


def relocation_student_t(opacity_old, scale_old, nu, N):
    """(opacity_new, scale_new) for components split into N copies, as the CUDA kernel computes.

    opacity_old: signed opacity (P,), scale_old: (P, 3), nu: (P,), N: copies per component (P,),
    clamped to [1, 51] as reloc_utils does.
    """
    N = N.clamp(1, N_MAX).long()
    o_new = 1 - torch.pow(1 - opacity_old, 1 / N.to(opacity_old.dtype))
    lg_half = math.lgamma(0.5)
    term1 = torch.exp(lg_half + torch.lgamma((nu + 2) / 2) - torch.lgamma((nu + 3) / 2))
    denom = torch.zeros_like(opacity_old)
    for i in range(1, int(N.max()) + 1):
        active = N >= i
        for k in range(i):
            term2 = (-1) ** k * torch.pow(o_new, k + 1)
            term3 = torch.exp(lg_half + torch.lgamma(((k + 1) * (nu + 3) - 1) / 2)
                              - torch.lgamma((k + 1) * (nu + 3) / 2))
            denom = denom + torch.where(active, math.comb(i - 1, k) * term2 * term3, 0.0)
    coeff = opacity_old * term1 / denom
    return o_new, coeff[:, None] ** 2 * scale_old


def filter_3d_coef(scale, sigma):
    """sqrt(|Σ| / |Σ + σ3²I|) per component, as NTModel.get_opacity_with_3D_filter. sigma: (P,)."""
    s2 = torch.square(scale)
    return torch.sqrt(s2.prod(dim=1) / (s2 + torch.square(sigma)[:, None]).prod(dim=1))


def effective_to_raw_opacity(opacity_eff, scale_new, sigma):
    """Raw opacity whose filtered value under scale_new is opacity_eff, clamped to |o| < 1.

    Returns (raw opacity with opacity_eff's shape, mask of components that hit the clamp).
    """
    coef = filter_3d_coef(scale_new, sigma).clamp_min(torch.finfo(scale_new.dtype).tiny)
    raw = opacity_eff / coef.reshape(opacity_eff.shape)
    lim = 1.0 - torch.finfo(opacity_eff.dtype).eps
    clamped = raw.abs() > lim
    return raw.clamp(-lim, lim), clamped.reshape(-1)


def relocate(opacity, scale, nu, N, sigma=None, reloc=relocation_student_t):
    """NTModel._update_params' opacity and scale rule; with sigma, on the effective opacity (M5.1).

    opacity is signed (o · negative). Returns (signed raw opacity (P, 1), raw scale (P, 3), clamped).
    """
    eps = torch.finfo(torch.float32).eps
    if sigma is not None:
        opacity = opacity * filter_3d_coef(scale, sigma)
    new_opacity, new_scaling = reloc(opacity, scale, nu, N)
    new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - eps, min=-1.0 + eps)
    new_opacity = torch.where((new_opacity >= 0) & (new_opacity < 0.005), 0.005, new_opacity)
    new_opacity = torch.where((new_opacity < 0) & (new_opacity > -0.005), -0.005, new_opacity)
    new_scaling = new_scaling.reshape(-1, 3)
    clamped = torch.zeros(new_opacity.shape[0], dtype=torch.bool, device=new_opacity.device)
    if sigma is not None:
        new_opacity, clamped = effective_to_raw_opacity(new_opacity, new_scaling, sigma)
    return new_opacity, new_scaling, clamped
