"""Front-to-back compositing as in SSS's forward kernel (diff-t-rasterization forward.cu:416-437).

Alpha may be negative (scooping). It is clamped to ±0.99, skipped when |alpha| < 1/255, and the
pixel stops once the transmittance would fall below 1e-4. Negative alpha raises T above 1.
"""
import torch


def composite(alpha, colors, background=None, clamp=0.99, alpha_min=1 / 255, t_min=1e-4):
    """alpha (N, P) sorted front to back, colors (N, C). Returns color (P, C) and final T (P,)."""
    n, p = alpha.shape
    color = torch.zeros(p, colors.shape[1], dtype=alpha.dtype, device=alpha.device)
    T = torch.ones(p, dtype=alpha.dtype, device=alpha.device)
    done = torch.zeros(p, dtype=torch.bool, device=alpha.device)
    for i in range(n):
        a = alpha[i].clamp(-clamp, clamp)
        skip = done | ((a > -alpha_min) & (a < alpha_min))
        test_T = T * (1 - a)
        stop = ~skip & (test_T < t_min)
        done = done | stop
        use = ~skip & ~stop
        color = color + torch.where(use[:, None], colors[i][None, :] * (a * T)[:, None], 0.0)
        T = torch.where(use, test_T, T)
    if background is not None:
        color = color + T[:, None] * background[None, :]
    return color, T
