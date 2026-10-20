"""README test: negative opacity composites without NaNs and matches a scalar reference loop."""
import torch

from mip_sss import design
from mip_sss.render import composite
from mip_sss.testing import S_MIP


def scalar_composite(alpha, colors, background):
    """SSS's per-pixel loop written out one pixel and one component at a time."""
    n, p = alpha.shape
    out = torch.zeros(p, colors.shape[1])
    final_T = torch.zeros(p)
    for j in range(p):
        C, T = torch.zeros(colors.shape[1]), 1.0
        for i in range(n):
            a = float(alpha[i, j])
            a = min(0.99, a) if a >= 0 else max(-0.99, a)
            if -1 / 255 < a < 1 / 255:
                continue
            test_T = T * (1 - a)
            if test_T < 1e-4:
                break
            C = C + colors[i] * a * T
            T = test_T
        out[j] = C + T * background
        final_T[j] = T
    return out, final_T


def test_negative_opacity():
    g = torch.Generator().manual_seed(3)
    n = 16
    mu = torch.rand(n, 2, generator=g) * 8
    A = torch.randn(n, 2, 2, generator=g)
    cov = A @ A.transpose(-1, -2) * torch.exp(torch.randn(n, 1, 1, generator=g) * 2) + 0.01 * torch.eye(2)
    nu = 10 ** (torch.rand(n, generator=g) * 3)
    o = torch.rand(n, generator=g) * 2 - 1  # about half the components scoop
    colors = torch.rand(n, 3, generator=g)
    fast = design.use_fast_path(cov, nu, S_MIP)
    assert fast.any() and (~fast).any(), "the scene should exercise both branches"
    x = torch.arange(0, 8, 0.5)
    X, Y = torch.meshgrid(x, x, indexing="xy")
    u = torch.stack([X.ravel(), Y.ravel()], -1)
    alpha = o[:, None] * design.kernel(u, mu, cov, nu, S_MIP)
    bg = torch.tensor([0.2, 0.3, 0.4])
    color, T = composite(alpha, colors, bg)
    assert torch.isfinite(color).all() and torch.isfinite(T).all()
    assert (T > 1).any(), "negative components should raise transmittance somewhere"
    ref_color, ref_T = scalar_composite(alpha, colors, bg)
    assert torch.allclose(color, ref_color, rtol=0, atol=1e-12)
    assert torch.allclose(T, ref_T, rtol=0, atol=1e-12)
