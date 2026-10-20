"""Pure-PyTorch reference kernels for Mip-SSS (milestone M1).

Shapes: N components, P pixels. mu (N, 2), cov (N, 2, 2) SPD, nu (N,), u (P, 2).
Each kernel returns the peak-normalized value of shape (N, P); alpha = opacity[:, None] * kernel.
Use float64. These functions are the ground truth for the CUDA kernels.
"""
import math

import torch

from .quadrature import QuadConfig, nodes


def det2(c):
    return c[..., 0, 0] * c[..., 1, 1] - c[..., 0, 1] * c[..., 1, 0]


def maha2(d, c):
    """dᵀ c⁻¹ d for 2-vectors d and 2x2 SPD matrices c (broadcast over leading dims)."""
    a, b, e = c[..., 0, 0], c[..., 0, 1], c[..., 1, 1]
    x, y = d[..., 0], d[..., 1]
    return (e * x * x - 2 * b * x * y + a * y * y) / (a * e - b * b)


def eye_like(c, n=2):
    return torch.eye(n, dtype=c.dtype, device=c.device)


def offsets(u, mu):
    return u[None, :, :] - mu[:, None, :]


def t_kernel(d, cov, nu):
    """SSS's 2D kernel [1 + dᵀΣ⁻¹d/ν]^(-(ν+2)/2) for offsets d (N, P, 2)."""
    nu = nu[:, None]
    return torch.exp(-(nu + 2) / 2 * torch.log1p(maha2(d, cov[:, None]) / nu))


def sss_kernel(u, mu, cov, nu):
    return t_kernel(offsets(u, mu), cov, nu)


def gauss_mip(u, mu, cov, s):
    """Mip-Splatting's 2D-filtered Gaussian (peak-1 convention)."""
    covs = cov + s * eye_like(cov)
    peak = torch.sqrt(det2(cov) / det2(covs))
    return peak[:, None] * torch.exp(-0.5 * maha2(offsets(u, mu), covs[:, None]))


def shortcut_cov(cov, nu, s, curvature_matched):
    """Σ' for Option A (Σ + sI) or Option A' (Σ + s(ν+2)/ν I, matches the t's core curvature)."""
    mult = (nu + 2) / nu if curvature_matched else torch.ones_like(nu)
    return cov + (mult * s)[:, None, None] * eye_like(cov)


def option_a(u, mu, cov, nu, s, curvature_matched=False):
    """A single t on Σ' with opacity scaled by sqrt(|Σ|/|Σ'|). Mass-preserving, tails too heavy."""
    covp = shortcut_cov(cov, nu, s, curvature_matched)
    peak = torch.sqrt(det2(cov) / det2(covp))
    return peak[:, None] * sss_kernel(u, mu, covp, nu)


def closed_form(name, u, mu, cov, nu, s):
    if name == "gaussian":
        return gauss_mip(u, mu, cov, s)
    return option_a(u, mu, cov, nu, s, curvature_matched=name == "A_prime")


def option_b(u, mu, cov, nu, s, cfg=QuadConfig()):
    """Scale-mixture quadrature: Σ_k ω_k g_k(u), each g_k an exactly filtered Gaussian."""
    z, logw = nodes(nu, cfg)
    lam = torch.exp(z)
    d = offsets(u, mu)
    node_cov = cov[:, None] / lam[..., None, None]
    filt_cov = node_cov + s * eye_like(cov)
    log_peak = 0.5 * (torch.log(det2(node_cov)) - torch.log(det2(filt_cov)))
    q = maha2(d[:, None], filt_cov[:, :, None])
    log_g = log_peak[..., None] - 0.5 * q
    if cfg.control == "none":
        out = torch.exp(torch.logsumexp(logw[..., None] + log_g, dim=1))
    else:
        # h_k integrates exactly (over λ) to the control closed form and has the same mass as g_k,
        # so the residual Σ ω_k (g_k − h_k) carries zero mass and vanishes at s = 0 and λ = 1.
        covp = shortcut_cov(cov, nu, s, cfg.control == "A_prime")
        h_log_peak = 0.5 * (torch.log(det2(cov)) - torch.log(det2(covp)))
        qh = lam[..., None] * maha2(d, covp[:, None])[:, None]
        h = torch.exp(h_log_peak[:, None, None] - 0.5 * qh)
        resid = (torch.exp(logw)[..., None] * (torch.exp(log_g) - h)).sum(1)
        out = closed_form(cfg.control, u, mu, cov, nu, s) + resid
    if math.isfinite(cfg.nu_fallback):
        far = (nu > cfg.nu_fallback)[:, None]
        if far.any():
            out = torch.where(far, closed_form(cfg.fallback, u, mu, cov, nu, s), out)
    return out


def eig2_min(cov):
    a, b, e = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    mid = 0.5 * (a + e)
    return mid - torch.sqrt(torch.clamp(mid * mid - (a * e - b * b), min=0))


def eig2_max(cov):
    a, b, e = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    mid = 0.5 * (a + e)
    return mid + torch.sqrt(torch.clamp(mid * mid - (a * e - b * b), min=0))


def mip_sss(u, mu, cov, nu, s, cfg=QuadConfig(), fast_path="none", threshold=10.0):
    """Hybrid: closed-form fast path where the smallest eigenvalue exceeds threshold·s, else Option B.

    threshold is a float or a callable nu -> per-component multiplier of s.
    """
    out = option_b(u, mu, cov, nu, s, cfg)
    if fast_path != "none":
        thr = threshold(nu) if callable(threshold) else threshold
        fast = (eig2_min(cov) > thr * s)[:, None]
        if fast.any():
            out = torch.where(fast, closed_form(fast_path, u, mu, cov, nu, s), out)
    return out


_SSS_QUANTILES = (63.657, 9.925, 5.841, 4.604, 4.032, 3.707, 3.499, 3.355, 3.250, 3.169, 3.106,
                  3.055, 3.012)


def sss_radius(cov, nu):
    """SSS's culling radius in pixels, ported from forward.cu:235-286 as written.

    Within [n, n+1) the code interpolates from the df=n+1 quantile up to the df=n quantile, i.e.
    in reverse, so the radius jumps at every integer ν; above ν = 13 it is 3σ.
    """
    a, b, e = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    mid = 0.5 * (a + e)
    lam1 = mid + torch.sqrt(torch.clamp(mid * mid - (a * e - b * b), min=0.1))
    vv = torch.floor(nu).long()
    q = torch.tensor(_SSS_QUANTILES, dtype=cov.dtype, device=cov.device)
    inside = (vv >= 1) & (vv <= 12)
    idx = vv.clamp(1, 12) - 1
    interp = (q[idx] - q[idx + 1]) * (nu - vv) + q[idx + 1]
    empirical = torch.where(inside, interp, torch.full_like(nu, 3.0))
    return torch.ceil(empirical * torch.sqrt(lam1))


def mass_option_b(cov, nu, cfg=QuadConfig()):
    """Exact ∫ option_b du (independent of s). The unfiltered t integrates to 2π|Σ|^½."""
    if cfg.control != "none":
        return 2 * math.pi * torch.sqrt(det2(cov))
    z, logw = nodes(nu, cfg)
    return 2 * math.pi * torch.sqrt(det2(cov)) * torch.exp(torch.logsumexp(logw - z, dim=-1))


def radius_option_b(cov, nu, opacity, s, cfg=QuadConfig(), alpha_min=1 / 255):
    """A screen radius beyond which |alpha| < alpha_min, from per-term bounds.

    The kernel is bounded by a sum of terms amp_j · f_j(r); requiring each term to stay below
    alpha_min / n_terms gives a conservative radius. Gaussian terms use their largest eigenvalue;
    the control's t term is solved in closed form.
    """
    z, logw = nodes(nu, cfg)
    lam = torch.exp(z)
    node_cov = cov[:, None] / lam[..., None, None]
    filt_cov = node_cov + s * eye_like(cov)
    peak = torch.sqrt(det2(node_cov) / det2(filt_cov))
    amps, variances = [torch.exp(logw) * peak], [eig2_max(filt_cov)]
    t_term = None
    if cfg.control != "none":
        covp = shortcut_cov(cov, nu, s, cfg.control == "A_prime")
        h_peak = torch.sqrt(det2(cov) / det2(covp))
        amps.append(torch.exp(logw) * h_peak[:, None])
        variances.append(eig2_max(covp)[:, None] / lam)
        t_term = (h_peak, eig2_max(covp))
    amp = opacity.abs()[:, None] * torch.cat(amps, dim=1)
    var = torch.cat(variances, dim=1)
    n_terms = amp.shape[1] + (t_term is not None)
    ratio = amp * n_terms / alpha_min
    r2 = torch.where(ratio > 1, 2 * var * torch.log(ratio.clamp(min=1)), torch.zeros_like(var))
    r2 = r2.max(dim=1).values
    if t_term is not None:
        h_peak, e = t_term
        ratio = opacity.abs() * h_peak * n_terms / alpha_min
        r2_t = nu * e * (ratio.clamp(min=1) ** (2 / (nu + 2)) - 1)
        r2 = torch.maximum(r2, r2_t)
    return torch.sqrt(r2)


# ---- 3D smoothing filter + 2D filter -------------------------------------------------------
# M (N, 2, 3) is the per-component linear map from 3D to screen space, (J W)[:2, :] in 3DGS terms,
# so the projected covariance of a 3D covariance C is M C Mᵀ.


def project(cov3, M):
    return M @ cov3 @ M.transpose(-1, -2)


def mip_gauss_3d(u, mu2, cov3, M, sigma3_sq, s):
    """Mip-Splatting with both filters on a Gaussian."""
    cov3f = cov3 + sigma3_sq[:, None, None] * eye_like(cov3, 3)
    c3 = torch.sqrt(torch.det(cov3) / torch.det(cov3f))
    return c3[:, None] * gauss_mip(u, mu2, project(cov3f, M), s)


def option_a_3d(u, mu2, cov3, M, nu, sigma3_sq, s, curvature_matched=False):
    cov3f = cov3 + sigma3_sq[:, None, None] * eye_like(cov3, 3)
    c3 = torch.sqrt(torch.det(cov3) / torch.det(cov3f))
    return c3[:, None] * option_a(u, mu2, project(cov3f, M), nu, s, curvature_matched)


def option_b_3d(u, mu2, cov3, M, nu, sigma3_sq, s, cfg=QuadConfig()):
    """Per node: 3D filter (with its det-ratio factor), projection, then the 2D filter.

    Weights use d = 2, which reproduces SSS's 2D kernel when both filters are off. With a control
    variate, the closed form is option_a_3d and h_k is its node-k Gaussian, so the residual
    vanishes at σ3 = s = 0 and at λ = 1 (both limits stay exact).
    """
    z, logw = nodes(nu, cfg)
    lam = torch.exp(z)
    node3 = cov3[:, None] / lam[..., None, None]
    filt3 = node3 + sigma3_sq[:, None, None, None] * eye_like(cov3, 3)
    log_c3 = 0.5 * (torch.logdet(node3) - torch.logdet(filt3))
    cov2 = project(filt3, M[:, None])
    filt2 = cov2 + s * eye_like(cov2)
    log_c2 = 0.5 * (torch.log(det2(cov2)) - torch.log(det2(filt2)))
    d = offsets(u, mu2)
    q = maha2(d[:, None], filt2[:, :, None])
    log_g = (log_c3 + log_c2)[..., None] - 0.5 * q
    if cfg.control == "none":
        out = torch.exp(torch.logsumexp(logw[..., None] + log_g, dim=1))
    else:
        cm = cfg.control == "A_prime"
        cov3f = cov3 + sigma3_sq[:, None, None] * eye_like(cov3, 3)
        cov2f = project(cov3f, M)
        covp = shortcut_cov(cov2f, nu, s, cm)
        h_log_peak = 0.5 * (torch.logdet(cov3) - torch.logdet(cov3f)
                            + torch.log(det2(cov2f)) - torch.log(det2(covp)))
        h = torch.exp(h_log_peak[:, None, None] - 0.5 * lam[..., None] * maha2(d, covp[:, None])[:, None])
        resid = (torch.exp(logw)[..., None] * (torch.exp(log_g) - h)).sum(1)
        out = option_a_3d(u, mu2, cov3, M, nu, sigma3_sq, s, cm) + resid
    if math.isfinite(cfg.nu_fallback):
        far = (nu > cfg.nu_fallback)[:, None]
        if far.any():
            fb = (mip_gauss_3d(u, mu2, cov3, M, sigma3_sq, s) if cfg.fallback == "gaussian" else
                  option_a_3d(u, mu2, cov3, M, nu, sigma3_sq, s, cfg.fallback == "A_prime"))
            out = torch.where(far, fb, out)
    return out
