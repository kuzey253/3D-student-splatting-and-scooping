"""Float64 PyTorch re-implementation of SSS's CUDA rasterizer, for CUDA parity tests (M2+).

Follows diff-t-rasterization step by step: frustum test (z > 0.2), projection with the 3DGS
view/projection matrices, EWA 2D covariance with the 1.3·tan(fov) clamp, the Mip-SSS 2D filter
(Option A) when kernel_size > 0, SSS's culling radius and 16x16 tile rectangles, the t kernel
evaluated at integer pixel coordinates, and front-to-back compositing (render.composite).

Inputs are the rasterizer's own: means3D (N,3), cov3D (N,3,3) (the precomputed-covariance path),
colors (N,3), opacity (N,) already activated (signed), nu (N,), and the camera's
world_view_transform / full_proj_transform exactly as SSS stores them (transposed, row-major).
"""
import math

import torch

from .kernels import det2, sss_radius
from .render import composite

BLOCK = 16


def _ewa_cov2d(t, cov3, R, fx, fy, tan_fovx, tan_fovy):
    """forward.cu computeCov2D: J R Σ Rᵀ Jᵀ with the screen-space clamp on t."""
    limx, limy = 1.3 * tan_fovx, 1.3 * tan_fovy
    tz = t[:, 2]
    tx = torch.clamp(t[:, 0] / tz, -limx, limx) * tz
    ty = torch.clamp(t[:, 1] / tz, -limy, limy) * tz
    zero = torch.zeros_like(tz)
    J = torch.stack([torch.stack([fx / tz, zero, -fx * tx / tz ** 2], -1),
                     torch.stack([zero, fy / tz, -fy * ty / tz ** 2], -1)], 1)
    T = J @ R
    return T @ cov3 @ T.transpose(1, 2)


class _ConicGradDamping(torch.autograd.Function):
    """Identity in the forward pass; scales the incoming gradient by D²/(D² + 1e-7).

    backward.cu turns conic gradients into 2D-covariance gradients with denom2inv =
    1/(D² + 1e-7) instead of 1/D² (D = det Σ'). Harmless with 3DGS's 0.3 px² dilation, but SSS
    removed it, so the gradients of sub-pixel components (D² ≲ 1e-7) are damped.
    """

    @staticmethod
    def forward(ctx, cov, scale):
        ctx.save_for_backward(scale)
        return cov.clone()

    @staticmethod
    def backward(ctx, grad):
        (scale,) = ctx.saved_tensors
        return grad * scale[:, None, None], None


def rasterize(means3D, cov3D, colors, opacity, nu, world_view, full_proj, tan_fovx, tan_fovy,
              H, W, bg, kernel_size=0.0, sss_nu_grad=True, sss_conic_grad=True):
    """Image (3, H, W) as SSS's CUDA renders it.

    sss_nu_grad=True reproduces SSS's ν gradient as written in backward.cu (which omits the
    −½·ln(1 + b/ν)·G term); sss_conic_grad=True reproduces its 1e-7 damping of conic gradients
    (see _ConicGradDamping). Both make the gradients comparable with the CUDA kernel as written.
    """
    dt = means3D.dtype
    n = means3D.shape[0]
    ones = torch.ones(n, 1, dtype=dt, device=means3D.device)
    ph = torch.cat([means3D, ones], 1)
    p_view = ph @ world_view[:, :3]
    p_hom = ph @ full_proj
    p_proj = p_hom[:, :3] / (p_hom[:, 3:4] + 1e-7)
    visible = p_view[:, 2] > 0.2

    fx, fy = W / (2 * tan_fovx), H / (2 * tan_fovy)
    R = world_view[:3, :3].T  # world-to-camera rotation: R[i][j] = view[4j + i]
    cov = _ewa_cov2d(p_view, cov3D, R, fx, fy, tan_fovx, tan_fovy)
    coef = torch.ones(n, dtype=dt, device=means3D.device)
    if kernel_size > 0:
        det0 = det2(cov)
        cov = cov + kernel_size * torch.eye(2, dtype=dt, device=means3D.device)
        coef = torch.where(det0 > 0, torch.sqrt(det0.clamp(min=0) / det2(cov)), torch.zeros_like(det0))
    det = det2(cov)
    cov_c = cov
    if sss_conic_grad:
        d2 = det.detach() ** 2
        cov_c = _ConicGradDamping.apply(cov, d2 / (d2 + 1e-7))
    conic = torch.stack([cov_c[:, 1, 1], -cov_c[:, 0, 1], cov_c[:, 0, 0]], 1) / det2(cov_c)[:, None]

    radius = sss_radius(cov.detach(), nu.detach())
    xy = torch.stack([((p_proj[:, 0] + 1) * W - 1) * 0.5, ((p_proj[:, 1] + 1) * H - 1) * 0.5], 1)
    gx, gy = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK
    xyd = xy.detach()
    rmin_x = ((xyd[:, 0] - radius) / BLOCK).trunc().clamp(0, gx)
    rmax_x = ((xyd[:, 0] + radius + BLOCK - 1) / BLOCK).trunc().clamp(0, gx)
    rmin_y = ((xyd[:, 1] - radius) / BLOCK).trunc().clamp(0, gy)
    rmax_y = ((xyd[:, 1] + radius + BLOCK - 1) / BLOCK).trunc().clamp(0, gy)
    active = visible & (det != 0) & ((rmax_x - rmin_x) * (rmax_y - rmin_y) > 0)

    py, px = torch.meshgrid(torch.arange(H, dtype=dt, device=means3D.device),
                            torch.arange(W, dtype=dt, device=means3D.device), indexing="ij")
    px, py = px.reshape(-1), py.reshape(-1)
    tx, ty = (px // BLOCK)[None], (py // BLOCK)[None]
    in_rect = ((tx >= rmin_x[:, None]) & (tx < rmax_x[:, None])
               & (ty >= rmin_y[:, None]) & (ty < rmax_y[:, None]))

    d_x = xy[:, 0:1] - px[None]
    d_y = xy[:, 1:2] - py[None]
    base = conic[:, 0:1] * d_x * d_x + conic[:, 2:3] * d_y * d_y + 2 * conic[:, 1:2] * d_x * d_y
    nu_c = nu[:, None]
    nu_pow = nu_c.detach() if sss_nu_grad else nu_c
    G = torch.exp(-(nu_pow + 2) / 2 * torch.log1p(base.clamp(min=0) / nu_c))
    alpha = (opacity * coef)[:, None] * G
    keep = active[:, None] & in_rect & (base > 0) & (nu_c >= 1)
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))

    order = torch.argsort(p_view[:, 2].detach())
    color, _ = composite(alpha[order], colors[order], bg)
    return color.T.reshape(3, H, W)


def _projection(p_view, R, fx, fy, tan_fovx, tan_fovy):
    """T = J W (N, 2, 3), with computeCov2D's screen-space clamp (mip_optionb.h optb_projection)."""
    limx, limy = 1.3 * tan_fovx, 1.3 * tan_fovy
    tz = p_view[:, 2]
    tx = torch.clamp(p_view[:, 0] / tz, -limx, limx) * tz
    ty = torch.clamp(p_view[:, 1] / tz, -limy, limy) * tz
    zero = torch.zeros_like(tz)
    J = torch.stack([torch.stack([fx / tz, zero, -fx * tx / tz ** 2], -1),
                     torch.stack([zero, fy / tz, -fy * ty / tz ** 2], -1)], 1)
    return J @ R


def optb_radius_bound(pre, nu, opacity, s, alpha_min=1 / 255):
    """The alpha-safe radius bound of an Option B component, as mip_optionb.h optb_fill."""
    from .optionb_staged import K
    hp = float(torch.exp(pre["loghp"]))
    absop, nterms = abs(float(opacity)), 2 * K + 1
    lmax_covp = float(torch.linalg.eigvalsh(pre["covp"])[-1])
    ratio_t = max(1.0, absop * hp * nterms / alpha_min)
    r2 = float(nu) * lmax_covp * (ratio_t ** (2 / (float(nu) + 2)) - 1)
    for nd in pre["nodes"]:
        rg = absop * float(torch.exp(nd["logg0"])) * nterms / alpha_min
        if rg > 1:
            r2 = max(r2, 2 * (1 / float(torch.linalg.eigvalsh(nd["Q"])[0])) * math.log(rg))
        rh = absop * float(nd["w"]) * hp * nterms / alpha_min
        if rh > 1:
            r2 = max(r2, 2 * lmax_covp / float(nd["lam"]) * math.log(rh))
    return math.sqrt(max(r2, 0.0))


def rasterize_option_b(means3D, cov3D, colors, opacity, nu, sigma3, world_view, full_proj,
                       tan_fovx, tan_fovy, H, W, bg, kernel_size):
    """Image (3, H, W) as the CUDA rasterizer renders it in Option B mode (M4).

    Every component: covp = Σ2 + P + sI with peak hp (the 3D + 2D Option A term). Fast-path
    components render hp·t(covp) through SSS's own kernel and backward, so their ν gradient and
    conic damping follow SSS (sss_nu_grad, _ConicGradDamping). Option B components render
    mip_sss.optionb_staged.direct, with exact gradients.
    """
    from . import optionb_staged as ob
    dt = means3D.dtype
    n = means3D.shape[0]
    ph = torch.cat([means3D, torch.ones(n, 1, dtype=dt, device=means3D.device)], 1)
    p_view = ph @ world_view[:, :3]
    p_hom = ph @ full_proj
    p_proj = p_hom[:, :3] / (p_hom[:, 3:4] + 1e-7)
    visible = p_view[:, 2] > 0.2
    fx, fy = W / (2 * tan_fovx), H / (2 * tan_fovy)
    R = world_view[:3, :3].T
    Tproj = _projection(p_view, R, fx, fy, tan_fovx, tan_fovy)
    xy = torch.stack([((p_proj[:, 0] + 1) * W - 1) * 0.5, ((p_proj[:, 1] + 1) * H - 1) * 0.5], 1)

    py, px = torch.meshgrid(torch.arange(H, dtype=dt, device=means3D.device),
                            torch.arange(W, dtype=dt, device=means3D.device), indexing="ij")
    pix = torch.stack([px.reshape(-1), py.reshape(-1)], 1)
    gx, gy = (W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK
    I2 = torch.eye(2, dtype=dt, device=means3D.device)
    alphas, fast_flags = [], []
    for i in range(n):
        zero = torch.zeros(H * W, dtype=dt, device=means3D.device)
        if not bool(visible[i]):
            alphas.append(zero)
            fast_flags.append(True)
            continue
        S3, Ti, nui, sig = cov3D[i], Tproj[i], nu[i], float(sigma3[i])
        pre = ob.preprocess(S3.detach(), torch.tensor(sig, dtype=dt), Ti.detach(), nui.detach(), kernel_size)
        fast_flags.append(pre["fast"])
        d = xy[i][None] - pix
        S2 = Ti @ S3 @ Ti.T
        covp = S2 + sig ** 2 * (Ti @ Ti.T) + kernel_size * I2
        if pre["fast"]:
            D, m, tr = ob.invariants(S3)
            S2f = covp - kernel_size * I2
            det3f = D + sig ** 2 * m + sig ** 4 * tr + sig ** 6
            hp = torch.exp(0.5 * (torch.log(D) - torch.log(det3f) + ob.logdet2(S2f) - ob.logdet2(covp)))
            dc = ob.logdet2(covp).exp().detach()
            covp_c = _ConicGradDamping.apply(covp[None], (dc ** 2 / (dc ** 2 + 1e-7))[None])[0]
            Qp = ob.sym_inv(covp_c)
            base = ((d @ Qp) * d).sum(1)
            G = hp * torch.exp(-(nui.detach() + 2) / 2 * torch.log1p(base.clamp(min=0) / nui))
        else:
            G = ob.direct(S3, torch.tensor(sig, dtype=dt), Ti, nui, kernel_size, d)
            base = ((d @ ob.sym_inv(covp.detach())) * d).sum(1)
        # radius: SSS's rule on covp, widened to the alpha-safe bound for Option B
        r = float(sss_radius(covp.detach()[None], nui.detach()[None])[0])
        xyd = xy[i].detach()
        def rect(rad):
            return [((xyd[0] - rad) / BLOCK).trunc().clamp(0, gx), ((xyd[0] + rad + BLOCK - 1) / BLOCK).trunc().clamp(0, gx),
                    ((xyd[1] - rad) / BLOCK).trunc().clamp(0, gy), ((xyd[1] + rad + BLOCK - 1) / BLOCK).trunc().clamp(0, gy)]
        x0, x1, y0, y1 = rect(r)
        if (x1 - x0) * (y1 - y0) == 0:
            alphas.append(zero)
            continue
        if not pre["fast"]:
            r = max(r, math.ceil(optb_radius_bound(pre, nui, opacity[i], kernel_size)))
            x0, x1, y0, y1 = rect(r)
        tx, ty = pix[:, 0] // BLOCK, pix[:, 1] // BLOCK
        in_rect = (tx >= x0) & (tx < x1) & (ty >= y0) & (ty < y1)
        alpha = opacity[i] * G
        keep = in_rect & (base > 0) & (nui >= 1)
        alphas.append(torch.where(keep, alpha, zero))
    alpha = torch.stack(alphas)
    order = torch.argsort(p_view[:, 2].detach())
    color, _ = composite(alpha[order], colors[order], bg)
    return color.T.reshape(3, H, W), torch.tensor(fast_flags)
