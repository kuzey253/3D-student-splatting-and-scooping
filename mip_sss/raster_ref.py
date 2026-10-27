"""Float64 PyTorch re-implementation of SSS's CUDA rasterizer, for CUDA parity tests (M2+).

Follows diff-t-rasterization step by step: frustum test (z > 0.2), projection with the 3DGS
view/projection matrices, EWA 2D covariance with the 1.3·tan(fov) clamp, the Mip-SSS 2D filter
(Option A) when kernel_size > 0, SSS's culling radius and 16x16 tile rectangles, the t kernel
evaluated at integer pixel coordinates, and front-to-back compositing (render.composite).

Inputs are the rasterizer's own: means3D (N,3), cov3D (N,3,3) (the precomputed-covariance path),
colors (N,3), opacity (N,) already activated (signed), nu (N,), and the camera's
world_view_transform / full_proj_transform exactly as SSS stores them (transposed, row-major).
"""
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
