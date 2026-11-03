"""Random test scenes and a CUDA render+backward helper for the M2 checks (runs on the GPU VM).

Works with both the M2 rasterizer and the pre-M2 one (which has no kernel_size setting), so the
same script can be run against an old build to check that the filter-off path is unchanged:

    PYTHONPATH=/content/ext_old python scripts/colab/m2_scene.py --out old.pt
"""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.graphics_utils import getProjectionMatrix, getWorld2View2  # noqa: E402

H, W = 48, 64
FOV = math.radians(60)
NUS = (1.0, 1.5, 3.0, 30.0, 500.0, 5000.0)


def camera(device="cuda"):
    import numpy as np
    R, T = np.eye(3), np.array([0.0, 0.0, 0.0])
    fovx = FOV
    fovy = 2 * math.atan(math.tan(fovx / 2) * H / W)
    wv = torch.tensor(getWorld2View2(R, T), dtype=torch.float32).transpose(0, 1)
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).float().transpose(0, 1)
    full = wv.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    campos = wv.inverse()[3, :3]
    return {"world_view": wv.to(device), "full_proj": full.to(device), "campos": campos.to(device),
            "tanfovx": math.tan(fovx / 2), "tanfovy": math.tan(fovy / 2)}


def scene(seed, n=40, device="cuda"):
    """Components in front of the camera, from sub-pixel to ~10 px, mixed ν, signed opacity."""
    g = torch.Generator().manual_seed(seed)
    z = 2 + 4 * torch.rand(n, generator=g)
    x = (torch.rand(n, generator=g) - 0.5) * 1.1 * z * math.tan(FOV / 2) * 2
    y = (torch.rand(n, generator=g) - 0.5) * 1.1 * z * math.tan(FOV / 2) * 2 * H / W
    means = torch.stack([x, y, z], 1)
    q, _ = torch.linalg.qr(torch.randn(n, 3, 3, generator=g))
    scales = torch.exp(torch.randn(n, 3, generator=g) * 1.0 - 3.3) * z[:, None] / 2
    cov3 = q @ torch.diag_embed(scales ** 2) @ q.transpose(1, 2)
    sign = torch.where(torch.rand(n, generator=g) < 0.3, -1.0, 1.0)
    opacity = sign * (0.05 + 0.85 * torch.rand(n, generator=g))
    nu = torch.tensor(NUS)[torch.randint(len(NUS), (n,), generator=g)]
    colors = torch.rand(n, 3, generator=g)
    weights = torch.randn(3, H, W, generator=g)  # loss = Σ weights · image
    return {k: v.to(device) for k, v in dict(means=means, cov3=cov3, opacity=opacity, nu=nu,
                                             colors=colors, weights=weights).items()}


def cov6(cov3):
    i, j = zip(*[(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)])
    return cov3[:, list(i), list(j)]


def render_cuda(sc, cam, kernel_size=0.0, exact_nu_grad=False):
    """Image and gradients (means, cov3D 6-vector, colors, opacity, nu) from the CUDA rasterizer."""
    from diff_t_rasterization import TRasterizationSettings, TRasterizer
    kw = dict(image_height=H, image_width=W, tanfovx=cam["tanfovx"], tanfovy=cam["tanfovy"],
              bg=torch.tensor([0.2, 0.3, 0.4], device="cuda"), scale_modifier=1.0,
              viewmatrix=cam["world_view"], projmatrix=cam["full_proj"], sh_degree=0,
              campos=cam["campos"], prefiltered=False, debug=False)
    if "kernel_size" in TRasterizationSettings._fields:
        kw["kernel_size"] = kernel_size
    elif kernel_size != 0.0:
        raise RuntimeError("this rasterizer build has no kernel_size")
    if exact_nu_grad:  # M5 diagnostic flag; older builds have no such field
        kw["exact_nu_grad"] = True
    rast = TRasterizer(raster_settings=TRasterizationSettings(**kw))
    leaves = {"means": sc["means"].clone(), "cov6": cov6(sc["cov3"]).contiguous(),
              "colors": sc["colors"].clone(), "opacity": sc["opacity"][:, None].clone(),
              "nu": sc["nu"][:, None].clone()}
    for t in leaves.values():
        t.requires_grad_(True)
    means2D = torch.zeros_like(leaves["means"], requires_grad=True)
    n = sc["means"].shape[0]
    img, radii = rast(means3D=leaves["means"], means2D=means2D, opacities=leaves["opacity"],
                      nu_degree=leaves["nu"], cover=torch.zeros(n, 1, device="cuda"),
                      negative_value=torch.ones(n, 1, device="cuda"),
                      colors_precomp=leaves["colors"], cov3D_precomp=leaves["cov6"])
    (img * sc["weights"]).sum().backward()
    out = {"image": img.detach(), "radii": radii.detach()}
    out.update({f"grad_{k}": v.grad.detach() for k, v in leaves.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()
    cam = camera()
    torch.save({s: render_cuda(scene(s), cam) for s in range(args.seeds)}, args.out)
    print(f"saved {args.seeds} scenes to {args.out}")


if __name__ == "__main__":
    main()
