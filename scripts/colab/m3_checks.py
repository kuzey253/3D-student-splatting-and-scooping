"""M3 checks on the GPU VM (3D smoothing filter, Option A), run from the SSS root.

1. Wiring: SSS's render() with pipe.mip_filter_3d on an NTModel built from random parameters, with
   and without the 2D filter, against mip_sss.raster_ref in float64 with Σ3' = R(S² + σ3²)Rᵀ and
   opacity × sqrt(|Σ3| / |Σ3'|). Image and gradients for _xyz, _scaling, _rotation, _opacity and
   _degree; tolerance 1e-4 (README).
2. Filter values: NTModel.compute_3D_filter on the real Tanks & Temples `train` cameras and SfM
   points against an independent computation from the 3DGS view/projection matrices.
3. Filter off is unaffected by a stored filter_3D: the image is bitwise identical either way.

Writes /content/m0/m3_checks.json; prints M3_CHECKS=OK or M3_CHECKS=FAIL.
"""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_scene import FOV, H, W, camera  # noqa: E402

from mip_sss.analysis import quat_to_rot  # noqa: E402
from mip_sss.raster_ref import rasterize  # noqa: E402
from scene.nt_model import NTModel  # noqa: E402
from t_renderer import render  # noqa: E402
from utils.graphics_utils import fov2focal, getProjectionMatrix, getWorld2View2  # noqa: E402

OUT = Path("/content/m0")
TOL = 1e-4
BG = [0.2, 0.3, 0.4]
PARAMS = ("_xyz", "_scaling", "_rotation", "_opacity", "_degree")


def random_params(seed, n=40):
    g = torch.Generator().manual_seed(seed)
    z = 2 + 4 * torch.rand(n, generator=g)
    x = (torch.rand(n, generator=g) - 0.5) * 1.1 * z * math.tan(FOV / 2) * 2
    y = (torch.rand(n, generator=g) - 0.5) * 1.1 * z * math.tan(FOV / 2) * 2 * H / W
    o = torch.where(torch.rand(n, generator=g) < 0.3, -1.0, 1.0) * (0.05 + 0.85 * torch.rand(n, generator=g))
    return {
        "_xyz": torch.stack([x, y, z], 1),
        "_scaling": torch.log(torch.exp(torch.randn(n, 3, generator=g) - 3.3) * z[:, None] / 2),
        "_rotation": torch.randn(n, 4, generator=g),
        "_opacity": torch.atanh(o)[:, None],
        "_degree": torch.tensor([1.0, 1.5, 3.0, 30.0, 500.0, 5000.0])[
            torch.randint(6, (n,), generator=g)][:, None],
        "colors": torch.rand(n, 3, generator=g),
        # σ3 between 0 and about a pixel's footprint at the component's depth
        "filter_3D": (torch.rand(n, generator=g) * 0.02 * z / 2)[:, None],
        "weights": torch.randn(3, H, W, generator=g),
    }


def cuda_render(p, cam, kernel_size, filter_3d, set_filter=True):
    pc = NTModel(0, 100.0)
    for k in PARAMS:
        setattr(pc, k, nn.Parameter(p[k].cuda().clone().requires_grad_(True)))
    pc._negative = nn.Parameter(torch.ones(p["_xyz"].shape[0], 1, device="cuda"), requires_grad=False)
    pc._features_dc = torch.zeros(p["_xyz"].shape[0], 1, 3, device="cuda")
    pc._features_rest = torch.zeros(p["_xyz"].shape[0], 0, 3, device="cuda")
    if set_filter:
        pc.filter_3D = p["filter_3D"].cuda()
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False,
                           mip_kernel_size=kernel_size, mip_filter_3d=filter_3d)
    img = render(cam, pc, pipe, torch.tensor(BG, device="cuda"), override_color=p["colors"].cuda())["render"]
    (img * p["weights"].cuda()).sum().backward()
    return img.detach(), {k: getattr(pc, k).grad.detach() for k in PARAMS}


def reference(p, cam, kernel_size):
    d = torch.float64
    leaves = {k: p[k].to(d).cuda().clone().requires_grad_(True) for k in PARAMS}
    s2 = torch.exp(leaves["_scaling"]) ** 2
    s2f = s2 + p["filter_3D"].to(d).cuda() ** 2
    R = quat_to_rot(leaves["_rotation"])
    cov3 = R @ torch.diag_embed(s2f) @ R.transpose(1, 2)
    o = torch.tanh(leaves["_opacity"][:, 0]) * torch.sqrt(s2.prod(1) / s2f.prod(1))
    nu = torch.nn.functional.hardtanh(leaves["_degree"][:, 0], 1.0, 1e4)  # as NTModel
    img = rasterize(leaves["_xyz"], cov3, p["colors"].to(d).cuda(), o, nu,
                    cam.world_view_transform.to(d), cam.full_proj_transform.to(d),
                    math.tan(cam.FoVx / 2), math.tan(cam.FoVy / 2), H, W,
                    torch.tensor(BG, dtype=d, device="cuda"), kernel_size=kernel_size)
    (img * p["weights"].to(d).cuda()).sum().backward()
    return img.detach(), {k: v.grad.detach() for k, v in leaves.items()}


def rel_norm(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def wiring():
    c = camera()
    fovy = 2 * math.atan(math.tan(FOV / 2) * H / W)
    cam = SimpleNamespace(FoVx=FOV, FoVy=fovy, image_width=W, image_height=H,
                          world_view_transform=c["world_view"], full_proj_transform=c["full_proj"],
                          camera_center=c["campos"])
    rows, ok = [], True
    for ks in (0.0, 0.1):
        for seed in range(6):
            p = random_params(seed)
            img, grads = cuda_render(p, cam, ks, True)
            ref_img, ref_grads = reference(p, cam, ks)
            row = {"kernel_size": ks, "seed": seed,
                   "image": float((img.double() - ref_img).abs().max() / ref_img.abs().max())}
            row.update({f"grad{k}": rel_norm(grads[k], ref_grads[k]) for k in PARAMS})
            row["ok"] = all(v <= TOL for k, v in row.items() if k == "image" or k.startswith("grad"))
            ok &= row["ok"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    # 3: with the flag off, a stored filter_3D must change nothing
    p = random_params(99)
    a, _ = cuda_render(p, cam, 0.1, False, set_filter=True)
    b, _ = cuda_render(p, cam, 0.1, False, set_filter=False)
    off_ok = bool(torch.equal(a, b))
    print(json.dumps({"flag_off_ignores_filter": off_ok}))
    return ok and off_ok, rows, off_ok


def filter_values(data=Path("/content/data/train")):
    from scene.dataset_readers import sceneLoadTypeCallbacks
    info = sceneLoadTypeCallbacks["Colmap"](str(data), "images", True)
    cams = [SimpleNamespace(R=c.R, T=c.T, FoVx=c.FovX, FoVy=c.FovY, image_width=c.width,
                            image_height=c.height) for c in info.train_cameras]
    pts = torch.tensor(np.asarray(info.point_cloud.points), dtype=torch.float32, device="cuda")
    pc = NTModel(0, 100.0)
    pc._xyz = pts
    pc.compute_3D_filter(cams, 0.2)
    got = pc.filter_3D[:, 0].double()

    # independent: 3DGS matrices, pixel position from the projection matrix
    P = torch.cat([pts.double(), torch.ones(len(pts), 1, dtype=torch.float64, device="cuda")], 1)
    dist = torch.full((len(pts),), 100000.0, dtype=torch.float64, device="cuda")
    seen = torch.zeros(len(pts), dtype=torch.bool, device="cuda")
    focal = 0.0
    for c in cams:
        wv = torch.tensor(getWorld2View2(c.R, c.T), dtype=torch.float64, device="cuda")
        proj = getProjectionMatrix(0.01, 100.0, c.FoVx, c.FoVy).double().cuda()
        pv = P @ wv.T
        clip = P @ (proj @ wv).T
        px = (clip[:, 0] / clip[:, 3] + 1) / 2 * c.image_width
        py = (clip[:, 1] / clip[:, 3] + 1) / 2 * c.image_height
        valid = ((pv[:, 2] > 0.2) & (px >= -0.15 * c.image_width) & (px <= 1.15 * c.image_width)
                 & (py >= -0.15 * c.image_height) & (py <= 1.15 * c.image_height))
        dist[valid] = torch.minimum(dist[valid], pv[valid, 2])
        seen |= valid
        focal = max(focal, fov2focal(c.FoVx, c.image_width))
    dist[~seen] = dist[seen].max()
    want = dist / focal * math.sqrt(0.2)
    err = float(((got - want).abs() / want).max())
    res = {"points": len(pts), "cameras": len(cams), "seen_fraction": float(seen.double().mean()),
           "max_rel_err": err, "sigma3_median": float(want.median())}
    print(json.dumps(res))
    return err <= 1e-5, res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok_w, rows, off_ok = wiring()
    ok_f, fres = filter_values()
    (OUT / "m3_checks.json").write_text(json.dumps(
        {"wiring": rows, "flag_off_ignores_filter": off_ok, "filter_values": fres,
         "wiring_ok": ok_w, "filter_values_ok": ok_f}, indent=1))
    print(f"M3_CHECKS={'OK' if ok_w and ok_f else 'FAIL'} wiring_ok={ok_w} filter_values_ok={ok_f}")


if __name__ == "__main__":
    main()
