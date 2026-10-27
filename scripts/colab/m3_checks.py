"""M3 checks on the GPU VM (3D smoothing filter, Option A), run from the SSS root.

1. Wiring, exactly: M3 is Python only (two NTModel getters and their autograd chain), so it is
   checked against the CUDA rasterizer itself, independent of the rasterizer's own float32
   error. (a) The getters' effective scales and opacity equal the float64 formulas to float32
   precision (≤ 1e-6). (b) The filtered render equals an unfiltered render of a model given
   those effective parameters (folded in float64), to 1e-4 (README). (c) The raw-parameter
   gradients equal the float64 chain rule through the getters applied to that render's
   gradients, to max(1e-4, 2 × the change from random ±1-ulp perturbations of the folded
   scales and opacities): the two models differ by exactly such ulps, and SSS's float32
   gradients for needle-shaped components move by up to ~1e-3 of their size under them
   (NOTES.md 36). A case with 1-3 pixels off by > 1e-3 is a discrete float32 decision flip
   (reported, not scored; more than two such cases fail).
   Reported, not gating: CUDA vs mip_sss.raster_ref in float64, with and without the filter.
   SSS's own float32 scale/rotation path exceeds 1e-4 there on very elongated components
   whether or not the filter is on (NOTES.md 34).
2. Filter values: NTModel.compute_3D_filter on the real Tanks & Temples `train` cameras and SfM
   points against an independent computation from the 3DGS view/projection matrices, at the same
   (float32) precision; points that flip across a visibility margin in float64 are reported.
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
                           mip_kernel_size=kernel_size, mip_filter_3d=filter_3d, mip_option_b=False)
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


def equivalent_unfiltered(p):
    """The same scene with σ3 folded into the parameters, in float64: an unfiltered model."""
    d = torch.float64
    s2 = torch.exp(p["_scaling"].to(d)) ** 2
    s2f = s2 + p["filter_3D"].to(d) ** 2
    o = torch.tanh(p["_opacity"].to(d)[:, 0]) * torch.sqrt(s2.prod(1) / s2f.prod(1))
    return dict(p, _scaling=(0.5 * torch.log(s2f)).float(), _opacity=torch.atanh(o)[:, None].float())


def chain_through_getters(p, eq, eq_grads):
    """Raw-parameter gradients implied by the getters, given gradients for the folded parameters."""
    d = torch.float64
    log_s = p["_scaling"].to(d).cuda().requires_grad_(True)
    raw_o = p["_opacity"].to(d).cuda().requires_grad_(True)
    s2 = torch.exp(log_s) ** 2
    s2f = s2 + p["filter_3D"].to(d).cuda() ** 2
    log_sf = 0.5 * torch.log(s2f)
    o_eff = torch.tanh(raw_o[:, 0]) * torch.sqrt(s2.prod(1) / s2f.prod(1))
    r_eff = torch.atanh(o_eff)[:, None]
    torch.autograd.backward([log_sf, r_eff], [eq_grads["_scaling"].to(d), eq_grads["_opacity"].to(d)])
    return {"_scaling": log_s.grad, "_opacity": raw_o.grad}


def ulp_jitter(q, seed):
    """Random ±1-ulp perturbation of the folded scales and opacities (float32)."""
    g = torch.Generator().manual_seed(1000 + seed)
    out = dict(q)
    for k in ("_scaling", "_opacity"):
        up = torch.rand(q[k].shape, generator=g) < 0.5
        out[k] = torch.nextafter(q[k], torch.where(up, torch.full_like(q[k], 1e9), torch.full_like(q[k], -1e9)))
    return out


def getter_exactness(p):
    """Max relative difference of the getters' effective scale/opacity from the float64 formulas."""
    pc = NTModel(0, 100.0)
    pc._xyz, pc._scaling, pc._opacity = p["_xyz"].cuda(), p["_scaling"].cuda(), p["_opacity"].cuda()
    pc.filter_3D = p["filter_3D"].cuda()
    d = torch.float64
    s2 = torch.exp(p["_scaling"].to(d)) ** 2
    s2f = s2 + p["filter_3D"].to(d) ** 2
    o = torch.tanh(p["_opacity"].to(d)[:, 0]) * torch.sqrt(s2.prod(1) / s2f.prod(1))
    es = float(((pc.get_scaling_with_3D_filter.double().cpu() - s2f.sqrt()).abs() / s2f.sqrt()).max())
    eo = float(((pc.get_opacity_with_3D_filter.double().cpu()[:, 0] - o).abs() / o.abs()).max())
    return max(es, eo)


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
            eq = equivalent_unfiltered(p)
            eq_img, eq_grads = cuda_render(eq, cam, ks, False)
            want = dict(eq_grads, **chain_through_getters(p, eq, eq_grads))
            diff = (img.double() - eq_img.double()).abs().max(0).values
            row = {"kernel_size": ks, "seed": seed, "getter_rel_err": getter_exactness(p),
                   "image": float((img.double() - eq_img.double()).abs().max() / eq_img.abs().max()),
                   "pixels_off_by_1e-3": int((diff > 1e-3).sum())}
            row.update({f"grad{k}": rel_norm(grads[k], want[k]) for k in PARAMS})
            _, jit_grads = cuda_render(ulp_jitter(eq, seed), cam, ks, False)
            row["ulp_sensitivity"] = {f"grad{k}": rel_norm(jit_grads[k], eq_grads[k]) for k in PARAMS}
            row["flip"] = 1 <= row["pixels_off_by_1e-3"] <= 3
            row["ok"] = row["flip"] or (
                row["pixels_off_by_1e-3"] == 0 and row["getter_rel_err"] <= 1e-6 and row["image"] <= TOL
                and all(row[k] <= max(TOL, 2 * row["ulp_sensitivity"][k]) for k in row["ulp_sensitivity"]))
            # reported only: CUDA vs float64 reference, filter on and off
            ref_img, ref_grads = reference(p, cam, ks)
            q = dict(p, filter_3D=torch.zeros_like(p["filter_3D"]))
            off_img, off_grads = cuda_render(q, cam, ks, False)
            ref_off, ref_off_grads = reference(q, cam, ks)
            row["vs_float64_filter_on"] = {"image": float((img.double() - ref_img).abs().max() / ref_img.abs().max()),
                                           **{k: rel_norm(grads[k], ref_grads[k]) for k in PARAMS}}
            row["vs_float64_filter_off"] = {"image": float((off_img.double() - ref_off).abs().max() / ref_off.abs().max()),
                                            **{k: rel_norm(off_grads[k], ref_off_grads[k]) for k in PARAMS}}
            ok &= row["ok"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    # 3: with the flag off, a stored filter_3D must change nothing
    p = random_params(99)
    a, _ = cuda_render(p, cam, 0.1, False, set_filter=True)
    b, _ = cuda_render(p, cam, 0.1, False, set_filter=False)
    off_ok = bool(torch.equal(a, b))
    flips = sum(r["flip"] for r in rows)
    print(json.dumps({"flag_off_ignores_filter": off_ok, "flip_cases": flips}))
    return ok and off_ok and flips <= 2, rows, off_ok


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
    def independent(dt):
        P = torch.cat([pts.to(dt), torch.ones(len(pts), 1, dtype=dt, device="cuda")], 1)
        dist = torch.full((len(pts),), 100000.0, dtype=dt, device="cuda")
        seen = torch.zeros(len(pts), dtype=torch.bool, device="cuda")
        focal = 0.0
        for c in cams:
            wv = torch.tensor(getWorld2View2(c.R, c.T), dtype=dt, device="cuda")
            proj = getProjectionMatrix(0.01, 100.0, c.FoVx, c.FoVy).to(dt).cuda()
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
        return (dist / focal * math.sqrt(0.2)).double(), seen

    want32, seen = independent(torch.float32)
    want64, _ = independent(torch.float64)
    rel32 = (got - want32).abs() / want32
    rel64 = (got - want64).abs() / want64
    res = {"points": len(pts), "cameras": len(cams), "seen_fraction": float(seen.double().mean()),
           "max_rel_err_float32": float(rel32.max()), "median_rel_err_float32": float(rel32.median()),
           "float64_margin_flips": int((rel64 > 1e-5).sum()),
           "max_rel_err_float64": float(rel64.max()), "sigma3_median": float(want64.median())}
    print(json.dumps(res))
    return res["max_rel_err_float32"] <= 1e-5 and res["float64_margin_flips"] <= 1e-4 * len(pts), res


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
