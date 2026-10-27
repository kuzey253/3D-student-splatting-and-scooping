"""M4 checks on the GPU VM (Option B in CUDA), run from the SSS root.

1. Parity: the CUDA rasterizer in Option B mode (cov3D_precomp path) against
   mip_sss.raster_ref.rasterize_option_b in float64, on random scenes with both modes present,
   with σ3 = 0 and σ3 > 0. Image and gradients for means, cov3D, colors, opacity and ν;
   tolerance 1e-4 (README). A case with 1-3 pixels off by > 1e-3 is a float32 decision flip
   (reported, not scored; more than two such cases fail).
2. Wiring: SSS's render() in Option B mode with an NTModel and its filter_3D, against the same
   reference with the covariance built from the scales and rotations. The image is gated at 1e-4;
   the gradients are reported only (SSS's float32 scale/rotation path, NOTES.md 34/36).
The unchanged-when-off checks are M2's and M3's suites, run against a build of the M3 commit.

Writes /content/m0/m4_checks.json; prints M4_CHECKS=OK or M4_CHECKS=FAIL.
"""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_scene import FOV, H, W, camera, cov6, scene  # noqa: E402
import m3_checks  # noqa: E402

from mip_sss.analysis import quat_to_rot  # noqa: E402
from mip_sss.raster_ref import rasterize_option_b  # noqa: E402

OUT = Path("/content/m0")
TOL = 1e-4
S = 0.1
BG = [0.2, 0.3, 0.4]
GRADS = ("grad_means", "grad_cov6", "grad_colors", "grad_opacity", "grad_nu")


def sigma3_for(sc, on, seed):
    if not on:
        return torch.zeros_like(sc["nu"])
    g = torch.Generator().manual_seed(500 + seed)
    return (torch.rand(sc["nu"].shape, generator=g) * 0.02 * sc["means"][:, 2].cpu() / 2).to(sc["nu"].device)


def cuda_option_b(sc, cam, sigma3):
    from diff_t_rasterization import TRasterizationSettings, TRasterizer
    st = TRasterizationSettings(image_height=H, image_width=W, tanfovx=cam["tanfovx"], tanfovy=cam["tanfovy"],
                                bg=torch.tensor(BG, device="cuda"), scale_modifier=1.0,
                                viewmatrix=cam["world_view"], projmatrix=cam["full_proj"], sh_degree=0,
                                campos=cam["campos"], prefiltered=False, debug=False, kernel_size=S,
                                option_b=True)
    rast = TRasterizer(raster_settings=st)
    leaves = {"means": sc["means"].clone(), "cov6": cov6(sc["cov3"]).contiguous(),
              "colors": sc["colors"].clone(), "opacity": sc["opacity"][:, None].clone(),
              "nu": sc["nu"][:, None].clone()}
    for t in leaves.values():
        t.requires_grad_(True)
    n = sc["means"].shape[0]
    img, radii = rast(means3D=leaves["means"], means2D=torch.zeros_like(leaves["means"], requires_grad=True),
                      opacities=leaves["opacity"], nu_degree=leaves["nu"], cover=torch.zeros(n, 1, device="cuda"),
                      negative_value=torch.ones(n, 1, device="cuda"), colors_precomp=leaves["colors"],
                      cov3D_precomp=leaves["cov6"], filter_3d=sigma3.float().contiguous())
    (img * sc["weights"]).sum().backward()
    out = {"image": img.detach().cpu(), "radii": radii.detach().cpu()}
    out.update({f"grad_{k}": v.grad.detach().cpu() for k, v in leaves.items()})
    return out


def reference(sc, cam, sigma3):
    """On the CPU (optionb_staged builds its constants there)."""
    d = torch.float64
    sc = {k: v.cpu() for k, v in sc.items()}
    cam = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in cam.items()}
    sigma3 = sigma3.cpu()
    c = sc["cov3"].to(d)
    leaves = {"means": sc["means"].to(d), "colors": sc["colors"].to(d), "opacity": sc["opacity"].to(d),
              "nu": sc["nu"].to(d),
              "cov6": torch.stack([c[:, 0, 0], c[:, 0, 1], c[:, 0, 2], c[:, 1, 1], c[:, 1, 2], c[:, 2, 2]], 1)}
    for t in leaves.values():
        t.requires_grad_(True)
    v = leaves["cov6"]
    cov3 = torch.stack([torch.stack([v[:, 0], v[:, 1], v[:, 2]], -1), torch.stack([v[:, 1], v[:, 3], v[:, 4]], -1),
                        torch.stack([v[:, 2], v[:, 4], v[:, 5]], -1)], 1)
    img, fast = rasterize_option_b(leaves["means"], cov3, leaves["colors"], leaves["opacity"], leaves["nu"],
                                   sigma3.to(d), cam["world_view"].to(d), cam["full_proj"].to(d), cam["tanfovx"],
                                   cam["tanfovy"], H, W, torch.tensor(BG, dtype=d), S)
    (img * sc["weights"].to(d)).sum().backward()
    out = {"image": img.detach(), "fast": fast}
    out.update({f"grad_{k}": t.grad.detach() for k, t in leaves.items()})
    return out


def rel_norm(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def parity():
    cam = camera()
    rows, ok = [], True
    for s3_on in (False, True):
        for seed in range(6):
            sc = scene(seed)
            sigma3 = sigma3_for(sc, s3_on, seed)
            cu, ref = cuda_option_b(sc, cam, sigma3), reference(sc, cam, sigma3)
            diff = (cu["image"].double() - ref["image"]).abs().max(0).values
            row = {"sigma3": s3_on, "seed": seed, "option_b_components": int((~ref["fast"]).sum()),
                   "image": float((cu["image"].double() - ref["image"]).abs().max() / ref["image"].abs().max()),
                   "pixels_off_by_1e-3": int((diff > 1e-3).sum())}
            row.update({g: rel_norm(cu[g].reshape(ref[g].shape), ref[g]) for g in GRADS})
            row["nan"] = any(bool(torch.isnan(t).any()) for t in cu.values() if torch.is_tensor(t))
            row["flip"] = 1 <= row["pixels_off_by_1e-3"] <= 3
            row["ok"] = not row["nan"] and (row["flip"] or (row["pixels_off_by_1e-3"] == 0 and row["image"] <= TOL
                                                           and all(row[g] <= TOL for g in GRADS)))
            ok &= row["ok"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    flips = sum(r["flip"] for r in rows)
    return ok and flips <= 2, rows


def wiring():
    """SSS render() in Option B mode with an NTModel (scales/rotations path) and its filter_3D."""
    c = camera()
    fovy = 2 * math.atan(math.tan(FOV / 2) * H / W)
    cam = SimpleNamespace(FoVx=FOV, FoVy=fovy, image_width=W, image_height=H, world_view_transform=c["world_view"],
                          full_proj_transform=c["full_proj"], camera_center=c["campos"])
    from scene.nt_model import NTModel
    from t_renderer import render
    from torch import nn
    rows, ok = [], True
    for seed in range(4):
        p = m3_checks.random_params(seed)
        pc = NTModel(0, 100.0)
        for k in m3_checks.PARAMS:
            setattr(pc, k, nn.Parameter(p[k].cuda().clone().requires_grad_(True)))
        n = p["_xyz"].shape[0]
        pc._negative = nn.Parameter(torch.ones(n, 1, device="cuda"), requires_grad=False)
        pc._features_dc = torch.zeros(n, 1, 3, device="cuda")
        pc._features_rest = torch.zeros(n, 0, 3, device="cuda")
        pc.filter_3D = p["filter_3D"].cuda()
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False,
                               mip_kernel_size=S, mip_filter_3d=True, mip_option_b=True)
        img = render(cam, pc, pipe, torch.tensor(BG, device="cuda"), override_color=p["colors"].cuda())["render"]
        (img * p["weights"].cuda()).sum().backward()
        d = torch.float64
        s2 = torch.exp(p["_scaling"].to(d)) ** 2
        Rm = quat_to_rot(p["_rotation"].to(d))
        cov3 = Rm @ torch.diag_embed(s2) @ Rm.transpose(1, 2)
        ref, _ = rasterize_option_b(p["_xyz"].to(d), cov3, p["colors"].to(d), torch.tanh(p["_opacity"].to(d)[:, 0]),
                                    torch.clamp(p["_degree"].to(d)[:, 0], 1, 1e4), p["filter_3D"].to(d)[:, 0],
                                    c["world_view"].cpu().to(d), c["full_proj"].cpu().to(d),
                                    math.tan(FOV / 2), math.tan(fovy / 2), H, W, torch.tensor(BG, dtype=d), S)
        diff = (img.detach().cpu().double() - ref).abs()
        row = {"seed": seed, "image": float(diff.max() / ref.abs().max()),
               "pixels_off_by_1e-3": int((diff.max(0).values > 1e-3).sum()),
               "grads_finite": all(bool(torch.isfinite(getattr(pc, k).grad).all()) for k in m3_checks.PARAMS)}
        row["ok"] = row["grads_finite"] and (1 <= row["pixels_off_by_1e-3"] <= 3 or row["image"] <= TOL)
        ok &= row["ok"]
        rows.append(row)
        print(json.dumps(row), flush=True)
    return ok, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok_p, prow = parity()
    ok_w, wrow = wiring()
    (OUT / "m4_checks.json").write_text(json.dumps({"parity": prow, "wiring": wrow, "parity_ok": ok_p,
                                                    "wiring_ok": ok_w}, indent=1))
    print(f"M4_CHECKS={'OK' if ok_p and ok_w else 'FAIL'} parity_ok={ok_p} wiring_ok={ok_w}")


if __name__ == "__main__":
    main()
