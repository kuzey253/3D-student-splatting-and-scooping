"""M2 checks on the GPU VM, run from the SSS root after both rasterizer builds exist.

1. CUDA parity: for random scenes and s ∈ {0, 0.1, 0.3}, the CUDA image and gradients (means,
   cov3D, colors, opacity, ν) against mip_sss.raster_ref in float64. README tolerance: 1e-4,
   measured as max |Δ| / max |ref| for the image and ‖Δ‖ / ‖ref‖ for each gradient.
2. Filter off is unchanged: with s = 0 the M2 build must match the pre-M2 build. The image must be
   bitwise identical; gradients are compared against the run-to-run noise of float atomics.

Writes /content/m0/m2_checks.json and prints M2_CHECKS=OK or M2_CHECKS=FAIL.
"""
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_scene import H, W, camera, render_cuda, scene  # noqa: E402

from mip_sss.raster_ref import rasterize  # noqa: E402

OUT = Path("/content/m0")
TOL = 1e-4
SEEDS = 6
KERNEL_SIZES = (0.0, 0.1, 0.3)
GRADS = ("grad_means", "grad_cov6", "grad_colors", "grad_opacity", "grad_nu")


def reference(sc, cam, kernel_size, sss_nu_grad=True):
    d = torch.float64
    leaves = {"means": sc["means"].to(d), "cov6": None, "colors": sc["colors"].to(d),
              "opacity": sc["opacity"].to(d), "nu": sc["nu"].to(d)}
    c = sc["cov3"].to(d)
    leaves["cov6"] = torch.stack([c[:, 0, 0], c[:, 0, 1], c[:, 0, 2], c[:, 1, 1], c[:, 1, 2],
                                  c[:, 2, 2]], 1)
    for t in leaves.values():
        t.requires_grad_(True)
    v = leaves["cov6"]
    cov3 = torch.stack([torch.stack([v[:, 0], v[:, 1], v[:, 2]], -1),
                        torch.stack([v[:, 1], v[:, 3], v[:, 4]], -1),
                        torch.stack([v[:, 2], v[:, 4], v[:, 5]], -1)], 1)
    img = rasterize(leaves["means"], cov3, leaves["colors"], leaves["opacity"], leaves["nu"],
                    cam["world_view"].to(d), cam["full_proj"].to(d), cam["tanfovx"],
                    cam["tanfovy"], H, W, torch.tensor([0.2, 0.3, 0.4], dtype=d, device="cuda"),
                    kernel_size=kernel_size, sss_nu_grad=sss_nu_grad)
    (img * sc["weights"].to(d)).sum().backward()
    out = {"image": img.detach()}
    out.update({f"grad_{k}": t.grad.detach() for k, t in leaves.items()})
    return out


def rel_norm(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def parity():
    cam = camera()
    rows, ok = [], True
    for ks in KERNEL_SIZES:
        for s in range(SEEDS):
            sc = scene(s)
            cu, ref = render_cuda(sc, cam, ks), reference(sc, cam, ks)
            diff = (cu["image"].double() - ref["image"]).abs()
            row = {"kernel_size": ks, "seed": s,
                   "image": float(diff.max() / ref["image"].abs().max()),
                   "pixels_off_by_1e-3": int((diff > 1e-3).sum())}
            row.update({g: rel_norm(cu[g].reshape(ref[g].shape), ref[g]) for g in GRADS})
            row["nan"] = any(bool(torch.isnan(v).any()) for v in cu.values())
            row["ok"] = (not row["nan"] and row["image"] <= TOL
                         and all(row[g] <= TOL for g in GRADS))
            ok &= row["ok"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    return ok, rows


def flag_off_unchanged():
    old_file = OUT / "m2_old.pt"
    if not old_file.exists():
        return False, {"error": f"{old_file} missing (old build did not run)"}
    old = torch.load(old_file)
    cam = camera()
    worst = {"image_bitwise_equal": True, "grad_max_rel": 0.0, "grad_noise_rel": 0.0}
    for s in old:
        sc = scene(s)
        new_a, new_b = render_cuda(sc, cam, 0.0), render_cuda(sc, cam, 0.0)
        worst["image_bitwise_equal"] &= bool(torch.equal(new_a["image"], old[s]["image"]))
        worst["radii_equal"] = worst.get("radii_equal", True) and bool(
            torch.equal(new_a["radii"], old[s]["radii"]))
        for g in GRADS:
            worst["grad_max_rel"] = max(worst["grad_max_rel"], rel_norm(new_a[g], old[s][g]))
            worst["grad_noise_rel"] = max(worst["grad_noise_rel"], rel_norm(new_a[g], new_b[g]))
    ok = (worst["image_bitwise_equal"] and worst["radii_equal"]
          and worst["grad_max_rel"] <= max(1e-6, 10 * worst["grad_noise_rel"]))
    return ok, worst


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok_p, rows = parity()
    ok_o, off = flag_off_unchanged()
    print(json.dumps({"flag_off": off}))
    (OUT / "m2_checks.json").write_text(json.dumps({"parity": rows, "flag_off": off,
                                                    "parity_ok": ok_p, "flag_off_ok": ok_o},
                                                   indent=1))
    print(f"M2_CHECKS={'OK' if ok_p and ok_o else 'FAIL'} parity_ok={ok_p} flag_off_ok={ok_o}")


if __name__ == "__main__":
    main()
