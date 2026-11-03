"""M5 checks on the GPU VM, run from the SSS root.

The diagnostic flag exact_nu_grad (--mip_exact_nu_grad) adds the −½·ln(1 + b/ν)·G term SSS omits
from its ν gradient (NOTES.md 13). With it on, the CUDA image and gradients must match
mip_sss.raster_ref with sss_nu_grad=False (the exact gradient) on M2's random scenes, for s ∈ {0, 0.1},
at the README's 1e-4. Its ν gradient must differ from SSS's own, so the flag is known to act.
Flag-off parity is covered by the M2/M3/M4 suites against the M4 build.

Writes /content/m0/m5_checks.json and prints M5_CHECKS=OK or M5_CHECKS=FAIL.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m2_checks import GRADS, OUT, SEEDS, TOL, reference, rel_norm  # noqa: E402
from m2_scene import camera, render_cuda, scene  # noqa: E402


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cam = camera()
    rows, ok = [], True
    for ks in (0.0, 0.1):
        for s in range(SEEDS):
            sc = scene(s)
            cu, ref = render_cuda(sc, cam, ks, exact_nu_grad=True), reference(sc, cam, ks, sss_nu_grad=False)
            sss = render_cuda(sc, cam, ks)
            diff = (cu["image"].double() - ref["image"]).abs()
            row = {"kernel_size": ks, "seed": s, "image": float(diff.max() / ref["image"].abs().max()),
                   "pixels_off_by_1e-3": int((diff > 1e-3).sum()),
                   "nu_grad_change_vs_sss": rel_norm(sss["grad_nu"], cu["grad_nu"])}
            row.update({g: rel_norm(cu[g].reshape(ref[g].shape), ref[g]) for g in GRADS})
            row["nan"] = any(bool(torch.isnan(v).any()) for v in cu.values())
            row["ok"] = (not row["nan"] and row["image"] <= TOL and all(row[g] <= TOL for g in GRADS)
                         and row["nu_grad_change_vs_sss"] > 10 * TOL)
            ok &= row["ok"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    (OUT / "m5_checks.json").write_text(json.dumps({"exact_nu_grad": rows, "ok": ok}, indent=1))
    print(f"M5_CHECKS={'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
