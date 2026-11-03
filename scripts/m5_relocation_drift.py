"""M5: how far relocation drifts from the filtered kernel (PLAN M5, README M5.1).

Relocation replaces one component by N copies at the same place, whose composite should render
like the original: alpha_new(u) = 1 − (1 − o′K′(u))^N against alpha_old(u) = o K(u). SSS derives
o′ and the scale factor for the unfiltered t kernel. This script measures, on a pixel grid, the max
|alpha_new − alpha_old| for

  - SSS: no filters, SSS's relocation (the drift relocation has on its own);
  - raw: both Mip filters on, SSS's relocation on the raw opacity (M3/M4 training);
  - effective: both filters on, relocation on the 3D-filtered opacity (M5.1, mip_sss.relocation).

Filtered kernels are evaluated both as the fast path renders them (Option A in 3D and 2D) and
exactly (the λ-integral reference, which Option B approximates to 1e-3). Components are given in
pixel units, viewed along their third axis; σ3 = sqrt(0.2) px is the 3D filter at the camera that
sets it (smaller elsewhere), s = 0.1 px².

    python3 scripts/m5_relocation_drift.py [--out results/m5/relocation_drift.md]
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mip_sss.kernels import option_a_3d, t_kernel  # noqa: E402
from mip_sss.reference import lambda_integral_3d  # noqa: E402
from mip_sss.relocation import filter_3d_coef, relocate  # noqa: E402

torch.set_default_dtype(torch.float64)
S2D = 0.1
SIGMA3 = math.sqrt(0.2)
SHAPES = {"large": (5.0, 3.0, 2.0), "pixel": (1.0, 0.5, 0.3), "sub-pixel": (0.2, 0.1, 0.05),
          "face-on disc": (3.0, 2.0, 0.01), "needle": (3.0, 0.02, 0.02)}


def grid(scales):
    # 101 points per axis, odd so the centre (where the unfiltered sub-pixel kernels peak) is on it
    half = 6 * max(scales[0], 1.0) + 4
    x = torch.linspace(-half, half, 101)
    return torch.stack(torch.meshgrid(x, x, indexing="ij"), -1).reshape(-1, 2)


def kernel(kind, u, scales, nu):
    cov3 = torch.diag(torch.tensor(scales) ** 2)[None]
    M = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    mu, nu_t = torch.zeros(1, 2), torch.tensor([float(nu)])
    if kind == "unfiltered":
        return t_kernel(u[:, None, :], cov3[:, :2, :2], nu_t)[:, 0]
    s3 = torch.tensor([SIGMA3 ** 2])
    if kind == "fast path":
        return option_a_3d(u, mu, cov3, M, nu_t, s3, S2D)[0]
    return lambda_integral_3d(u, mu, cov3, M, nu_t, s3, S2D)[0]


def drift(shape, nu, o, N):
    scales = SHAPES[shape]
    u = grid(scales)
    o_t, s_t = torch.tensor([o]), torch.tensor([scales])
    nu_t, N_t = torch.tensor([float(nu)]), torch.tensor([N - 1]) + 1
    raw = relocate(o_t, s_t, nu_t, N_t)
    eff = relocate(o_t, s_t, nu_t, N_t, sigma=torch.tensor([SIGMA3]))
    o_eff = o * float(filter_3d_coef(s_t, torch.tensor([SIGMA3])))
    row = {"o_eff": o_eff, "floored": 1 - (1 - o_eff) ** (1 / N) < 0.005}
    for label, kind, (o_new, s_new, clamped) in (("SSS", "unfiltered", raw),
                                                 ("raw, fast path", "fast path", raw),
                                                 ("effective, fast path", "fast path", eff),
                                                 ("raw, exact", "exact", raw),
                                                 ("effective, exact", "exact", eff)):
        old = o * kernel(kind, u, scales, nu)
        new = 1 - (1 - float(o_new) * kernel(kind, u, tuple(s_new[0].tolist()), nu)) ** N
        row[label] = (float((new - old).abs().max()), bool(clamped))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    labels = ["SSS", "raw, fast path", "effective, fast path", "raw, exact", "effective, exact"]
    lines = ["| shape (px) | ν | o | N | o_eff | " + " | ".join(labels) + " |",
             "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in labels) + " |"]
    worst = {k: 0.0 for k in labels}
    for shape, nu, o, N in itertools.product(SHAPES, (1, 100, 3000), (0.3, 0.9), (2, 4)):
        row = drift(shape, nu, o, N)
        for k in labels:
            worst[k] = max(worst[k], row[k][0])
        lines.append(f"| {shape} {SHAPES[shape]} | {nu} | {o} | {N} | {row['o_eff']:.3f}{'†' if row['floored'] else ''} | "
                     + " | ".join(f"{row[k][0]:.3f}" + ("*" if row[k][1] else "") for k in labels) + " |")
        print(lines[-1], flush=True)
    lines.append("| **worst** | | | | | " + " | ".join(f"**{worst[k]:.3f}**" for k in labels) + " |")
    lines.append("")
    lines.append("Max |alpha_new − alpha_old| over the grid. * = the written-back raw opacity hit "
                 "the |o| < 1 clamp. † = the relocated effective opacity is below SSS's 0.005 floor, so the "
                 "effective rule raises it to 0.005.")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
