"""M1 quadrature study: error of Option B candidates against the exact filtered kernel.

For every candidate rule and every ν, reports the worst case over COV_CASES (anisotropic, from
sub-pixel to large) at s = S_MIP of: core relative error, tail relative error, max abs error,
mass error, and the max abs error of ∂T/∂ν (central differences of both sides, normalized by
max |∂T_ref/∂ν|). Also reports max abs error at s = 0 against SSS's closed-form kernel.

Usage: python scripts/quad_study.py [--quick]
Writes results/m1/quad_study.csv and results/m1/quad_study.md.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mip_sss import QuadConfig, mass_option_b, option_b, sss_kernel  # noqa: E402
from mip_sss.kernels import det2  # noqa: E402
from mip_sss.reference import lambda_integral  # noqa: E402
from mip_sss.testing import (COV_CASES, NUS_EXTENDED, NUS_README, S_MIP, errors,  # noqa: E402
                             ray_points, rot_cov)

torch.set_default_dtype(torch.float64)
OUT = ROOT / "results" / "m1"


def candidates(quick):
    yield "README as written (fixed, K=8, Gaussian above ν=30)", QuadConfig(
        "fixed", 8, nu_fallback=30.0, fallback="gaussian")
    for K in ((8, 16) if quick else (4, 8, 12, 16, 32)):
        for control in ("none", "A", "A_prime"):
            cv = f" cv={control}"
            yield f"fixed[-12,3] K={K}{cv}", QuadConfig("fixed", K, control=control)
            yield f"adaptive K={K}{cv}", QuadConfig("adaptive", K, control=control)
            yield f"adaptive_tail τ=1e-4 K={K}{cv}", QuadConfig(
                "adaptive_tail", K, left=4.0, right=4.0, tail_tau=1e-4, control=control)
            yield f"adaptive_tail τ=1/255 K={K}{cv}", QuadConfig(
                "adaptive_tail", K, left=3.0, right=3.0, tail_tau=1 / 255, control=control)
            yield f"gauss K={K}{cv}", QuadConfig("gauss", K, control=control)
            for tail, left in ((K // 4, 1.0), (K // 2, 1.0), (K // 4, 2.0)):
                if tail:
                    yield f"gauss2 tail={tail} split=c-{left:g}w K={K}{cv}", QuadConfig(
                        "gauss", K, gauss_tail_nodes=tail, left=left, control=control)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer K values, for a fast look")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cases = {}
    for nu in NUS_EXTENDED:
        for i, (eigs, th) in enumerate(COV_CASES):
            cov = rot_cov(eigs, th)[None]
            u = ray_points(cov[0], nu, S_MIP)
            mu = torch.zeros(1, 2)
            h = 1e-3 * nu
            nus = [torch.tensor([v]) for v in (nu, nu - h, nu + h)]
            ref, ref_m, ref_p = (lambda_integral(u, mu, cov, v, S_MIP)[0] for v in nus)
            cases[nu, i] = dict(u=u, mu=mu, cov=cov, nus=nus, ref=ref, h=h,
                                dref=(ref_p - ref_m) / (2 * h),
                                exact0=sss_kernel(u, mu, cov, nus[0])[0])
    print(f"references: {time.time() - t0:.0f}s")

    rows = []
    for name, cfg in candidates(args.quick):
        t1 = time.time()
        for nu in NUS_EXTENDED:
            worst = dict(core_rel=0.0, core255_rel=0.0, tail_rel=0.0, max_abs=0.0, mass_rel=0.0,
                         dnu_rel=0.0, s0_abs=0.0)
            for i in range(len(COV_CASES)):
                c = cases[nu, i]
                nu_t, nu_m, nu_p = c["nus"]
                approx = option_b(c["u"], c["mu"], c["cov"], nu_t, S_MIP, cfg)[0]
                e = errors(approx, c["ref"])
                e["core255_rel"] = errors(approx, c["ref"], core=1 / 255)["core_rel"]
                d_approx = (option_b(c["u"], c["mu"], c["cov"], nu_p, S_MIP, cfg)[0]
                            - option_b(c["u"], c["mu"], c["cov"], nu_m, S_MIP, cfg)[0]) / (2 * c["h"])
                e["dnu_rel"] = float((d_approx - c["dref"]).abs().max() / c["dref"].abs().max())
                e["mass_rel"] = 0.0  # every closed-form fallback preserves mass exactly
                if not math.isfinite(cfg.nu_fallback) or nu <= cfg.nu_fallback:
                    mass = mass_option_b(c["cov"], nu_t, cfg)
                    e["mass_rel"] = float((mass / (2 * math.pi * det2(c["cov"]).sqrt()) - 1).abs())
                s0 = option_b(c["u"], c["mu"], c["cov"], nu_t, 0.0, cfg)[0]
                e["s0_abs"] = float((s0 - c["exact0"]).abs().max())
                for k in worst:
                    worst[k] = max(worst[k], e[k])
            rows.append(dict(candidate=name, nu=nu, **worst))
        print(f"{name}: {time.time() - t1:.1f}s")

    keys = ["candidate", "nu", "core_rel", "core255_rel", "tail_rel", "max_abs", "mass_rel",
            "dnu_rel", "s0_abs"]
    with open(OUT / "quad_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    write_markdown(rows)
    print(f"total {time.time() - t0:.0f}s; wrote {OUT / 'quad_study.csv'} and quad_study.md")


def write_markdown(rows):
    by = {}
    for r in rows:
        by.setdefault(r["candidate"], []).append(r)

    def worst(rs, key, nus):
        return max(r[key] for r in rs if r["nu"] in nus)

    lines = [
        "# M1 quadrature study",
        "",
        f"Generated by `scripts/quad_study.py`. s = {S_MIP} px², worst case over {len(COV_CASES)} "
        "anisotropic covariances (sub-pixel to large). Reference: λ-integral, cross-checked against "
        "FFT convolution in the tests. Core: T ≥ 1e-3; renderable core: T ≥ 1/255 (alpha below "
        "1/255 is skipped at |o| ≤ 1); tail: 1e-6 ≤ T < 1e-3. README target: core rel ≤ 1e-3 for "
        "every ν below 30. ∂/∂ν rel: max abs error of ∂T/∂ν over max |∂T_ref/∂ν|.",
        "",
        "## Summary per candidate",
        "",
        "| candidate | core rel, ν<30 | renderable core rel, ν<30 | core rel, all ν | "
        "tail rel, all ν | max abs, all ν | mass rel | ∂/∂ν rel | max abs at s=0 | "
        "meets target (ν<30) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    readme = set(NUS_README)
    allnu = set(NUS_EXTENDED)
    for name, rs in by.items():
        c30 = worst(rs, "core_rel", readme)
        lines.append(
            f"| {name} | {c30:.1e} | {worst(rs, 'core255_rel', readme):.1e} | "
            f"{worst(rs, 'core_rel', allnu):.1e} | "
            f"{worst(rs, 'tail_rel', allnu):.1e} | {worst(rs, 'max_abs', allnu):.1e} | "
            f"{worst(rs, 'mass_rel', allnu):.1e} | {worst(rs, 'dnu_rel', allnu):.1e} | "
            f"{worst(rs, 's0_abs', allnu):.1e} | {'yes' if c30 <= 1e-3 else 'no'} |")
    lines += ["", "## Core relative error by ν", "",
              "| candidate | " + " | ".join(f"ν={v:g}" for v in NUS_EXTENDED) + " |",
              "| --- |" + " --- |" * len(NUS_EXTENDED)]
    for name, rs in by.items():
        vals = {r["nu"]: r["core_rel"] for r in rs}
        lines.append(f"| {name} | " + " | ".join(f"{vals[v]:.1e}" for v in NUS_EXTENDED) + " |")
    (OUT / "quad_study.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
