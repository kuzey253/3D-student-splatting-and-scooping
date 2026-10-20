# M1 report: PyTorch reference and quadrature study

2026-10-20. Branch `mip-sss` (uncommitted), on SSS commit `33a70d3`. CPU only (Intel i7-8557U, torch 2.2.2, float64). Every number here comes from the scripts listed at the end.

## What exists now

| Path | Contents |
| --- | --- |
| `mip_sss/kernels.py` | SSS kernel, Mip-Splatting `G_mip`, Option A and A′, Option B (2D, and the 3D-filter path), the hybrid, exact mass, a culling-radius bound, and a port of SSS's radius rule |
| `mip_sss/quadrature.py` | Node rules (fixed, README-adaptive, adaptive_tail, Gauss, two-segment Gauss), control variates, large-ν fallback, and the closed-form ν gradient for fixed grids |
| `mip_sss/reference.py` | Two independent brute-force references: λ-integral and FFT convolution |
| `mip_sss/filter3d.py` | Port of Mip-Splatting's `compute_3D_filter` (f̂, σ3) |
| `mip_sss/render.py` | SSS's compositing loop, including negative alpha |
| `tests/` | At the time of this comparison: 565 tests over five Option B candidates. After the gate: 448 tests for the chosen design (see the end of this report) |
| `scripts/` | Study scripts and the pre-study |

The package is standalone: it doesn't import SSS modules and needs no CUDA.

## Test status

Full matrix: [test_matrix.md](test_matrix.md). Totals: 420 passed, 135 failed, 10 skipped. Every failure is an accuracy, mass or reduces-to-SSS check for a candidate that is too coarse. Nothing that is independent of the candidate fails.

- **The references agree.** The λ-integral matches SSS's closed form at s = 0 to < 1e-10, for all ν and covariances. FFT convolution reproduces `G_mip` exactly, and matches the λ-integral to < 1e-9 relative wherever T ≥ 1e-6. This independently confirms the scale-mixture derivation.
- **Only `tail32_cvA` passes every README test.** That is the adaptive grid with τ = 1e-4, K = 32, and the Option A control variate.
- **The README's own spec (`readme`) fails accuracy (0/35), mass (0/7) and reduces-to-SSS (0/25).**
- **Every candidate passes** gradcheck, unit peak, reduces-to-Mip-Splatting, negative-opacity compositing, and the radius bound.
- **Confirmed by tests:**
  - Option A's tails are too heavy for sub-pixel components, as the README predicted.
  - SSS's ν-gradient bug.
  - The radius port and the Mip-Splatting filter port.
  - The 3D-filter limits.

## Quadrature

Source: [quad_study.md](quad_study.md), 121 candidates × 12 values of ν × 5 anisotropic covariances, with s = 0.1. Worst cases over ν ∈ [1, 10⁴]:

| Candidate | exps / pixel | core rel, ν<30 (T ≥ 1e-3) | renderable core rel (T ≥ 1/255) | max abs | mass rel | max abs, s = 0 |
| --- | --- | --- | --- | --- | --- | --- |
| README (fixed, K=8, Gaussian > 30) | 8 | 4.1 | 1.5 | 2.6e-1 | 4.4e-1 | 2.6e-1 |
| adaptive_tail τ=1e-4, K=32, cv=A | 64 + 1 pow | **1.4e-4** | 3.5e-5 | 2.5e-7 | 0 | 4e-16 |
| adaptive_tail τ=1/255, K=12, cv=A | 24 + 1 pow | 1.6e-2 | 4.3e-3 | **2.4e-4** | 0 | 3e-16 |
| two-segment Gauss, K=8 (4 tail), cv=A | 16 + 1 pow | 2.2e-1 | 3.6e-2 | **3.3e-4** | 0 | 3e-16 |
| Gauss, K=8 | 8 | 1.8e-1 | 1.1e-1 | 8.1e-4 | 9.7e-5 | 8.3e-4 |

- **K = 8 cannot meet the README's relative target with any rule tried.** The best relative error at T ≥ 1e-3 (ν < 30) is 3.6e-2 at K = 8, 1.6e-2 at K = 12, 1.8e-3 at K = 16 and 1.4e-4 at K = 32.
- **The error sits at small ν (1–4), at the edge of the core.** For ν ≥ 8 the Gauss rule is near machine precision even at K = 8.
- **In absolute terms, K = 8–12 with a control variate is already 12–16× below one 8-bit step** (3.9e-3).
- **The control variate is what makes small K pass the limit tests.** It makes the s = 0 limit (reduces to SSS) and mass exact for any K. It doubles the exps per node.
- **The README's fallback breaks training.** Above ν = 30 it gives ∂T/∂ν = 0, and the render jumps at ν = 30. SSS initialises every ν at 100, so ν would never train.

## Fast path (hybrid)

Source: [shortcut_study.md](shortcut_study.md). This is the smallest size ratio ρ = λ_min/s from which a closed form stays within ε of the exact kernel:

| ν | Option A, ε = 1e-3 | Option A, ε = 1/255 | Option A′, ε = 1e-3 | Option A′, ε = 1/255 |
| --- | --- | --- | --- | --- |
| 1 | never | 1000 | 1000 | 300 |
| 4 | 1000 | 300 | 300 | 100 |
| 30 | 100 | 30 | 30 | 10 |
| 100 | 30 | any | 10 | any |
| ≥ 300 | any | any | any | any |

The README's single threshold of ρ > 10 is too loose below ν ≈ 50 at ε = 1/255, and below ν ≈ 300 at ε = 1e-3. A ν-aware threshold table is cheap to evaluate per component.

## Culling radius and cost

Source: [radius_study.md](radius_study.md), at |o| = 1.

- **SSS's own rule already cuts visible alpha** for components a pixel or larger at ν ≥ 4. This is inherited behaviour, like 3DGS's 3σ cut.
- **The conservative Option B bound covers up to 7× SSS's tile area** for components of a pixel or more; the worst case is the 32-node rule at ν ≈ 2. It covers less than SSS's area for sub-pixel components, and at ν ∈ (1, 2), where SSS's reversed interpolation makes its own radius very large.
- **A tighter radius is possible.** Solving the node sum along the major axis, a few Newton steps in preprocessing, would bring this close to the true radius. That is an M4 detail.

Estimated cost: the per-pixel work of a component on the Option B path is about (exps per pixel) × (tile area). How many components take that path depends on the ν and λ_min histograms of a trained model, which is M0's job.

## Recommendation for the gate

1. **Accuracy criterion.** Adopt max abs error ≤ 1e-3 together with exact limits and exact mass, and keep reporting relative error. With the README's strict relative criterion only K = 32 passes, at 64 exps per pixel.
2. **Option B rule.** `adaptive_tail τ=1/255, K=12, cv=A`:
   - max abs 2.4e-4, exact mass and s = 0 limit;
   - its nodes are closed-form in ν (digamma and trigamma), so the ν gradient flows without tabulation;
   - 24 exps + 1 pow per pixel.

   If you choose the strict criterion instead, the same family at K = 32 with τ = 1e-4.
3. **Large ν.** Drop the Gaussian fallback. Use Option A as the fast path for ν ≥ 300, where it is within 1e-3 at every size and keeps the ν gradient.
4. **Fast path.** Option A (matching the control variate) with the ν-aware ρ table at ε = 1e-3.
5. **SSS baseline.** Add the `exact_nu_grad` baseline as planned.

## After the gate (2026-10-20)

You adopted the recommendation above. The design is pinned in `mip_sss/design.py`:

- **Criterion:** max alpha error ≤ 1e-3 over the whole domain, with exact limits and exact 2D mass.
- **Option B:** `adaptive_tail`, τ = 1/255, K = 12, with the Option A control variate.
- **Fast path:** Option A wherever the ν-aware table allows it. The table is regenerated by `scripts/fastpath_table.py` with aspect ratios up to 1:64 and a 10% margin, so Option A covers every size from ν = 400, not 300. There is no Gaussian fallback, so ν always has a gradient.
- **3D path:** now has the control variate too, with Option A in 3D as the closed form. Its fast path needs both the 3D ratio λ_min(Σ3)/σ3² and the projected 2D ratio to pass the table.

The suite was rewritten for the design and now passes **448/448** ([test_matrix.md](test_matrix.md)):

- the limits are exact to 1e-12;
- mass is verified by brute-force integration to 1e-6;
- max abs ≤ 1e-3 holds against FFT and the λ-integral, for all ν in [1, 10⁴], 30 random fast-path components (aspect up to 1:64, sizes up to 1000× the threshold), and the 3D path including thin splats;
- gradcheck covers both branches, and ∂T/∂ν is non-zero everywhere.

The tables above compare candidates and are unchanged. The radius study now has a `design` column.

M0 tooling, `scripts/m0_component_stats.py`, is ready and tested. It reads a trained model and reports the ν and ρ histograms, the share of components on the Option B path at 1/1 to 1/8 scale, and an estimated cost ratio against SSS.

## Reproduce

Run from `sss/`:

```
python3 -m pytest tests -q
python3 scripts/test_matrix.py      # results/m1/test_matrix.md
python3 scripts/fastpath_table.py   # mip_sss/fastpath_table.py, ~15 min
python3 scripts/quad_study.py       # results/m1/quad_study.{md,csv}, ~1 min
python3 scripts/shortcut_study.py   # results/m1/shortcut_study.{md,csv}
python3 scripts/radius_study.py     # results/m1/radius_study.md
python3 scripts/prestudy/quad_check.py      # pre-study tables (results/prestudy/)
python3 scripts/prestudy/shortcut_check.py
```
