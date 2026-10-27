# M4 report: Option B (scale-mixture quadrature) in CUDA

2026-10-27. Colab G4: RTX PRO 6000 Blackwell (sm_120), torch 2.11.0+cu130, CUDA 13.0.

There were four VM sessions (units from the account balance before and after each):

| Session | Time | Build | Units |
| --- | --- | --- | --- |
| 1 | 17:03–17:45 | `ae694c0` | 6.16 |
| 2 | 18:28–18:42 | `3c890e6` | 1.73 |
| 3 | 18:54–19:09 | `5f0fe20` | 2.22 |
| 4 | 19:17–19:38 | `d42e77c` (= `850ea3f` + this report) | 2.55 |
| **Total** | | | **12.66** (at 8.9 units/h) |

## Status

- **Option B in CUDA is written and matches the float64 reference on the GPU.** All checks passed in all three sessions.
- **Training with the 3D filter on completed 7000 iterations.** That run was on the first build (session 1).
- **Training without the 3D filter (σ3 = 0) hit NaN gradients in all four sessions,** at iterations 3860, 4360, 2390 and 5700. The guard that fires is `grad_means3D`, the first one checked.
- **The cause is not found.** Three float32 failure modes in the Option B code were found and fixed (NOTES.md 43–45), and each is real: it is reproduced and cleared by stress-testing the exact device code compiled as host C++. But the GPU failure persisted through all three fixes, at no consistent iteration. So the cause lies outside what the stress tests exercise.
- **Next is a diagnostic run that captures the failing state** (see Open), not another fix from a hypothesis.

## What was built (NOTES.md 37–42)

- **Flag:** `--mip_option_b`, a pipeline parameter, default off.
  - With `--mip_filter_3d`, σ3 goes into the quadrature per node, cov2_k = Σ2/λ_k + σ3²TTᵀ, instead of the M3 scale change.
  - That is required: the scale-change shortcut leaves an alpha error of 0.17 on S1, against 3.2e-4 per node.
- **Two modes per component:**
  - **Fast path** (the ν-aware ρ table): renders hp·t(covp) through SSS's own kernel. This is Option A in 2D and 3D.
  - **Option B:** the K = 12 adaptive_tail quadrature with the Option A control variate, T = hp·t(covp) + Σ_k (g_k − h_k).
- **Option B data:**
  - Option B components get a 203-float slot (filled after a scan of the per-component flags) and an alpha-safe radius.
  - Their backward pass uses 9 per-component accumulators, chained to dL/dcov3D and dL/dmean3D.
- **All per-component device code is in `mip_optionb.h`.**
- **It was checked without a GPU in two ways:**
  - against the hand-derived float64 staging in `mip_sss/optionb_staged.py`, whose backward is checked against autograd;
  - by compiling the header as host C++ in `tests/test_optionb_header.py`, which compares it with the staging and stress-tests degenerate components.
- **With the flag off, nothing new is allocated or launched.**

## GPU checks (session 3, build `5f0fe20`; sessions 1 and 2 passed the same checks)

| Check | Result |
| --- | --- |
| Flag off vs the M3 build (`7183bc4`) | image bitwise equal, radii equal, gradients within 1.1e-5 (atomic-add noise 9.2e-6) |
| M2 suite (Option A in 2D vs float64, 18 cases) | pass, worst 7.9e-5 |
| M3 suite (3D filter wiring, 12 cases; filter values on real cameras) | pass |
| Option B CUDA vs float64 `rasterize_option_b`: 12 scenes, σ3 off and on, 22–29 Option B components each | 12/12 pass, 0 float32 flips; worst relative error: image 7.9e-5, means 7.3e-5, cov3D 4.9e-5, colors 8.4e-5, opacity 8.2e-5, ν 1.1e-5 (tolerance 1e-4) |
| SSS `render()` in Option B mode on an `NTModel` (scales/rotations path) | image ≤ 7.2e-5 in 4/4, all gradients finite |

**Known float32 limit (NOTES.md 41).** The ν gradient of an Option B component is off by up to 6e-3 at ν = 150. The staged Python run in float32 shows the same error, so it comes from cancellation in the formula, not from the CUDA code. The checks' Option B components have ν ≤ 30.

## Training: `train`, 7000 iterations, session 1 (build `ae694c0`, before the fixes in NOTES.md 43–45)

| Run | PSNR | SSIM | LPIPS | FPS | Peak GPU memory | Train time | ν median | ν at 1 | Option B share (visible pairs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3 config: 2D + 3D filters, s = 0.1 (`train_s01_3d`) | 19.68 | 0.702 | 0.342 | 130 | 7.0 GB | 239 s | 3491 | 3.0% | 2.9% (estimated) |
| Option B + 3D filter (`train_s01_3d_ob`) | 19.69 | 0.701 | 0.346 | 26 | 9.2 GB | 542 s | 2592 | 31.8% | 34.9% |
| Option B, σ3 = 0 (`train_s01_ob`) | NaN gradients at iterations 3860 / 4360 / 2390 (sessions 1–3) | | | | | | | | |

- **Quality is unchanged at the training resolution.** That is the expected result, as in M3; the anti-aliasing is meant to show at other scales (M6).
- **ν drifts down under Option B.** 31.8% of components end at the lower bound ν = 1, against 3.0% without Option B.
  - The fast path keeps SSS's ν gradient, which drops a term (NOTES.md 13), while Option B components get the exact gradient. The difference between the two may drive the drift.
  - This needs a look in M5 before anything is concluded.
  - Because of the drift, the Option B share rises from about 3% to 35% of visible pairs.
- **Speed and memory follow from that share.** Training is 2.3× slower, rendering 5× slower, and peak memory 2.2 GB higher. The 1000-iteration dry run had suggested only 1.3× slower training.
- **The fixes since this run change arithmetic only, not the math.** This run should still be repeated on the final build, together with the σ3 = 0 run.

## The NaN at σ3 = 0: three causes, all fixed

| Fix | Cause | Change | Verified |
| --- | --- | --- | --- |
| `3c890e6` (NOTES.md 43) | ½(1/D − 1/det3f) gives ∞ − ∞ for flat components; per-node log det from rounded entries | D gradient carried as D·∂/∂D; 3D terms through e and L3 = D + e3, exactly zero at σ3 = 0 | host test; GPU session 2 still failed |
| `5f0fe20` (NOTES.md 44) | a·c − b² cancels for nearly rank-1 footprints (Σ2/λ_k up to ~1e7 px²) | determinants built from non-negative pieces: det(A + B) = det A + det B + tr(adj(A)B), det(A + sI) = det A + s tr A + s² | host stress (20 of 8,348 → 0); GPU session 3 still failed |
| `850ea3f` (NOTES.md 45) | tiny, invisible components: det Σ2f is denormal, 1/det = ∞, and the zero incoming gradient gives 0·∞ = NaN; the 3D filter's P term prevents it | inverses divide entry by entry, as M2 does; zero incoming gradients return exact zeros | host stress with the render's alpha test: the session-3 header fails 313 of 3,449 tiny components, this one 0 of 14,436 tiny ones; GPU session 4 still failed (iteration 5700) |

All three are float32 hazards in my code, and none is a bug in SSS. The host test covers ordinary, extreme and tiny components through the whole per-component path, with the alpha test the render applies. There are 537 local tests in total, all passing.

**What the host tests do not exercise,** so where the remaining cause must be:
- the render kernels' per-pixel compositing of Option B values: forward and backward, many components per pixel, |alpha| clamped at 0.99, negative opacities;
- the atomic sums of per-pixel accumulators over a whole image;
- the mean2D gradient path;
- the interplay with training: relocation, the opacity and scale regularizers, and SSS's ν handling, as ν drifts to 1 under Option B.

The only sign so far is which guard fires: a NaN in the mean gradient, at iterations 2390–5700, only at σ3 = 0. Session 4 (build `d42e77c`) passed every check again: flag-off parity against M3, M2, M3 and M4.

## Open

1. **Capture the failing state on the G4 instead of guessing.**
   - At the first non-finite gradient, save the camera, the iteration, whether the rendered image and loss were finite, and the gradient tensors.
   - For every component with a non-finite gradient, also save its parameters (mean, scales, rotation, opacity, ν, σ3, radius, mode, slot).
   - Then replay those components locally, in the host harness and the float64 reference.
   - Cost: one `train_s01_ob` run until it fails, about 10–15 minutes including the build, ~2 units.
   - `train_s01_3d_ob` was not re-run on the final build. It did not fail in session 1.
2. **ν drift under Option B** (31.8% at ν = 1): understand it before M5's comparisons.
3. **The float32 ν gradient for Option B components with ν in (30, 400)** (NOTES.md 41).

Outputs are in `results/m4/`:

| Directory | Contents |
| --- | --- |
| `dry_20261027_170330`, `dry_20261027_171218` | the two session-1 dry runs that stopped on check-script bugs |
| `dry_20261027_171508`, `run_20261027_172201` | session 1's passing dry run and real run |
| `dry_20261027_182804`, `run_20261027_183503` | session 2 |
| `dry_20261027_185410`, `run_20261027_190143` | session 3 |
| `dry_20261027_191736`, `run_20261027_192502` | session 4 (its `train_s01_3d_ob` directory holds the dry-run files; the real run stopped at `train_s01_ob`) |
