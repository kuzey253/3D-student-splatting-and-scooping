# M5 report: SSS integration

2026-11-03. Colab G4: RTX PRO 6000 Blackwell (sm_120), torch 2.11.0+cu130, CUDA 13.0, Python 3.13.16 (the same image versions as M4 session 8, whose runs are compared below).

One VM session, 13:37–14:57, build `57b2255`: a dry run (13:37–13:48), then the real run (13:49–14:57). **12.89 units** (balance 443.52 → 430.63, at 10.05 units/h); the quote was ~7 (up to 8.5), and the overrun is the Option B + exact-ν run, which took 23 min instead of ~9. At 14:41 the CLI lost the session name while the VM kept running; it was re-attached with `attach_session.py --name g4 --require-large --keep-alive` and the runner's next status check succeeded. The VM was stopped, the keep-alive helper had exited, and the backend reported no active runtimes.

## Status

**M5 is done. The ν drift seen in M4 is explained: it is caused by the exact ν gradient, not by Option B's rendering or by relocation.** One design question is left for you (below).

- **README M5.1, relocation on the effective parameters:** `--mip_relocate_effective` (NOTES.md 49). Recycling and SSS's relocation formula use the 3D-filtered opacity; the raw opacity written back renders the relocated effective one. The scale rule stays SSS's (a logged deviation: writing back an effective scale is infeasible for any axis below ~0.36σ3).
- **README M5.2, filtered eigenvalue regularizer:** `--mip_reg_filtered` (NOTES.md 51).
- **README M5.3, SGHMC unchanged.**
- **PLAN M5, how far relocation drifts from the filtered kernel:** measured offline, `relocation_drift.md` (NOTES.md 49). For pixel-sized components, SSS's rule on the raw opacity drifts up to 0.104 in alpha (fast path) and 0.072 (exact kernel); the effective rule gives 0.007 and 0.010. Along the way: SSS's own relocation drifts up to 0.17 without any filter, mostly because it squares its scale factor (NOTES.md 50, an upstream quirk left as is).
- **Diagnostic for the M4 ν drift:** `--mip_exact_nu_grad` (NOTES.md 52) gives fast-path components the exact ν gradient that Option B components already get.

All flags default to off. With them off, plain SSS is unchanged: on the GPU, the image is bitwise equal to the M4 final build `1d9fa05`, radii are equal, and gradients agree to 5.4e-6 (atomic-add noise 5.2e-6). 548 local tests pass.

## GPU checks (dry run, `dry_20261103_133737`)

| Check | Result |
| --- | --- |
| Flag off vs the M4 build `1d9fa05` | image bitwise equal, radii equal, gradients within 5.4e-6 (noise 5.2e-6) |
| M2 suite (Option A in 2D vs float64, 18 cases) | pass, worst 7.9e-5 |
| M3 suite (3D filter wiring and values) | pass |
| M4 suite (Option B CUDA vs float64) | pass |
| M5: `exact_nu_grad` vs `raster_ref` with the exact ν gradient, s ∈ {0, 0.1} × 6 seeds, tolerance 1e-4 | 12/12 pass; image ≤ 7.9e-5, gradients ≤ 6.25e-5, ν gradient ≤ 8.6e-6. The flag changes the ν gradient by 1.5–2.2× (relative), so it is not a no-op |
| Four 1000-iteration trainings with the real run's flags | all completed; `mip_reloc_stats.json` written |

The real run's check files are byte-identical copies of the dry run's (the collect step picked them up from the VM), so they are not a second confirmation.

## Training: `train`, 7000 iterations

New runs (this session, build `57b2255`), with M3/M4 baselines on the same image versions:

| Run | Flags beyond s = 0.1 + 3D filter | PSNR | SSIM | LPIPS | FPS | Peak GPU memory | Train time | ν median | ν at 1 | Option B share (visible pairs, 1/1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3 config `train_s01_3d` (M4 session 1) | — | 19.68 | 0.702 | 0.342 | 130 | 7.0 GB | 239 s | 3491 | 3.0% | 2.9% (estimated) |
| `train_s01_3d_exnu` | exact ν gradient | 19.70 | 0.702 | 0.345 | 40 | 12.8 GB | 415 s | 1 | **50.1%** | 53.5% (estimated) |
| `train_s01_3d_m5` | effective relocation + filtered reg | 19.85 | 0.705 | 0.342 | 125 | 6.9 GB | 241 s | 3540 | 2.6% | 2.4% (estimated) |
| Option B `train_s01_3d_ob` (M4 session 8) | Option B | 19.85 | 0.703 | 0.344 | 28 | 8.9 GB | 531 s | 2590 | 31.5% | 34.4% |
| `train_s01_3d_ob_exnu` | Option B + exact ν gradient | 19.75 | 0.700 | 0.347 | 8 | 17.9 GB | 1376 s | 1 | **61.6%** | 65.3% |
| `train_s01_3d_ob_m5` | Option B + effective relocation + filtered reg | 19.80 | 0.702 | 0.345 | 26 | 9.3 GB | 541 s | 2660 | 30.9% | 33.7% |

All four new runs completed 7000 iterations with no NaN.

**ν drift: the exact ν gradient drives ν to 1.**
- With every component on the exact gradient, 50.1% end at ν = 1 without Option B and 61.6% with it, against 3.0% under SSS's gradient. In M4, only Option B components had the exact gradient, and 31.5% ended at ν = 1.
- Effective relocation and the filtered regularizer leave ν where it was (2.6% vs 3.0%; 30.9% vs 31.5% under Option B). So relocation is not the cause.
- **Reading:** SSS's ν gradient omits −½·ln(1 + b/ν)·G (NOTES.md 13). The true loss gradient, at least over 7000 iterations, prefers heavy tails; SSS's omission keeps ν high. ν = 1 is absorbing (Hardtanh), so components that reach it stay.
- **Quality at the training resolution does not change:** 19.70 dB with the exact gradient against 19.66–19.68 for the M3 configuration, and 19.75 against 19.69–19.85 under Option B.
- **The cost is large:** SSS sets the footprint radius to 9.9·sqrt(λmax) at ν = 1 against 3·sqrt(λmax) for large ν (`forward.cu`), about 11× the area. Rendering drops 3.3× (130 → 40 FPS) and peak memory rises by 5.8 GB. With Option B on top, rendering drops to 8 FPS and training takes 23 min.

**Effective relocation + filtered regularizer (README M5.1–2).**
- No measurable change in ν, speed or memory.
- PSNR is 0.17–0.19 dB above the M3 configuration (19.85 vs 19.66–19.68), SSIM 0.705 vs 0.702–0.703, LPIPS equal. That is one run each, and the same Option B configuration differed by 0.16 dB between M4 sessions 1 and 8 (arithmetic fixes only in between), so the difference is not resolved.
- Under Option B, no difference (19.80 vs 19.69–19.85).
- Write-back clamp: 1 of 3,385,573 relocations in each run hit |o| < 1. The relocation count is the same in both runs because recycling is capped at 5% of P and growth is +5% per step up to the cap, so it is deterministic once more than 5% are dead; it says nothing about how many died.
- As in M3/M4, the anti-aliasing (and any relocation benefit) is meant to show at other scales (M6).

## Open: which ν gradient should Option B components get? (your call)

Option B components get the exact ν gradient, the fast path gets SSS's. That mix is what made 31.5% of components drift to ν = 1 in M4. The drift raises the Option B share from ~3% to 34% and widens footprints, which is a large part of why Option B renders 4–5× slower (not separated from the quadrature's own cost).

- **Keep as is:** Option B stays the exact derivative of what it renders. The cost is the drift and the speed.
- **Make Option B match SSS's ν gradient:** expected to bring the drift back to ~3% and the Option B share with it. The term SSS omits is defined for the t kernel; for the quadrature it needs a derivation of the matching term per node, and a GPU check and run.
- **Leave it to M6:** decide once the multi-scale results show whether heavy tails help anti-aliasing.

## Commands

Offline (CPU):

```bash
python3 scripts/m5_relocation_drift.py --out results/m5/relocation_drift.md
python3 -m pytest tests
```

GPU (Colab G4, `colab new --gpu G4 --session g4`), dry run then real run:

```bash
SESSION=g4 DRY=1 PARITY=1 OLD_REF=1d9fa05 M3CHECKS=1 M4CHECKS=1 M5CHECKS=1 RUNS="$RUNS" bash scripts/colab/run_m0.sh
SESSION=g4 RUNS="$RUNS" bash scripts/colab/run_m0.sh
```

with

```bash
RUNS="train_s01_3d_exnu:train:iterations=7000,mip_kernel_size=0.1,mip_filter_3d=true,mip_exact_nu_grad=true \
train_s01_3d_ob_exnu:train:iterations=7000,mip_kernel_size=0.1,mip_filter_3d=true,mip_option_b=true,mip_exact_nu_grad=true \
train_s01_3d_m5:train:iterations=7000,mip_kernel_size=0.1,mip_filter_3d=true,mip_relocate_effective=true,mip_reg_filtered=true \
train_s01_3d_ob_m5:train:iterations=7000,mip_kernel_size=0.1,mip_filter_3d=true,mip_option_b=true,mip_relocate_effective=true,mip_reg_filtered=true"
```

The runner trains, renders the test views, computes metrics and FPS, and writes `summary.json` and `stats/component_stats.md` (`scripts/m0_component_stats.py`) per run. The baselines are `results/m4/run_20261027_172201/train_s01_3d` and `results/m4/run_20261102_141838/train_s01_3d_ob`.

Outputs are in `results/m5/`:

| Path | Contents |
| --- | --- |
| `relocation_drift.md` | the offline drift table |
| `dry_20261103_133737` | the dry run: all checks, four 1000-iteration trainings |
| `run_20261103_134858` | the four 7000-iteration trainings |
