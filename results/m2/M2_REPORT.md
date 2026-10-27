# M2 report: 2D Mip filter (Option A) in SSS's CUDA rasterizer

2026-10-27. Colab G4: an RTX PRO 6000 Blackwell (102 GB, sm_120); Python 3.13.16, torch 2.11.0+cu130, CUDA 13.0. Built from `208ca1c` plus the fixes committed in `d591b6d`. The session ran 15:10:46–15:35:08 at 8.9 units/h and cost 3.55 compute units, both dry runs included.

## What changed

- **Flag:** `--mip_kernel_size s` (pipeline parameter; default 0 = off).
- **Forward (`forward.cu`):** Σ′ = Σ + sI in preprocess, and the signed opacity is multiplied by √(|Σ|/|Σ′|). The radius comes from SSS's own rule applied to Σ′.
- **Backward (`backward.cu`):** the coefficient's gradient goes into the 2D covariance, and the raw-opacity gradient is the effective-opacity gradient × coef. The backward pass now receives the raw opacities.
- **Flag-off guarantee:** every change is guarded by `s > 0`, so with the flag off the code computes exactly what SSS does.
- Details in NOTES.md 25–29.

## Checks (`scripts/colab/m2_checks.py`), all passed

| Check | Result | Tolerance |
| --- | --- | --- |
| CUDA vs float64 reference, 6 random scenes × s ∈ {0, 0.1, 0.3}: image | worst 7.9e-5 | 1e-4 |
| Same, gradients for means, 3D covariance, colours, opacity, ν | worst 6.3e-5 | 1e-4 |
| No NaNs (scenes include negative opacity and ν from 1 to 5000) | none | — |
| Filter off vs pre-M2 build (`19b1b53`): image and radii | bitwise identical | exact |
| Same, gradients | 1.3e-5 (atomic-add noise between runs is 1.9e-5) | ≤ noise |

The first dry run failed the unfiltered covariance-gradient check, with errors up to 1.0e-3. The cause is upstream: SSS's backward divides by det² + 1e-7, which damps the gradients of sub-pixel components now that SSS has dropped 3DGS's dilation (NOTES.md 29). The reference now reproduces that epsilon, as it already did for SSS's ν gradient. The tolerance was not changed. The filter itself removes the effect.

## Training sanity check: Tanks & Temples `train`, 7000 iterations, training resolution, one run each

| Run | PSNR | SSIM | LPIPS | FPS | Peak GPU memory | Train time | ν mean | ν = 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| filter off (`train_s0`) | 19.76 | 0.705 | 0.338 | 144 | 6.7 GB | 224 s | 3632 | 2.0% |
| filter on, s = 0.1 (`train_s01`) | 19.61 | 0.704 | 0.339 | 145 | 6.8 GB | 223 s | 3602 | 2.1% |

- **Stable.** Training with the filter is stable and costs no speed or memory (Option A is one det ratio per component).
- **Quality gap unproven.** At the training resolution the filter scores 0.15 dB lower here. With one stochastic SGHMC run each and only 7000 iterations, that can't yet be separated from run-to-run noise.
- **What the filter is for.** Its purpose shows when rendering at other scales, which is M6's protocol. The local test `test_filter_recovers_subpixel_component_between_samples` shows the mechanism: a 0.17 px component that the unfiltered renderer drops entirely is rendered, with about its full mass, once filtered.

Outputs are in `run_20261027_152224/`, and the dry runs in `dry1_*` (the failed parity) and `dry2_*`.
