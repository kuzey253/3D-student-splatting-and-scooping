# M3 report: 3D smoothing filter (Option A)

2026-10-27. Colab G4: an RTX PRO 6000 Blackwell (sm_120); torch 2.11.0+cu130. Built from `d4e9ee3`. The work took two VM sessions (15:43–15:52 and 15:52–16:34) and cost 6.36 compute units at 8.9 units/h. About 1.8 of those were the VM sitting idle for 12 minutes after the run finished, before I stopped it.

## What changed (NOTES.md 30–33)

- **Flags:** `--mip_filter_3d` (default off) and `--mip_filter_3d_variance` (default 0.2).
- **`NTModel` getters:**
  - filtered scales, √(s² + σ3²);
  - filtered opacity, o·√(∏s² / ∏(s² + σ3²)), which is Option A in 3D.

  The renderer uses both when the flag is on. The CUDA rasterizer is unchanged.
- **σ3:** comes from Mip-Splatting's computation (`mip_sss.filter3d`). It is recomputed before training, after every relocation, and every 100 iterations otherwise. It is saved in the PLY.

## Checks (`scripts/colab/m3_checks.py`), all passed

| Check | Result |
| --- | --- |
| Getters vs float64 formulas (effective scale and opacity) | ≤ 2.2e-7 relative |
| Filtered render vs unfiltered render of the folded model: image | ≤ 1.3e-5 |
| Same: gradients for position, log-scale, rotation, raw opacity and ν, via the float64 chain rule through the getters | 10/12 cases ≤ 1e-4 (mostly ~1e-6); 2 cases within max(1e-4, 2 × measured ±1-ulp sensitivity) |
| Discrete float32 decision flips | 0 in the real run (1 in an earlier dry run) |
| σ3 on real cameras (`train`: 263 cameras, 182,686 SfM points) vs an independent computation from the 3DGS matrices | ≤ 2.7e-7 at float32 precision; 1 point flips across a camera's visibility margin in float64 |
| Flag off with a stored σ3 | image bitwise identical |
| `render.py` + `metrics.py` vs training-time evaluation (proves the flags reach evaluation) | equal to about 1e-6 dB, all three runs |

**Two upstream findings came out of getting these checks right:**
- **SSS's float32 scale/rotation backward is noisy for needle-shaped components** (NOTES.md 34, 36). Against float64 it is off by up to 4.7e-4 with the filter off, and a 1–2 ulp change in a component's scales moves its rotation gradient by about 1e-3 of its size.
- **That is why the wiring check works the way it does.** It tests M3's own code exactly, and compares gradients against the measured noise of the identical computation, as M2 did with atomic-add noise. The tolerance itself was never relaxed.

## Training sanity check: `train`, 7000 iterations, training resolution, one run each

| Run | PSNR | SSIM | LPIPS | FPS | Peak GPU memory | Train time | ν mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| filter off (`train_s0`) | 19.67 | 0.705 | 0.337 | 141 | 7.7 GB | 225 s | 3650 |
| 3D filter (`train_3d`) | 19.77 | 0.705 | 0.341 | 138 | 7.7 GB | 231 s | 3593 |
| 2D + 3D filters, s = 0.1 (`train_s01_3d`) | 19.66 | 0.703 | 0.343 | 139 | 7.8 GB | 232 s | 3520 |

- **No measurable quality effect at the training resolution.** The same filter-off configuration scored 19.76 in the M2 session, so the run-to-run spread is about 0.1 dB, and all three variants fall inside it. That is the expected outcome: the filters are meant to help at other scales (M6).
- **Small cost.** Recomputing σ3 every 100 iterations adds about 3% training time, and rendering is about 2% slower.

Outputs are in `run_20261027_160410/`; the four dry runs are in `dry_*`.
