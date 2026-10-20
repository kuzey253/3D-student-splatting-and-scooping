# M0 report: SSS baselines on a Colab A100

2026-10-20. Run `run_20261020_181819`, from commit `d0c2f06`. The command was `SESSION=a100 bash scripts/colab/run_m0.sh`, with the shipped configs and 40,000 iterations per scene. Outputs are in [run_20261020_181819/](run_20261020_181819/); the dry run is in [DRY_RUN.md](DRY_RUN.md).

## Machine and cost

| | |
| --- | --- |
| GPU and image | A100-SXM4-40GB; Python 3.13.15, torch 2.11.0+cu130, nvcc 13.0, driver 580.82.07. Same image as the dry run. The G4 was unavailable (503) earlier the same day. |
| VM up | 18:17:50 → 20:23:19, 2 h 05 min |
| Compute units | real run 10.35 (balance 290.92 → 280.57, at 5.3 units/h); M0 total including the dry run: 12.5 |
| Incident | At about 19:20 the CLI dropped its local record of the session while the VM kept running. It was re-attached with the runbook's `attach_session.py --keep-alive`; training never stopped. |

## Reproduction: matches the published SSS numbers

| Scene | Source | PSNR | SSIM | LPIPS | Components | Train time | FPS (render only) | Peak GPU memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (T&T, 980×545) | **ours** | **23.45** | **0.851** | **0.167** | 1.10M | 36.3 min | 82.2 | 7.3 GB |
| | SSS paper | 23.32 | 0.850 | 0.166 | — | 33 min* | 87* | — |
| garden (360, 1297×840) | **ours** | **28.07** | **0.883** | **0.091** | 3.00M | 62.1 min | 48.8 | 11.0 GB |
| | SSS paper | 28.09 | 0.882 | 0.009† | — | 45 min* | 71* | — |

The paper's numbers come from [arXiv 2503.10148v4](https://arxiv.org/abs/2503.10148), Tables 1–6 and 21, averaged over three runs on an RTX 4090. They were read from the paper's HTML tables.

- \* The paper's time and FPS are averages over the whole dataset (all scenes), on a 4090, so they are not directly comparable with one scene on an A100.
- † As printed. Every other method scores about 0.10 on garden, so this is almost certainly a typo.
- Our metrics come from SSS's own `render.py` and `metrics.py`. Training time includes loading the images.

## What the trained models look like (`scripts/m0_component_stats.py`)

| | train | garden |
| --- | --- | --- |
| ν between 1000 and 9999 | 97.3% | 95.9% |
| ν exactly 1 (the lower bound) | 1.4% (15,869) | 2.4% (72,072) |
| ν at the upper bound 10⁴ | 0.8% | 0.3% |
| ν anywhere in 1.01–300 | 163 components | 456 components |
| negative opacity | 22.7% | 14.4% |
| visible pairs on Option B, at 1/1, 1/2, 1/4 and 1/8 scale | 1.2% at every scale | 2.7% at every scale |
| sub-pixel pairs (λ_min < s), at 1/1 → 1/8 | 25% → 89% | 4% → 69% |
| estimated kernel-evaluation cost vs SSS, 1/1 → 1/8 | 1.78× → 1.08× | 2.12× → 1.44× |

- **Learned ν is all or nothing.** Components end up either effectively Gaussian (ν ≥ 1000) or pinned at the Cauchy bound ν = 1, with almost nothing in between. These models were trained with SSS's original ν gradient, which drops a term (NOTES.md 13). The planned "exact ν gradient" baseline will show whether that bias shapes this split.
- **The quadrature path is rare.** Only the ν = 1 components take Option B, 1–3% of visible pairs. Everything else takes Option A, which is within 1e-3 of exact at ν ≥ 400.
- **Zooming out makes most components sub-pixel.** At 1/8 scale, 89% (train) and 69% (garden) of visible pairs are sub-pixel, which is the regime the anti-aliasing filters target.
- **The estimated overhead comes mainly from the culling radius, not from Option B.** The design's radius goes out to where alpha falls below 1/255, while SSS's rule cuts large-ν components at 3σ, as 3DGS does. M4 has to choose between them. Using SSS's rule on the filtered covariance for Option A (README M4.4 allows "whichever is larger") would bring the cost close to SSS's. This is an estimate (transcendentals × tiles), not a measurement.
