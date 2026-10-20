# M0 dry run, 2026-10-20

`SESSION=a100 DRY=1 bash scripts/colab/run_m0.sh`, from commit `dde8491` plus the uncommitted build fixes and runner (`dry_20261020_162208/GIT_STATE.txt`). Scenes: Tanks & Temples `train` and Mip-NeRF 360 `garden`. Dry settings: 1000 iterations, component cap 100k. Full outputs are in [dry_20261020_162208/](dry_20261020_162208/).

## Machine

| | |
| --- | --- |
| Requested | `colab new --gpu G4`: refused twice with 503 "Service Unavailable" (16:17, 16:18). Then `--gpu A100`: one 503, then allocated on the second try. |
| GPU | NVIDIA A100-SXM4-40GB (sm 8.0, 42.4 GB) |
| Image | Python 3.13.15, torch 2.11.0+cu130, nvcc 13.0, driver 580.82.07, 83 GB RAM, 194 GB free disk |
| Rate | 5.3 compute units/h (Colab `ccu-info` for this session) |
| Wall clock | VM up 16:20:58 → 16:49:00, about 28 min, about 2.5 units |

## Steps

| Step | Result | Time |
| --- | --- | --- |
| setup | extensions built and imported with the three build-only fixes (NOTES.md 22) | build 240 s |
| data | 360_v2.zip 12.54 GB (687 s, about 18 MB/s); tandt_db.zip 0.68 GB (46 s); train 301 images, garden 185 | 12.4 min |
| pipeline | both scenes passed every phase (train, render, metrics, fps, stats); one transient status-check failure, retried | 8.8 min |

## Per scene (1000 iterations: checks the chain; not a quality result)

| | train | garden |
| --- | --- | --- |
| components (SfM initialisation; the 100k dry cap only limits growth) | 182,686 | 138,766 |
| training speed | 58 it/s | 53 it/s |
| test PSNR / SSIM / LPIPS | 14.84 / 0.518 / 0.556 | 21.19 / 0.612 / 0.404 |
| render size, test views | 980×545, 38 | 1297×840, 24 |
| FPS (render only) | 470 | 639 |
| peak GPU memory | 3.1 GB | 3.8 GB |
| learned ν: median; share at ν = 1 | 886; 2.5% | 1990; 0.4% |
| negative opacity | 17.1% | 3.3% |
| Option B share of visible pairs, scale 1/1 → 1/8 | 17.5% → 19.1% | 0.9% → 1.5% |
| estimated kernel-evaluation cost vs SSS, 1/1 → 1/8 | 1.88× → 2.19× | 1.63× → 1.18× |

## Notes for the real run

- ν leaves its initial 100 quickly (median 886–1990 after 1000 iterations), so most components take Option A.
- Part of the estimated overhead comes from the radius, not from Option B. The design's radius goes out to the alpha cutoff, while SSS's rule cuts large-ν components at 3σ (see `radius_study.md`). That is worth revisiting in M4.
- Every fresh VM spends about 12 minutes downloading Mip-NeRF 360. Fetching only the needed scene with HTTP range requests would save most of that.
