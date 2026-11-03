"""M6: PSNR, SSIM and LPIPS per scale (PLAN item 10; README experiments).

SSS's and Mip-Splatting's metrics.py average every test image together; the README asks for each
scale separately. The metrics are SSS's own (utils.image_utils.psnr, utils.loss_utils.ssim,
lpipsPyTorch LPIPS-VGG 0.1, as metrics.py), on render.py's PNGs, so SSS and Mip-Splatting runs are
scored by the same code.

Two layouts:

  - multi-scale Blender: one render.py pass renders every test camera; --metadata gives the
    converted scene's metadata.json, whose test split lists the cameras in render order with their
    scale label j (image size 1/2^j). Sizes are checked against it.
        python3 scripts/m6_metrics.py --metadata DATA/lego/metadata.json \
            --pair all=MODEL/test/ours_30000/renders:MODEL/test/ours_30000/gt --out per_scale.json
  - zoom-in (Mip-NeRF 360, trained at -r 8): one render.py pass per -r, one --pair per scale.
        python3 scripts/m6_metrics.py --pair 8=R8/renders:R8/gt --pair 1=R1/renders:R1/gt --out ...

Output: {"scales": {label: {"PSNR", "SSIM", "LPIPS", "n", "size"}}, "all": mean over every image}.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.image_utils import psnr  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


def load(path, device):
    return tf.to_tensor(Image.open(path)).unsqueeze(0)[:, :3].to(device)


def blender_labels(metadata):
    """Scale label and (width, height) of each test camera, in the order readMultiScale loads them."""
    meta = json.loads(Path(metadata).read_text())["test"]
    return [(int(l), (int(w), int(h))) for l, w, h in zip(meta["label"], meta["width"], meta["height"])]


def score(pairs, labels=None, device="cpu", use_lpips=True):
    """pairs: [(label, renders_dir, gt_dir)]. With labels, the single pair is split by label."""
    lp = None
    if use_lpips:
        from lpipsPyTorch.modules.lpips import LPIPS
        lp = LPIPS("vgg", "0.1").to(device)
    rows = []  # (label, psnr, ssim, lpips, size)
    for label, rdir, gdir in pairs:
        names = sorted(p.name for p in Path(rdir).glob("*.png"))
        gt_names = sorted(p.name for p in Path(gdir).glob("*.png"))
        if not names or names != gt_names:
            raise ValueError(f"{rdir}: {len(names)} renders vs {len(gt_names)} gt images")
        if labels is not None and len(names) != len(labels):
            raise ValueError(f"{rdir}: {len(names)} renders but metadata lists {len(labels)} test cameras")
        for i, name in enumerate(names):
            r, g = load(Path(rdir) / name, device), load(Path(gdir) / name, device)
            if r.shape != g.shape:
                raise ValueError(f"{name}: render {tuple(r.shape)} vs gt {tuple(g.shape)}")
            size = (r.shape[-1], r.shape[-2])
            lab = label
            if labels is not None:
                lab, want = labels[int(Path(name).stem)]
                if size != want:
                    raise ValueError(f"{name}: size {size}, metadata says {want} (label {lab})")
            with torch.no_grad():
                rows.append((str(lab), float(psnr(r, g)), float(ssim(r, g)),
                             float(lp(r, g)) if lp is not None else None, size))
    out = {"scales": {}}
    for lab in dict.fromkeys(r[0] for r in rows):
        sel = [r for r in rows if r[0] == lab]
        out["scales"][lab] = {"PSNR": mean(r[1] for r in sel), "SSIM": mean(r[2] for r in sel),
                              "LPIPS": mean(r[3] for r in sel) if use_lpips else None, "n": len(sel),
                              "size": list(sel[0][4])}
    out["all"] = {"PSNR": mean(r[1] for r in rows), "SSIM": mean(r[2] for r in rows),
                  "LPIPS": mean(r[3] for r in rows) if use_lpips else None, "n": len(rows)}
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, help="label=renders_dir:gt_dir")
    ap.add_argument("--metadata", help="multi-scale Blender metadata.json; splits the single pair by scale")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_lpips", action="store_true")
    a = ap.parse_args()
    pairs = []
    for p in a.pair:
        label, dirs = p.split("=", 1)
        rdir, gdir = dirs.split(":", 1)
        pairs.append((label, rdir, gdir))
    labels = blender_labels(a.metadata) if a.metadata else None
    if labels is not None and len(pairs) != 1:
        raise SystemExit("--metadata takes exactly one --pair")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = score(pairs, labels, device, use_lpips=not a.no_lpips)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for lab, s in res["scales"].items():
        print(f"scale {lab}: PSNR {s['PSNR']:.3f}  SSIM {s['SSIM']:.4f}  LPIPS {s['LPIPS']}  n={s['n']}  {s['size']}")
    print("M6_METRICS=OK")


if __name__ == "__main__":
    main()
