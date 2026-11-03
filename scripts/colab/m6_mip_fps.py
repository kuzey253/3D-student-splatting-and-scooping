"""Rendering speed of a trained Mip-Splatting model (M6 baseline), timed as scripts/colab/m0_fps.py
times SSS: warm up on a few views, then --passes passes over the test views with CUDA
synchronisation around the timed region (render only; no image I/O).

Run on the GPU VM from the Mip-Splatting checkout (its modules are imported from the working directory):

    cd /content/mip-splatting && python /content/sss/scripts/colab/m6_mip_fps.py -m <model_dir> --out <fps.json> \
        [--passes 5] [--only_suffix _d0]
"""
import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--passes", default=5, type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only_suffix")
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                          dtype=torch.float32, device="cuda")
        views = scene.getTestCameras()
        if args.only_suffix:
            views = [v for v in views if v.image_name.endswith(args.only_suffix)]
        ks = dataset.kernel_size
        for v in views[:5]:
            render(v, gaussians, pipe, bg, kernel_size=ks)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.passes):
            for v in views:
                render(v, gaussians, pipe, bg, kernel_size=ks)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    frames = args.passes * len(views)
    out = {"fps": frames / dt, "frames": frames, "seconds": dt,
           "size": [views[0].image_width, views[0].image_height],
           "components": int(gaussians.get_xyz.shape[0])}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
