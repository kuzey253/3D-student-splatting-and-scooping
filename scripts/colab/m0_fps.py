"""Rendering speed of a trained SSS model on its test views. Run on the GPU VM from the SSS root.

Loads the model the way render.py does, warms up on a few views, then times --passes passes over
all test views with CUDA synchronisation around the timed region (render only; no image I/O).

Usage: python scripts/colab/m0_fps.py -m <model_dir> --out <fps.json> [--passes 5] [--only_suffix _d0]

--only_suffix times only the test views whose image name ends with it (M6: _d0 is the full-resolution
scale of multi-scale Blender).
"""
import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from scene import Scene_nt  # noqa: E402
from scene.nt_model import NTModel  # noqa: E402
from t_renderer import render  # noqa: E402


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
        prims = NTModel(dataset.sh_degree, dataset.nu_degree)
        scene = Scene_nt(dataset, prims, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                          dtype=torch.float32, device="cuda")
        views = scene.getTestCameras()
        if args.only_suffix:
            views = [v for v in views if v.image_name.endswith(args.only_suffix)]
        for v in views[:5]:
            render(v, prims, pipe, bg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.passes):
            for v in views:
                render(v, prims, pipe, bg)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    frames = args.passes * len(views)
    out = {"fps": frames / dt, "frames": frames, "seconds": dt,
           "size": [views[0].image_width, views[0].image_height],
           "components": int(prims.get_xyz.shape[0])}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
