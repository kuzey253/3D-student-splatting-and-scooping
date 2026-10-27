"""The M4 NaN capture in diff_t_rasterization (MIP_NAN_CAPTURE), run on CPU tensors: it saves the
non-finite components' rows, per-stage counts, the camera and the scene snapshot the replay
(scripts/m4_nan_replay.py) reads."""
import ast
import os
from pathlib import Path
from typing import NamedTuple
from types import SimpleNamespace

import torch

SRC = Path(__file__).resolve().parents[1] / "submodules/diff-t-rasterization/diff_t_rasterization/__init__.py"


def load_capture():
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_nan_capture")
    ns = {"os": os, "torch": torch}
    exec(compile(ast.Module([fn], []), str(SRC), "exec"), ns)
    return ns["_nan_capture"]


class Settings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    viewmatrix: torch.Tensor


def test_capture_saves_bad_rows_and_scene(tmp_path, monkeypatch):
    monkeypatch.setenv("MIP_NAN_CAPTURE", str(tmp_path))
    capture = load_capture()
    P = 50
    g = torch.Generator().manual_seed(0)
    inputs = dict(means3D=torch.randn(P, 3, generator=g), scales=torch.rand(P, 3, generator=g),
                  rotations=torch.randn(P, 4, generator=g), cov3Ds_precomp=torch.empty(0), radii=torch.ones(P),
                  sh=torch.randn(P, 16, 3, generator=g), nu_degree=torch.ones(P, 1), negative_value=torch.ones(P, 1),
                  opacities=torch.rand(P, 1, generator=g), colors_precomp=torch.empty(0), filter_3d=None)
    grads = dict(grad_means3D=torch.zeros(P, 3), dL_dB=torch.zeros(P, 9), dL_dconic=torch.zeros(P, 2, 2),
                 grad_cov3Ds_precomp=torch.zeros(P, 6), grad_sh=torch.empty(0))
    grads["grad_means3D"][[3, 17]] = float("nan")
    grads["dL_dB"][17, 4] = float("inf")
    ctx = SimpleNamespace(color_finite=True)
    capture(ctx, Settings(10, 20, 0.5, torch.eye(4)), torch.zeros(3, 10, 20), inputs, grads)
    cap = torch.load(tmp_path / "nan_capture.pt")
    assert cap["n_bad"] == 2 and cap["idx"].tolist() == [3, 17]
    assert cap["counts"]["grad_means3D"] == 2 and cap["counts"]["dL_dB"] == 1 and cap["nan_counts"]["dL_dB"] == 0
    assert cap["rows"]["nonfinite"]["dL_dB"].tolist() == [False, True]
    assert torch.equal(cap["rows"]["scales"], inputs["scales"][[3, 17]])
    assert cap["camera"]["image_width"] == 20 and cap["color_finite"] and cap["grad_out_finite"]
    scene = torch.load(tmp_path / "nan_capture_scene.pt")
    assert scene["means3D"].shape == (P, 3) and scene["sh_dc"].shape == (P, 3) and scene["grad_out"].shape == (3, 10, 20)
    # only the first NaN is captured
    grads["grad_means3D"][5] = float("nan")
    capture(ctx, Settings(10, 20, 0.5, torch.eye(4)), torch.zeros(3, 10, 20), inputs, grads)
    assert torch.load(tmp_path / "nan_capture.pt")["n_bad"] == 2
