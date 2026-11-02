"""Replay an M4 NaN capture (diff_t_rasterization's MIP_NAN_CAPTURE) on the CPU.

For each captured component it reports which backward stages were non-finite on the GPU, its
parameters and mode, and replays the per-component backward (optb_component + optb_cov_backward,
mip_optionb.h compiled as host C++) on the incoming gradients the GPU had, in the camera frame.
That separates a NaN made per pixel (incoming accumulators already non-finite), in the per-component
chain (replay non-finite), or after it (SSS's covariance/projection/SH backward).

    python3 scripts/m4_nan_replay.py results/m4/<run>/train_s01_ob/nan_capture.pt [--out summary.json]
"""
import argparse
import ctypes
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mip_sss.analysis import quat_to_rot  # noqa: E402

F = ctypes.POINTER(ctypes.c_float)
STAGES = ("dL_dB", "dL_dconic", "grad_opacities", "grad_nu_degree", "grad_means2D", "grad_cov3Ds_precomp",
          "grad_scales", "grad_rotations", "grad_sh", "grad_means3D")


def build_harness():
    cxx = shutil.which("clang++") or shutil.which("g++")
    out = Path(tempfile.mkdtemp()) / "liboptb.so"
    subprocess.run([cxx, "-std=c++17", "-O1", "-shared", "-fPIC", "-I", str(ROOT / "tests/optb_host"),
                    "-I", str(ROOT / "submodules/diff-t-rasterization/cuda_rasterizer"), "-o", str(out),
                    str(ROOT / "tests/optb_host/harness.cpp")], check=True)
    return ctypes.CDLL(str(out))


def f32(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32).reshape(-1))
    return a, a.ctypes.data_as(F)


def camera_frame(cam, means, cov3):
    """Camera-space means and covariances, as transformPoint4x3 / T = J W use the view matrix."""
    v = cam["viewmatrix"].double().reshape(-1)
    Wm = torch.stack([torch.stack([v[4 * j + i] for j in range(3)]) for i in range(3)])  # W[i][j] = view[4j + i]
    t = means.double() @ Wm.T + torch.stack([v[12], v[13], v[14]])
    return t, Wm @ cov3 @ Wm.T, Wm


def cov6_to_world(d6, Wm):
    """dL/dcov6 (SSS convention: off-diagonals doubled) from the camera frame to the world frame."""
    G = torch.tensor([[d6[0], d6[1] / 2, d6[2] / 2], [d6[1] / 2, d6[3], d6[4] / 2], [d6[2] / 2, d6[4] / 2, d6[5]]],
                     dtype=torch.float64)
    Gw = Wm.T @ G @ Wm
    return torch.stack([Gw[0, 0], 2 * Gw[0, 1], 2 * Gw[0, 2], Gw[1, 1], 2 * Gw[1, 2], Gw[2, 2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--out")
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()
    cap = torch.load(a.capture, weights_only=False)
    cam, rows = cap["camera"], cap["rows"]
    n = len(cap["idx"])
    print(f"P = {cap['P']}, components with a non-finite gradient: {cap['n_bad']} (captured {n})")
    print(f"forward image finite: {cap['color_finite']}, incoming image gradient finite: {cap['grad_out_finite']}")
    print("non-finite rows per stage:", {k: v for k, v in cap["counts"].items() if v})
    print("  of which NaN:", {k: v for k, v in cap["nan_counts"].items() if v})

    H, W = cam["image_height"], cam["image_width"]
    tanx, tany = float(cam["tanfovx"]), float(cam["tanfovy"])
    fx, fy = W / (2 * tanx), H / (2 * tany)
    s = float(cam["kernel_size"])
    scales = rows["scales"].double() * float(cam["scale_modifier"])
    R = quat_to_rot(rows["rotations"].double())
    cov3 = R @ torch.diag_embed(scales ** 2) @ R.transpose(1, 2)
    t, cov_cam, Wm = camera_frame(cam, rows["means3D"], cov3)
    sigma = rows["filter_3d"].double().reshape(-1) if "filter_3d" in rows else torch.zeros(n, dtype=torch.float64)
    lib = build_harness()

    first_stage = {}
    replay = {"finite": 0, "nonfinite": 0, "skipped": 0}
    table = []
    for i in range(n):
        bad = [k for k in STAGES if k in rows["nonfinite"] and bool(rows["nonfinite"][k][i])]
        first = bad[0] if bad else "none"
        first_stage[first] = first_stage.get(first, 0) + 1
        c6 = cov_cam[i][[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
        nu, op = float(rows["nu_degree"][i].reshape(-1)[0]), float(rows["opacities"][i].reshape(-1)[0])
        dcon = rows["dL_dconic"][i].reshape(-1) if "dL_dconic" in rows else torch.zeros(4)
        accB = rows["dL_dB"][i] if "dL_dB" in rows and rows["dL_dB"].numel() else torch.zeros(9)
        keep = [f32(t[i]), f32(c6), f32(dcon), f32(accB)]
        outs = [f32(np.zeros(6)), f32(np.zeros(3))]
        dop, fast, loghp = ctypes.c_float(), ctypes.c_int(), ctypes.c_float()
        # grad_opacities is after the fast path's ×hp; the incoming one is grad / hp (filled below)
        drawn = lib.run_cov_backward(keep[0][1], keep[1][1], ctypes.c_float(float(sigma[i])), ctypes.c_float(nu),
                                     ctypes.c_float(s), ctypes.c_float(op), ctypes.c_float(fx), ctypes.c_float(fy),
                                     ctypes.c_float(tanx), ctypes.c_float(tany), keep[2][1], ctypes.c_float(0.0),
                                     keep[3][1], outs[0][1], outs[1][1], ctypes.byref(dop), ctypes.byref(fast),
                                     ctypes.byref(loghp), None)
        hp = float(np.exp(loghp.value)) if np.isfinite(loghp.value) else 0.0
        g_op = float(rows["grad_opacities"][i].reshape(-1)[0]) if "grad_opacities" in rows else 0.0
        if drawn and fast.value and hp > 0 and np.isfinite(g_op):  # rerun with the incoming opacity gradient
            lib.run_cov_backward(keep[0][1], keep[1][1], ctypes.c_float(float(sigma[i])), ctypes.c_float(nu),
                                 ctypes.c_float(s), ctypes.c_float(op), ctypes.c_float(fx), ctypes.c_float(fy),
                                 ctypes.c_float(tanx), ctypes.c_float(tany), keep[2][1], ctypes.c_float(g_op / hp),
                                 keep[3][1], outs[0][1], outs[1][1], ctypes.byref(dop), ctypes.byref(fast),
                                 ctypes.byref(loghp), None)
        if not drawn:
            replay["skipped"] += 1
            rep = "skipped (preprocess would cull)"
        else:
            ok = np.isfinite(outs[0][0]).all() and np.isfinite(outs[1][0]).all()
            replay["finite" if ok else "nonfinite"] += 1
            rep = "finite" if ok else "NON-FINITE"
            gpu = rows["grad_cov3Ds_precomp"][i].double() if "grad_cov3Ds_precomp" in rows else None
            if ok and gpu is not None and torch.isfinite(gpu).all() and gpu.norm() > 0:
                mine = cov6_to_world(torch.tensor(outs[0][0], dtype=torch.float64), Wm)
                rep += f", dL/dcov3D vs GPU {float((mine - gpu).norm() / gpu.norm()):.1e}"
        ev = np.linalg.eigvalsh(cov3[i].numpy())
        row = {"idx": int(cap["idx"][i]), "nonfinite_stages": bad, "mode": "fast" if fast.value else "B",
               "nu": nu, "opacity": op, "depth": float(t[i, 2]), "scales": scales[i].tolist(),
               "cov3_eig": ev.tolist(), "loghp": loghp.value, "radius": float(rows["radii"][i]),
               "incoming_finite": bool(np.isfinite(dcon.numpy()).all() and np.isfinite(accB.numpy()).all()),
               "replay": rep}
        table.append(row)
    print("first non-finite stage (in backward order):", first_stage)
    print("per-component replay of the captured incoming gradients:", replay)
    for r in table[: a.show]:
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": {k: cap[k] for k in ("P", "n_bad", "counts", "nan_counts",
                                                                            "color_finite", "grad_out_finite")},
                                           "first_stage": first_stage, "replay": replay, "rows": table}, indent=1))


if __name__ == "__main__":
    main()
