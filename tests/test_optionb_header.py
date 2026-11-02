"""mip_optionb.h (the M4 CUDA device code) compiled as host C++, against optionb_staged.py.

Checks the transliteration in float32 before any GPU time, and that degenerate components (flat,
needle, edge-on; σ3 = 0 and > 0; ν = 1) give finite values and gradients. Skipped without a C++
compiler.
"""
import ctypes
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from mip_sss import optionb_staged as ob
from mip_sss.raster_ref import optb_radius_bound

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent / "optb_host"
CXX = shutil.which("clang++") or shutil.which("g++")
F = ctypes.POINTER(ctypes.c_float)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    if CXX is None:
        pytest.skip("no C++ compiler")
    out = tmp_path_factory.mktemp("optb") / "liboptb.so"
    subprocess.run([CXX, "-std=c++17", "-O1", "-shared", "-fPIC", "-I", str(HERE),
                    "-I", str(ROOT / "submodules/diff-t-rasterization/cuda_rasterizer"),
                    "-o", str(out), str(HERE / "harness.cpp")], check=True)
    return ctypes.CDLL(str(out))


def _f32(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return a, a.ctypes.data_as(F)


def run(lib, T, cov6, sigma, nu, s, opacity, d, dLdT):
    n = len(dLdT)
    keep = [_f32(T.reshape(-1)), _f32(cov6), _f32(d.reshape(-1)), _f32(dLdT)]
    out = {k: _f32(np.zeros(m)) for k, m in (("T", n), ("acc", 9), ("dd", 2 * n), ("dcov6", 6), ("dT", 6))}
    sc = {k: ctypes.c_float() for k in ("dnu", "loghp", "radius")}
    fast = ctypes.c_int()
    lib.run(keep[0][1], keep[1][1], ctypes.c_float(sigma), ctypes.c_float(nu), ctypes.c_float(s),
            ctypes.c_float(opacity), n, keep[2][1], keep[3][1], out["T"][1], out["acc"][1], ctypes.byref(sc["dnu"]),
            out["dd"][1], out["dcov6"][1], out["dT"][1], ctypes.byref(fast), ctypes.byref(sc["loghp"]),
            ctypes.byref(sc["radius"]))
    res = {k: v[0].astype(np.float64) for k, v in out.items()}
    res.update({k: v.value for k, v in sc.items()})
    res["fast"] = bool(fast.value)
    return res


def case(seed, sigma):  # as tests/test_optionb_staged.py
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
    S3 = q @ torch.diag(torch.exp(torch.randn(3, generator=g, dtype=torch.float64) - 7)) @ q.T
    T = torch.randn(2, 3, generator=g, dtype=torch.float64) * 6 + torch.tensor([[40.0, 0, 0], [0, 40.0, 0]],
                                                                               dtype=torch.float64)
    d = torch.randn(40, 2, generator=g, dtype=torch.float64) * 1.5
    dLdT = torch.randn(40, generator=g, dtype=torch.float64)
    return S3, torch.tensor(sigma, dtype=torch.float64), T, d, dLdT


def rel(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


@pytest.mark.parametrize("nu", [1.0, 1.5, 3.0, 30.0])
@pytest.mark.parametrize("sigma", [0.0, 0.004])
@pytest.mark.parametrize("seed", range(4))
def test_header_matches_staged(lib, seed, sigma, nu):
    S3, sig, T, d, dLdT = case(seed, sigma)
    s, o = 0.1, 0.7
    nut = torch.tensor(nu, dtype=torch.float64)
    pre = ob.preprocess(S3, sig, T, nut, s)
    val, acc = ob.pixel(d, pre, nut, dLdT)
    dS3, dT = ob.chain(S3, sig, T, acc)
    dcov6 = torch.stack([dS3[0, 0], dS3[0, 1] + dS3[1, 0], dS3[0, 2] + dS3[2, 0], dS3[1, 1],
                         dS3[1, 2] + dS3[2, 1], dS3[2, 2]])
    cov6 = S3[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    r = run(lib, T.numpy(), cov6.numpy(), float(sig), nu, s, o, d.numpy(), dLdT.numpy())
    assert r["fast"] == pre["fast"]
    assert abs(r["loghp"] - float(pre["loghp"])) < 1e-5
    assert rel(r["T"], val) < 1e-5
    if pre["fast"]:
        return
    assert rel(r["dd"], acc["dd"]) < 1e-4
    assert rel(r["dcov6"], dcov6) < 1e-4
    assert rel(r["dT"], dT) < 1e-4
    # ν ≤ 30: float32 cancellation grows above (NOTES.md 41)
    assert rel([r["dnu"]], [float(acc["dnu"])]) < 1e-4
    assert abs(r["radius"] / optb_radius_bound(pre, nut, o, s) - 1) < 1e-5


def test_degenerate_components_stay_finite(lib):
    """Flat, needle and edge-on components (det Σ3 down to 0, near-singular Σ2), ν = 1 (Option B),
    σ3 = 0 and > 0: whenever preprocess would draw the component, every value and gradient is
    finite (the M4 G4 run hit NaN gradients here, NOTES.md 43)."""
    rng = np.random.default_rng(0)
    drawn = 0
    for trial in range(1500):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        sc = np.exp(rng.uniform(-9, -1, 3)) * np.array([1, 1, 10.0 ** rng.uniform(-12, 0)])
        S = q @ np.diag(sc ** 2) @ q.T
        if trial % 3 == 0:  # flat plane seen edge-on
            S = np.diag([sc[2] ** 2, sc[0] ** 2, sc[1] ** 2])
        cov6 = S[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]].astype(np.float32)
        z, f = 4.0, 55.0
        x, y = rng.uniform(-2, 2, 2)
        T = np.array([f / z, 0, -f * x / z ** 2, 0, f / z, -f * y / z ** 2])
        sigma = 0.0 if trial % 2 == 0 else float(np.exp(rng.uniform(-8, -4)))
        d = rng.normal(size=(30, 2)) * 2
        dLdT = rng.normal(size=30) * (rng.uniform(size=30) < 0.7)
        r = run(lib, T, cov6, sigma, 1.0, 0.1, 0.8, d, dLdT)
        c = cov6.astype(np.float32)
        D = c[0] * (c[3] * c[5] - c[4] * c[4]) - c[1] * (c[1] * c[5] - c[4] * c[2]) + c[2] * (c[1] * c[4] - c[3] * c[2])
        if not math.isfinite(r["loghp"]) or (sigma > 0 and not D > 0):
            continue  # preprocess skips it (forward.cu)
        drawn += 1
        for k in ("T", "acc", "dd", "dcov6", "dT", "dnu", "radius"):
            assert np.isfinite(r[k]).all(), (trial, sigma, k)
    assert drawn > 1000


@pytest.mark.parametrize("sizes", ["ordinary", "tiny"])
def test_full_component_path_stays_finite(lib, sizes):
    """The whole per-component path (projection, mode decision, slot, per-pixel value and backward
    over the full alpha-safe radius with the render's alpha test, then optb_cov_backward) on
    components from sub-pixel to ~1e5 px², flat, needle and edge-on, all ν, σ3 = 0 and > 0
    (NOTES.md 44); "tiny" (world scales 1e-13 to 3e-4) has denormal 2D determinants (NOTES.md 45)."""
    rng = np.random.default_rng(1)
    drawn = {"fast": 0, "B": 0}
    for trial in range(3000):
        z = rng.uniform(1, 40)
        x, y = rng.uniform(-0.8, 0.8, 2) * z
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        sc = np.exp(rng.normal(-4.5, 1.8, 3)) if sizes == "ordinary" else np.exp(rng.uniform(-30, -8, 3))
        kind = trial % 4
        if kind == 1:
            sc[2] *= 10.0 ** rng.uniform(-14, -3)  # flat
        if kind == 2:
            sc[1:] *= 10.0 ** rng.uniform(-10, -2, 2)  # needle
        S = q @ np.diag(sc ** 2) @ q.T
        if kind == 3:  # flat, seen edge-on
            ray = np.array([x, y, z]) / np.linalg.norm([x, y, z])
            B, _ = np.linalg.qr(np.stack([np.cross(ray, rng.normal(size=3)), ray, rng.normal(size=3)], 1))
            S = B @ np.diag([sc[0] ** 2, (sc[1] * 1e-9) ** 2, sc[2] ** 2]) @ B.T
        cov6 = S[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
        sigma = 0.0 if trial % 5 < 3 else float(np.exp(rng.uniform(-9, -3)))
        nu = float(rng.choice([1.0, 1.5, 3.0, 30.0, 150.0, 500.0, 5000.0]))
        op = float(rng.choice([-1, 1]) * rng.uniform(0.02, 0.99))
        n = 48
        dfrac = rng.uniform(-1, 1, (n, 2))
        dfrac[:6] *= 1e-3
        dLdT = rng.normal(size=n) * (rng.uniform(size=n) < 0.6)
        dcon = rng.normal(size=4) * 10.0 ** rng.uniform(-6, 0) * (rng.uniform() < 0.7)
        keep = [_f32(np.array([x, y, z])), _f32(cov6), _f32(dfrac.reshape(-1)), _f32(dLdT), _f32(dcon)]
        outs = {k: _f32(np.zeros(m)) for k, m in (("T", n), ("dd", 2 * n), ("dcov6", 6), ("dt", 3))}
        sc_ = {k: ctypes.c_float() for k in ("dnu", "dop", "radius")}
        fast = ctypes.c_int()
        ok = lib.run_full(keep[0][1], keep[1][1], ctypes.c_float(sigma), ctypes.c_float(nu), ctypes.c_float(0.1),
                          ctypes.c_float(op), ctypes.c_float(1000.0), ctypes.c_float(0.5), n, keep[2][1],
                          keep[3][1], keep[4][1], ctypes.c_float(float(rng.normal())), outs["T"][1],
                          ctypes.byref(sc_["dnu"]), outs["dd"][1], outs["dcov6"][1], outs["dt"][1],
                          ctypes.byref(sc_["dop"]), ctypes.byref(fast), ctypes.byref(sc_["radius"]))
        if not ok:
            continue
        drawn["fast" if fast.value else "B"] += 1
        for k, v in list(outs.items()) + list(sc_.items()):
            val = v[0] if isinstance(v, tuple) else v.value
            assert np.isfinite(val).all(), (trial, kind, sigma, nu, k)
    assert drawn["B"] > 1000 and drawn["fast"] > 100


def test_backward_uses_the_forward_determinant(lib):
    """The component captured at the first NaN of the G4 run (NOTES.md 47): a fast-path disc seen
    nearly edge-on, whose det Σ2 (true 163 px⁴) is ±1–2 ulps of a·c (ulp 128). The forward kernel
    drew it with a positive det; if a later kernel's own rounding gives 0, Σ2f⁻¹ = ∞ and the zero
    opacity gradient makes 0·∞ = NaN. Passing the forward's det (as backward.cu now does) is finite."""
    view = np.array([0.11051491647958755, 0.13440485298633575, 0.9847445487976074, 0.0, -0.03881309926509857,
                     0.9906415939331055, -0.13085384666919708, 0.0, -0.9931163191795349, -0.023759685456752777,
                     0.11469734460115433, 0.0, 0.42403748631477356, 0.13484427332878113, 2.72666335105896, 1.0])
    mean = np.array([0.33434492349624634, -1.702770709991455, -1.419220209121704])
    scales = np.array([1.4464781284332275, 0.00017416523769497871, 0.30633673071861267])
    q = np.array([0.8559366464614868, -0.2753865122795105, -0.43533140420913696, -0.044959891587495804])
    r, x, y, z = q / np.linalg.norm(q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)],
                  [2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)],
                  [2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)]])
    W = np.array([[view[4 * j + i] for j in range(3)] for i in range(3)])
    t = W @ mean + view[12:15]
    cov = W @ R @ np.diag(scales ** 2) @ R.T @ W.T
    cov6 = cov[[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    H, Wd, tx, ty = 545, 980, 0.8446965112441064, 0.4679476755039769
    fx, fy = Wd / (2 * tx), H / (2 * ty)
    D = float(np.linalg.det(cov))
    dcon = np.array([-0.0004224069125484675, 0.0002058896643575281, 0.0, -0.00010035467857960612])

    def backward(keep):
        k = [_f32(t), _f32(cov6), _f32(dcon), _f32(np.zeros(9))]
        outs = [_f32(np.zeros(6)), _f32(np.zeros(3))]
        dop, fast, lh = ctypes.c_float(), ctypes.c_int(), ctypes.c_float()
        kp = _f32(np.array(keep)) if keep is not None else None
        drawn = lib.run_cov_backward(k[0][1], k[1][1], ctypes.c_float(0.0), ctypes.c_float(5657.1), ctypes.c_float(0.1),
                                     ctypes.c_float(0.2385), ctypes.c_float(fx), ctypes.c_float(fy), ctypes.c_float(tx),
                                     ctypes.c_float(ty), k[2][1], ctypes.c_float(0.0), k[3][1], outs[0][1], outs[1][1],
                                     ctypes.byref(dop), ctypes.byref(fast), ctypes.byref(lh),
                                     kp[1] if kp is not None else None)
        return drawn, bool(fast.value), np.concatenate([outs[0][0], outs[1][0]])

    drawn, fast, _ = backward(None)
    assert drawn and fast  # the forward draws it on the fast path (ν = 5657)
    _, _, g = backward([198.0, D])  # the forward's det (one of the float32 roundings of a·c − b²)
    assert np.isfinite(g).all()
    _, _, g0 = backward([0.0, D])  # what the GPU's backward kernel had: det rounded to 0
    assert not np.isfinite(g0).all()
