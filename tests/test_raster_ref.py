"""The float64 rasterizer reference (CUDA parity ground truth) against the M1 kernels, on CPU."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from mip_sss import option_a, sss_kernel
from mip_sss.raster_ref import rasterize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "colab"))
from m2_scene import H, W, camera, scene  # noqa: E402

BG = torch.tensor([0.2, 0.3, 0.4])


def cam64():
    c = camera(device="cpu")
    return {k: (v.double() if torch.is_tensor(v) else v) for k, v in c.items()}


@pytest.mark.parametrize("kernel_size", [0.0, 0.1, 0.4])
@pytest.mark.parametrize("nu", [1.0, 4.0, 1000.0])
def test_single_component_matches_m1_kernels(kernel_size, nu):
    cam = cam64()
    z = 4.0
    means = torch.tensor([[0.13, -0.07, z]], dtype=torch.float64)
    sig = 0.03
    cov3 = torch.diag(torch.tensor([sig ** 2, (2 * sig) ** 2, sig ** 2], dtype=torch.float64))[None]
    color = torch.tensor([[0.9, 0.5, 0.1]], dtype=torch.float64)
    o = torch.tensor([0.7], dtype=torch.float64)
    nut = torch.tensor([nu], dtype=torch.float64)
    img = rasterize(means, cov3, color, o, nut, cam["world_view"], cam["full_proj"], cam["tanfovx"],
                    cam["tanfovy"], H, W, BG.double(), kernel_size=kernel_size)

    # independent route: pinhole projection (camera at the origin, looking down +z) + M1 kernels
    fx, fy = W / (2 * cam["tanfovx"]), H / (2 * cam["tanfovy"])
    x, y = means[0, 0], means[0, 1]
    J = torch.tensor([[fx / z, 0, -fx * x / z ** 2], [0, fy / z, -fy * y / z ** 2]], dtype=torch.float64)
    cov2 = (J @ cov3[0] @ J.T)[None]
    mu = torch.stack([fx * x / z + W / 2 - 0.5, fy * y / z + H / 2 - 0.5])[None]
    py, px = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64),
                            indexing="ij")
    u = torch.stack([px.reshape(-1), py.reshape(-1)], 1)
    k = option_a(u, mu, cov2, nut, kernel_size) if kernel_size > 0 else sss_kernel(u, mu, cov2, nut)
    alpha = (o[:, None] * k)[0]
    alpha = torch.where(alpha.abs() < 1 / 255, torch.zeros_like(alpha), alpha.clamp(max=0.99))
    expect = (alpha[:, None] * color + (1 - alpha)[:, None] * BG.double()).T.reshape(3, H, W)
    # pixels the culling radius excludes keep the background in the reference; compare the rest
    inside = (img - BG.double()[:, None, None]).abs().sum(0) > 0
    assert inside.sum() > 10
    assert torch.allclose(img[:, inside], expect[:, inside], atol=1e-9)


def test_random_scene_is_finite_and_differentiable():
    cam = cam64()
    sc = {k: v.double() for k, v in scene(0, device="cpu").items()}
    means = sc["means"].clone().requires_grad_(True)
    o = sc["opacity"].clone().requires_grad_(True)
    nu = sc["nu"].clone().requires_grad_(True)
    img = rasterize(means, sc["cov3"], sc["colors"], o, nu, cam["world_view"], cam["full_proj"],
                    cam["tanfovx"], cam["tanfovy"], H, W, BG.double(), kernel_size=0.1)
    (img * sc["weights"]).sum().backward()
    for t in (img, means.grad, o.grad, nu.grad):
        assert torch.isfinite(t).all()
    assert o.grad.abs().sum() > 0 and nu.grad.abs().sum() > 0


def test_filter_recovers_subpixel_component_between_samples():
    """The aliasing M2 fixes: a 0.17 px component centred between pixel samples renders nothing
    unfiltered (every sample is below 1/255); with the filter it renders with about its mass."""
    cam = cam64()
    z, sig = 4.0, 0.012
    means = torch.tensor([[0.0, 0.0, z]], dtype=torch.float64)  # projects to (31.5, 23.5)
    cov3 = torch.eye(3, dtype=torch.float64)[None] * sig ** 2
    color, o, nu = torch.ones(1, 3, dtype=torch.float64), 0.5, 30.0
    imgs = [rasterize(means, cov3, color, torch.tensor([o], dtype=torch.float64),
                      torch.tensor([nu], dtype=torch.float64), cam["world_view"], cam["full_proj"],
                      cam["tanfovx"], cam["tanfovy"], H, W, torch.zeros(3, dtype=torch.float64),
                      kernel_size=ks)[0] for ks in (0.0, 0.3)]
    assert imgs[0].max() == 0
    fx = W / (2 * cam["tanfovx"])
    mass = o * 2 * math.pi * (fx * sig / z) ** 2  # o · ∫T over the image plane, unfiltered
    assert abs(float(imgs[1].sum()) / mass - 1) < 0.2  # minus the alpha < 1/255 fringe


def test_pipeline_run_specs():
    spec = importlib.util.spec_from_file_location("m0_pipeline", ROOT / "scripts/colab/m0_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.parse_run("train_mip:train:iterations=7000,mip_kernel_size=0.1") == (
        "train_mip", "train", {"iterations": 7000, "mip_kernel_size": 0.1})
    assert mod.parse_run("garden") == ("garden", "garden", {})
    assert mod.parse_run("g2:garden") == ("g2", "garden", {})
    assert math.isclose(mod.parse_run("a:b:x=1e-3")[2]["x"], 1e-3)


# ---- Option B mode (M4) ----
from mip_sss import design  # noqa: E402
from mip_sss.raster_ref import rasterize_option_b  # noqa: E402


def test_option_b_mode_all_fast_equals_m2_reference():
    """With σ3 = 0 and only fast-path components, Option B mode is Option A: the M2 image."""
    cam = cam64()
    sc = {k: v.double() for k, v in scene(1, device="cpu").items()}
    nu = torch.full_like(sc["nu"], 2000.0)  # every component takes the fast path
    a = rasterize(sc["means"], sc["cov3"], sc["colors"], sc["opacity"], nu, cam["world_view"],
                  cam["full_proj"], cam["tanfovx"], cam["tanfovy"], H, W, BG.double(), kernel_size=0.1)
    b, fast = rasterize_option_b(sc["means"], sc["cov3"], sc["colors"], sc["opacity"], nu,
                                 torch.zeros_like(nu), cam["world_view"], cam["full_proj"],
                                 cam["tanfovx"], cam["tanfovy"], H, W, BG.double(), 0.1)
    assert bool(fast.all())
    assert torch.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("sigma3", [0.0, 0.01])
@pytest.mark.parametrize("nu", [1.0, 3.0, 1000.0])
def test_option_b_single_component_matches_design(sigma3, nu):
    cam = cam64()
    z = 4.0
    means = torch.tensor([[0.13, -0.07, z]], dtype=torch.float64)
    cov3 = torch.diag(torch.tensor([0.12, 0.3, 0.2], dtype=torch.float64) ** 2)[None]
    color = torch.tensor([[0.9, 0.5, 0.1]], dtype=torch.float64)
    o = torch.tensor([0.7], dtype=torch.float64)
    nut = torch.tensor([nu], dtype=torch.float64)
    img, fast = rasterize_option_b(means, cov3, color, o, nut, torch.tensor([sigma3]), cam["world_view"],
                                   cam["full_proj"], cam["tanfovx"], cam["tanfovy"], H, W, BG.double(), 0.1)
    assert bool(fast[0]) == (nu >= 400)
    fx, fy = W / (2 * cam["tanfovx"]), H / (2 * cam["tanfovy"])
    x, y = means[0, 0], means[0, 1]
    M = torch.tensor([[fx / z, 0, -fx * x / z ** 2], [0, fy / z, -fy * y / z ** 2]], dtype=torch.float64)
    mu = torch.stack([fx * x / z + W / 2 - 0.5, fy * y / z + H / 2 - 0.5])[None]
    py, px = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    u = torch.stack([px.reshape(-1), py.reshape(-1)], 1)
    k = design.kernel_3d(u, mu, cov3, M[None], nut, torch.tensor([sigma3 ** 2]), 0.1)[0]
    alpha = o * k
    alpha = torch.where(alpha.abs() < 1 / 255, torch.zeros_like(alpha), alpha.clamp(max=0.99))
    expect = (alpha[:, None] * color + (1 - alpha)[:, None] * BG.double()).T.reshape(3, H, W)
    inside = (img - BG.double()[:, None, None]).abs().sum(0) > 0
    assert inside.sum() > 10
    assert torch.allclose(img[:, inside], expect[:, inside], atol=1e-9)
