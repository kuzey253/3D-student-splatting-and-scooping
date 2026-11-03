"""SSS relocation port and the 3D-filtered relocation (README M5.1, mip_sss.relocation)."""
import math

import torch

from mip_sss.relocation import effective_to_raw_opacity, filter_3d_coef, relocate, relocation_student_t


def kernel_loop(o, nu, N):
    """utils.cu compute_relocation_student_t for one component, as plain Python."""
    on = 1 - (1 - o) ** (1 / N)
    lg = math.lgamma
    t1 = math.exp(lg(0.5) + lg((nu + 2) / 2) - lg((nu + 3) / 2))
    d = 0.0
    for i in range(1, N + 1):
        for k in range(i):
            d += (math.comb(i - 1, k) * (-1) ** k * on ** (k + 1)
                  * math.exp(lg(0.5) + lg(((k + 1) * (nu + 3) - 1) / 2) - lg((k + 1) * (nu + 3) / 2)))
    return on, (o * t1 / d) ** 2


def cases(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = (torch.rand(n, generator=g) * 1.9 - 0.95)
    o = torch.where(o.abs() < 0.02, 0.3, o)
    scale = torch.exp(torch.randn(n, 3, generator=g) * 2 - 3)
    nu = torch.exp(torch.rand(n, generator=g) * math.log(1e4))
    N = torch.randint(1, 8, (n,), generator=g)
    sigma = torch.exp(torch.randn(n, generator=g) - 3)
    return o, scale, nu, N, sigma


def test_port_matches_the_cuda_kernel_loop():
    o, scale, nu, N, _ = cases()
    o_new, s_new = relocation_student_t(o, scale, nu, N)
    for i in range(o.shape[0]):
        on, c2 = kernel_loop(float(o[i]), float(nu[i]), int(N[i]))
        assert math.isclose(float(o_new[i]), on, rel_tol=1e-12, abs_tol=1e-15)
        assert torch.allclose(s_new[i], c2 * scale[i], rtol=1e-10)


def test_n_copies_composite_to_the_old_opacity_at_the_centre():
    o, scale, nu, N, _ = cases()
    o_new, _ = relocation_student_t(o, scale, nu, N)
    assert torch.allclose(1 - (1 - o_new) ** N, o, atol=1e-12)


def test_sss_squares_the_line_integral_scale_factor():
    """Pins an upstream quirk (NOTES.md 49): coeff = s′/s preserves the line integral of the
    composite through the centre, but the kernel scales by coeff²."""
    nu, o, N = 30.0, 0.8, 3
    on, c2 = kernel_loop(o, nu, N)
    x = torch.linspace(-60, 60, 200001)
    K = lambda s: (1 + x ** 2 / (nu * s ** 2)) ** (-(nu + 3) / 2)  # 1D slice of the 3D t
    old = torch.trapezoid(o * K(1.0), x)
    for factor, matches in ((math.sqrt(c2), True), (c2, False)):
        new = torch.trapezoid(1 - (1 - on * K(factor)) ** N, x)
        assert math.isclose(float(new), float(old), rel_tol=1e-6) is matches


def test_without_sigma_relocate_is_sss_rule():
    o, scale, nu, N, _ = cases()
    new_o, new_s, clamped = relocate(o, scale, nu, N)
    o_ref, s_ref = relocation_student_t(o, scale, nu, N)
    eps = torch.finfo(torch.float32).eps
    o_ref = o_ref.clamp(-1 + eps, 1 - eps)
    o_ref = torch.where((o_ref >= 0) & (o_ref < 0.005), 0.005, o_ref)
    o_ref = torch.where((o_ref < 0) & (o_ref > -0.005), -0.005, o_ref)
    assert torch.equal(new_o[:, 0], o_ref) and torch.equal(new_s, s_ref) and not clamped.any()


def test_effective_relocation_writes_back_the_relocated_filtered_opacity():
    o, scale, nu, N, sigma = cases(200)
    new_o, new_s, clamped = relocate(o, scale, nu, N, sigma=sigma)
    o_eff_new, s_ref = relocation_student_t(o * filter_3d_coef(scale, sigma), scale, nu, N)
    assert torch.equal(new_s, s_ref)  # the scale rule is SSS's, applied to the raw scale
    ok = ~clamped & (o_eff_new.abs() >= 0.005)  # SSS's 0.005 floor applies to the effective opacity
    assert ok.sum() > 50
    assert torch.allclose((new_o[:, 0] * filter_3d_coef(new_s, sigma))[ok], o_eff_new[ok], rtol=1e-12)
    lim = 1 - torch.finfo(torch.float64).eps
    assert (new_o.abs() <= lim).all()
    assert torch.equal(clamped, (o_eff_new.abs().clamp_min(0.005) / filter_3d_coef(new_s, sigma)) > lim)


def test_effective_relocation_reduces_to_sss_as_sigma_vanishes():
    o, scale, nu, N, _ = cases()
    raw = relocate(o, scale, nu, N)
    eff = relocate(o, scale, nu, N, sigma=torch.zeros(o.shape[0]))
    assert torch.equal(raw[0], eff[0]) and torch.equal(raw[1], eff[1])


def test_effective_to_raw_opacity_is_finite_for_degenerate_scales():
    o_eff = torch.tensor([[0.5], [-0.5], [0.01]])
    scale = torch.tensor([[1.0, 1.0, 0.0], [1e-30, 1e-30, 1e-30], [1.0, 1.0, 1.0]])
    raw, clamped = effective_to_raw_opacity(o_eff, scale, torch.tensor([0.1, 0.1, 0.1]))
    assert torch.isfinite(raw).all() and clamped.tolist() == [True, True, False]


def load_ntmodel(monkeypatch):
    """scene.nt_model on CPU: the CUDA-only modules it imports are stubbed, relocation uses the port."""
    import sys
    import types
    stubs = {"simple_knn": types.ModuleType("simple_knn"), "simple_knn._C": types.ModuleType("simple_knn._C"),
             "plyfile": types.ModuleType("plyfile"), "utils.reloc_utils": types.ModuleType("utils.reloc_utils")}
    stubs["simple_knn._C"].distCUDA2 = None
    stubs["plyfile"].PlyData = stubs["plyfile"].PlyElement = None
    stubs["utils.reloc_utils"].compute_relocation_cuda = None
    stubs["utils.reloc_utils"].compute_relocation_student_t_cuda = (
        lambda opacity_old, scale_old, nu_degree, N: relocation_student_t(opacity_old, scale_old, nu_degree, N))
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "scene.nt_model", raising=False)
    monkeypatch.delitem(sys.modules, "scene", raising=False)
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scene/nt_model.py"
    spec = importlib.util.spec_from_file_location("nt_model_cpu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NTModel


def make_model(NTModel, effective):
    o, scale, nu, N, sigma = cases(60, seed=3)
    m = NTModel(3, 100.0)
    P = o.shape[0]
    m._xyz = torch.randn(P, 3)
    m._features_dc, m._features_rest = torch.randn(P, 1, 3), torch.randn(P, 15, 3)
    m._rotation = torch.randn(P, 4)
    m._negative = torch.where(o < 0, -1.0, 1.0)[:, None]
    m._opacity = m.inverse_opacity_activation((o.abs() * 0.999)[:, None])
    m._scaling = torch.log(scale)
    m._degree = nu[:, None]
    m.filter_3D = sigma[:, None]
    m.relocate_effective = effective
    return m, N


def test_ntmodel_plain_relocation_is_unchanged(monkeypatch):
    NTModel = load_ntmodel(monkeypatch)
    m, N = make_model(NTModel, effective=False)
    idx = torch.arange(0, 60, 2)
    ratio = (N - 1)[:, None]
    out = m._update_params(idx, ratio)
    o_ref, s_ref, _ = relocate(m.get_opacity[idx, 0] * m.get_negative[idx, 0], m.get_scaling[idx],
                               m.get_nu_degree[idx, 0], ratio[idx, 0] + 1)
    assert torch.equal(out[3], m.inverse_opacity_activation(o_ref / m.get_negative[idx]))
    assert torch.equal(out[4], torch.log(s_ref))
    assert torch.equal(m.get_relocation_opacity, m.get_opacity)
    assert m.reloc_stats == {"relocated": 0, "clamped": 0}


def test_ntmodel_effective_relocation_renders_the_relocated_filtered_opacity(monkeypatch):
    NTModel = load_ntmodel(monkeypatch)
    m, N = make_model(NTModel, effective=True)
    idx = torch.arange(60)
    ratio = (N - 1)[:, None]
    sigma = m.filter_3D[:, 0]
    assert torch.equal(m.get_relocation_opacity, m.get_opacity_with_3D_filter)
    o_eff_new, _ = relocation_student_t(m.get_relocation_opacity[:, 0] * m.get_negative[:, 0], m.get_scaling,
                                        m.get_nu_degree[:, 0], ratio[:, 0] + 1)
    out = m._update_params(idx, ratio)
    rendered = torch.tanh(out[3][:, 0]) * m.get_negative[:, 0] * filter_3d_coef(torch.exp(out[4]), sigma)
    _, _, clamped = relocate(m.get_opacity[:, 0] * m.get_negative[:, 0], m.get_scaling, m.get_nu_degree[:, 0],
                             ratio[:, 0] + 1, sigma=sigma)
    ok = ~clamped & (o_eff_new.abs() >= 0.005)
    assert ok.sum() > 10
    assert torch.allclose(rendered[ok], o_eff_new[ok], rtol=1e-9)
    assert m.reloc_stats == {"relocated": 60, "clamped": int(clamped.sum())}
