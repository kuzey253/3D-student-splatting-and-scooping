"""The two brute-force references must agree with closed forms and with each other."""
import pytest
import torch

from mip_sss import gauss_mip, sss_kernel
from mip_sss.reference import lambda_integral
from mip_sss.testing import COV_CASES, NUS_EXTENDED, S_MIP, fft_reference, ray_points, rot_cov


@pytest.mark.parametrize("nu", NUS_EXTENDED)
@pytest.mark.parametrize("case", range(len(COV_CASES)))
def test_lambda_integral_is_sss_kernel_at_s0(nu, case):
    cov = rot_cov(*COV_CASES[case])[None]
    u, mu, nut = ray_points(cov[0], nu, 0.0), torch.zeros(1, 2), torch.tensor([nu])
    ref = lambda_integral(u, mu, cov, nut, 0.0)
    exact = sss_kernel(u, mu, cov, nut)
    m = exact > 1e-12
    assert ((ref - exact).abs() / exact)[m].max() < 1e-10


def test_lambda_integral_grid_is_converged():
    cov = rot_cov(*COV_CASES[0])[None]
    u, mu, nut = ray_points(cov[0], 1.0, S_MIP), torch.zeros(1, 2), torch.tensor([1.0])
    a = lambda_integral(u, mu, cov, nut, S_MIP)
    b = lambda_integral(u, mu, cov, nut, S_MIP, per_width=80, left=40.0, right=30.0, tau=1e-18)
    m = b > 1e-7
    assert ((a - b).abs() / b)[m].max() < 1e-12


@pytest.mark.parametrize("case", [0, 3])
def test_fft_reproduces_mip_gaussian(case):
    cov, u, val = fft_reference(*COV_CASES[case], None, S_MIP, kernel="gaussian")
    g = gauss_mip(u, torch.zeros(1, 2), cov[None], S_MIP)[0]
    m = g > 1e-8
    assert (val - g).abs().max() < 1e-12
    assert ((val - g).abs() / g)[m].max() < 1e-6


@pytest.mark.parametrize("nu", [1.0, 4.0, 30.0])
@pytest.mark.parametrize("case", [0, 1, 2, 4])
def test_fft_agrees_with_lambda_integral(nu, case):
    cov, u, val = fft_reference(*COV_CASES[case], nu, S_MIP)
    ref = lambda_integral(u, torch.zeros(1, 2), cov[None], torch.tensor([nu]), S_MIP)[0]
    m = ref > 1e-6
    assert m.sum() > 100
    assert ((val - ref).abs() / ref)[m].max() < 1e-9
