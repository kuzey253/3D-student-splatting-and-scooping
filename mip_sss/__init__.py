"""Mip-SSS: alias-free Student Splatting and Scooping. Pure-PyTorch reference (milestone M1)."""
from .kernels import (gauss_mip, mass_option_b, mip_gauss_3d, mip_sss, option_a, option_a_3d,
                      option_b, option_b_3d, radius_option_b, sss_kernel)
from .quadrature import QuadConfig, nodes
from . import design  # noqa: E402  (the configuration chosen at the M1 gate)
