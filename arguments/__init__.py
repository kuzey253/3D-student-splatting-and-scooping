#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import SUPPRESS, ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True

        self.nu_degree = 10.

        self.cap_max = -1
        self.init_type = "sfm"
        self.load_allres = False  # multi-scale Blender: also train on the downscaled images (M6)

        self.C_burnin = 5e5
        self.C = 1e2
        self.burnin_iterations = 7000
        self.opacity_threshold = 0.005

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.mip_kernel_size = 0.0  # Mip-SSS 2D filter variance in px²; 0 = off (Mip-Splatting: 0.1)
        self.mip_filter_3d = False  # Mip-SSS 3D smoothing filter (Option A); off by default
        self.mip_filter_3d_variance = 0.2  # Mip-Splatting's 3D filter scale: σ3 = sqrt(this) / f̂
        self.mip_option_b = False  # Mip-SSS Option B (M4): quadrature + fast path in CUDA; off by default
        self.mip_exact_nu_grad = False  # M5 diagnostic: SSS-kernel ν gradient with SSS's omitted term
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000

        # default
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.005
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False

        self.degree_lr = 10.0
        self.negetive_lr = 10.0

        self.scale_reg = 0.01
        self.opacity_reg = 0.01
        # Mip-SSS M5, both need --mip_filter_3d; off by default
        self.mip_relocate_effective = False  # recycle/relocate on the 3D-filtered opacity (README M5.1)
        self.mip_reg_filtered = False  # scale regularizer on the 3D-filtered scales (README M5.2)

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def explicit_args(parser, argv):
    """Destinations given on the command line (any spelling argparse accepts, e.g. -r, --resolution)."""
    saved = [(a, a.default) for a in parser._actions]
    try:
        for a, _ in saved:
            a.default = SUPPRESS
        return set(vars(parser.parse_args(argv)))
    finally:
        for a, d in saved:
            a.default = d
