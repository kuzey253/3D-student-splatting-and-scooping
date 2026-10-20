"""Load a trained SSS model and project its components into its cameras (CPU, float64).

Mirrors SSS's activations (scene/nt_model.py) and its EWA projection (forward.cu computeCov2D:
the 1.3·tan(fov) clamp, no dilation) and tile rectangles (getRect, 16x16 tiles). Used by
scripts/m0_component_stats.py; needs no CUDA and no SSS imports.
"""
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1",
             "uint8": "u1", "char": "i1", "int8": "i1", "short": "i2", "int16": "i2",
             "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4",
             "uint32": "u4"}
BLOCK = 16


def read_ply(path):
    """The vertex element of a binary PLY with scalar properties, as {name: float64 array}."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    fmt = header[1].split()[1]
    if fmt not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError(f"unsupported PLY format {fmt}")
    order = "<" if fmt == "binary_little_endian" else ">"
    n, props, in_vertex = 0, [], False
    for line in header:
        tok = line.split()
        if tok[0] == "element":
            in_vertex = tok[1] == "vertex"
            if in_vertex:
                n = int(tok[2])
        elif tok[0] == "property" and in_vertex:
            if tok[1] == "list":
                raise ValueError("list properties are not supported")
            props.append((tok[2], order + PLY_TYPES[tok[1]]))
    arr = np.frombuffer(data, dtype=np.dtype(props), count=n, offset=end)
    return {name: arr[name].astype(np.float64) for name, _ in props}


def write_ply(path, fields):
    """Write {name: 1D array} as a binary little-endian float32 vertex PLY (as SSS's save_ply)."""
    names = list(fields)
    n = len(fields[names[0]])
    arr = np.empty(n, dtype=[(k, "<f4") for k in names])
    for k in names:
        arr[k] = fields[k]
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
              + "".join(f"property float {k}\n" for k in names) + "end_header\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(header.encode("ascii") + arr.tobytes())


def quat_to_rot(q):
    """SSS/3DGS build_rotation: q = (r, x, y, z), normalized."""
    q = q / q.norm(dim=1, keepdim=True)
    r, x, y, z = q.unbind(1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)], -1),
        torch.stack([2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)], -1),
        torch.stack([2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def latest_iteration(model_dir):
    its = [int(m.group(1)) for p in (Path(model_dir) / "point_cloud").glob("iteration_*")
           if (m := re.fullmatch(r"iteration_(\d+)", p.name))]
    if not its:
        raise FileNotFoundError(f"no point_cloud/iteration_* under {model_dir}")
    return max(its)


def load_sss_model(model_dir, iteration=None):
    """Activated parameters of a saved SSS model (nt_model.py activations)."""
    iteration = latest_iteration(model_dir) if iteration is None else iteration
    v = read_ply(Path(model_dir) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply")
    t = {k: torch.from_numpy(a) for k, a in v.items()}
    xyz = torch.stack([t["x"], t["y"], t["z"]], 1)
    scales = torch.exp(torch.stack([t[f"scale_{i}"] for i in range(3)], 1))
    R = quat_to_rot(torch.stack([t[f"rot_{i}"] for i in range(4)], 1))
    cov3 = R @ torch.diag_embed(scales ** 2) @ R.transpose(1, 2)
    return {
        "iteration": iteration,
        "xyz": xyz,
        "cov3": cov3,
        "scales": scales,
        "opacity": torch.tanh(t["opacity"]) * t["negative"],
        "nu": t["degree"].clamp(1.0, 1e4),  # Hardtanh(1, 10000)
    }


@dataclass
class Camera:
    name: str
    R_c2w: torch.Tensor
    center: torch.Tensor
    fx: float
    fy: float
    width: int
    height: int


def load_cameras(model_dir):
    """cameras.json as written by SSS (camera_to_JSON): camera-to-world rotation and centre."""
    entries = json.loads((Path(model_dir) / "cameras.json").read_text())
    return [Camera(e["img_name"], torch.tensor(e["rotation"], dtype=torch.float64),
                   torch.tensor(e["position"], dtype=torch.float64), e["fx"], e["fy"],
                   e["width"], e["height"]) for e in entries]


def project(xyz, cov3, cam, downscale=1.0):
    """Screen means (N, 2), 2D covariances (N, 2, 2) and depth-validity (N,) at 1/downscale res."""
    fx, fy = cam.fx / downscale, cam.fy / downscale
    W, H = cam.width / downscale, cam.height / downscale
    R = cam.R_c2w.T  # world to camera
    t = (xyz - cam.center) @ R.T
    tz = t[:, 2]
    valid = tz > 0.2
    tz = torch.where(valid, tz, torch.ones_like(tz))
    limx, limy = 1.3 * W / (2 * fx), 1.3 * H / (2 * fy)
    tx = torch.clamp(t[:, 0] / tz, -limx, limx) * tz
    ty = torch.clamp(t[:, 1] / tz, -limy, limy) * tz
    zero = torch.zeros_like(tz)
    J = torch.stack([torch.stack([fx / tz, zero, -fx * tx / tz ** 2], -1),
                     torch.stack([zero, fy / tz, -fy * ty / tz ** 2], -1)], 1)
    T = J @ R
    cov2 = T @ cov3 @ T.transpose(1, 2)
    mean2 = torch.stack([fx * t[:, 0] / tz + W / 2, fy * t[:, 1] / tz + H / 2], 1)
    return mean2, cov2, valid


def tiles_touched(mean2, radius, width, height):
    """Number of 16x16 tiles in the screen rectangle mean ± radius (3DGS getRect), clamped."""
    gx, gy = math.ceil(width / BLOCK), math.ceil(height / BLOCK)
    x0 = torch.clamp(((mean2[:, 0] - radius) / BLOCK).floor(), 0, gx)
    x1 = torch.clamp(((mean2[:, 0] + radius + BLOCK - 1) / BLOCK).floor(), 0, gx)
    y0 = torch.clamp(((mean2[:, 1] - radius) / BLOCK).floor(), 0, gy)
    y1 = torch.clamp(((mean2[:, 1] + radius + BLOCK - 1) / BLOCK).floor(), 0, gy)
    return (x1 - x0) * (y1 - y0)
