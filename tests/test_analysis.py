"""M0 analysis tools: PLY round trip, SSS activations, EWA projection, tile counting, the script."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from mip_sss.analysis import (Camera, load_cameras, load_sss_model, project, quat_to_rot,
                              read_ply, tiles_touched, write_ply)

ROOT = Path(__file__).resolve().parents[1]


def fake_model(tmp_path, n=50, seed=0):
    rng = np.random.default_rng(seed)
    fields = {"x": rng.normal(size=n), "y": rng.normal(size=n), "z": rng.normal(size=n) + 6,
              "nx": np.zeros(n), "ny": np.zeros(n), "nz": np.zeros(n)}
    for i in range(3):
        fields[f"f_dc_{i}"] = rng.normal(size=n)
    for i in range(45):
        fields[f"f_rest_{i}"] = np.zeros(n)
    fields["opacity"] = rng.normal(size=n)
    for i in range(3):
        fields[f"scale_{i}"] = rng.normal(size=n) - 3
    for i in range(4):
        fields[f"rot_{i}"] = rng.normal(size=n)
    fields["degree"] = np.concatenate([[0.5, 2e4], rng.uniform(1, 200, size=n - 2)])
    fields["negative"] = np.ones(n)
    write_ply(tmp_path / "point_cloud" / "iteration_7000" / "point_cloud.ply", fields)
    write_ply(tmp_path / "point_cloud" / "iteration_30000" / "point_cloud.ply", fields)
    cams = [{"id": 0, "img_name": "a", "width": 320, "height": 240, "position": [0, 0, 0],
             "rotation": np.eye(3).tolist(), "fx": 300.0, "fy": 300.0},
            {"id": 1, "img_name": "b", "width": 320, "height": 240, "position": [0.5, 0, -1],
             "rotation": np.eye(3).tolist(), "fx": 350.0, "fy": 350.0}]
    (tmp_path / "cameras.json").write_text(json.dumps(cams))
    return fields


def test_ply_round_trip(tmp_path):
    fields = {"x": np.array([1.0, -2.5]), "opacity": np.array([0.25, 3.0])}
    write_ply(tmp_path / "a.ply", fields)
    back = read_ply(tmp_path / "a.ply")
    assert set(back) == {"x", "opacity"}
    assert np.allclose(back["x"], fields["x"]) and np.allclose(back["opacity"], fields["opacity"])


def test_load_sss_model_activations(tmp_path):
    fields = fake_model(tmp_path)
    m = load_sss_model(tmp_path)
    assert m["iteration"] == 30000
    assert torch.allclose(m["opacity"], torch.tanh(torch.tensor(fields["opacity"])))
    assert m["nu"][0] == 1.0 and m["nu"][1] == 1e4  # Hardtanh(1, 10000)
    s = torch.exp(torch.tensor(np.stack([fields[f"scale_{i}"] for i in range(3)], 1)))
    assert torch.allclose(torch.linalg.eigvalsh(m["cov3"]), torch.sort(s ** 2, dim=1).values)


def test_quaternion_matches_3dgs_identity():
    R = quat_to_rot(torch.tensor([[2.0, 0, 0, 0], [0.0, 1, 0, 0]]))
    assert torch.allclose(R[0], torch.eye(3))
    assert torch.allclose(R[1], torch.diag(torch.tensor([1.0, -1, -1])))  # 180° about x


def test_projection_of_isotropic_component_on_axis():
    cam = Camera("c", torch.eye(3), torch.zeros(3), 100.0, 100.0, 200, 100)
    xyz = torch.tensor([[0.0, 0.0, 5.0]])
    cov3 = (0.1 ** 2) * torch.eye(3)[None]
    mean2, cov2, valid = project(xyz, cov3, cam)
    assert valid.all()
    assert torch.allclose(mean2, torch.tensor([[100.0, 50.0]]))
    assert torch.allclose(cov2[0], (100 / 5 * 0.1) ** 2 * torch.eye(2))
    _, cov2_half, _ = project(xyz, cov3, cam, downscale=2)
    assert torch.allclose(cov2_half, cov2 / 4)


def test_projection_with_rotated_camera():
    # camera at the origin looking along world +x: camera z = world x (C2W columns are camera axes)
    R_c2w = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cam = Camera("c", R_c2w, torch.zeros(3), 100.0, 100.0, 200, 200)
    xyz = torch.tensor([[4.0, 0.0, 0.0], [-4.0, 0.0, 0.0]])
    cov3 = torch.diag(torch.tensor([1e-2, 4e-2, 9e-2])).expand(2, 3, 3)
    mean2, cov2, valid = project(xyz, cov3, cam)
    assert valid.tolist() == [True, False]
    # camera x = world y, camera y = world z, at depth 4 with focal 100: scale (100/4)²
    assert torch.allclose(cov2[0], torch.diag(torch.tensor([4e-2, 9e-2])) * (100 / 4) ** 2)


def test_tiles_touched():
    mean2 = torch.tensor([[8.0, 8.0], [100.0, 40.0], [-50.0, -50.0]])
    r = torch.tensor([1.0, 20.0, 10.0])
    # 64x48 image = 4x3 tiles. One tile; off-screen right (x from 80); off-screen top-left.
    assert tiles_touched(mean2, r, 64, 48).tolist() == [1.0, 0.0, 0.0]
    # x tiles 0..3, y tiles 0..2 (clamped): 4 x 3
    assert tiles_touched(torch.tensor([[32.0, 24.0]]), torch.tensor([20.0]), 64, 48)[0] == 12


def test_m0_script_runs(tmp_path):
    fake_model(tmp_path)
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "m0_component_stats.py"), str(tmp_path),
                    "--out", str(out), "--cameras", "2"], check=True, capture_output=True)
    report = json.loads((out / "component_stats.json").read_text())
    assert report["components_total"] == 50
    assert sum(report["nu"]["counts"]) == 50
    for s in report["scales"].values():
        assert s["visible_pairs"] > 0
        assert 0 <= s["option_b_share"] <= 1
        assert math.isfinite(s["estimated_eval_cost_vs_sss"])
    assert (out / "component_stats.md").exists()


def test_load_cameras(tmp_path):
    fake_model(tmp_path)
    cams = load_cameras(tmp_path)
    assert [c.name for c in cams] == ["a", "b"] and cams[1].fx == 350.0
