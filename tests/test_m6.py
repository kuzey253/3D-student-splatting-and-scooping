"""M6 tooling: the -r fix, the multi-scale Blender converter and reader, per-scale metrics, the
Blender data check (PLAN M6, items 3 and 10)."""
import importlib.util
import json
import sys
import types
import zipfile
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "colab"))

from arguments import ModelParams, OptimizationParams, PipelineParams, explicit_args  # noqa: E402


def train_parser():
    """train.py's parser."""
    parser = ArgumentParser()
    ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    return parser


def apply_config(argv, config):
    """train.py's config handling."""
    parser = train_parser()
    args = parser.parse_args(argv)
    given = explicit_args(parser, argv)
    for key, value in config.items():
        if key not in given:
            setattr(args, key, value)
    return args, parser


@pytest.mark.parametrize("argv", [["-r", "8"], ["--resolution", "8"], ["--resolution=8"], ["--resol", "8"]])
def test_cli_flags_win_over_the_config(argv):
    args, _ = apply_config(argv, {"resolution": 4, "cap_max": 3_000_000})
    assert args.resolution == 8 and args.cap_max == 3_000_000


def test_without_cli_flags_the_config_wins_as_in_sss():
    args, _ = apply_config(["-s", "x"], {"resolution": 4, "iterations": 40000, "white_background": True})
    assert (args.resolution, args.iterations, args.white_background) == (4, 40000, True)


def test_explicit_args_restores_the_parser_defaults():
    args, parser = apply_config(["--eval"], {"resolution": 2})
    assert {"eval"} == explicit_args(parser, ["--eval"])
    fresh = parser.parse_args([])
    assert fresh.resolution == -1 and fresh.iterations == 30_000 and fresh.white_background is False


# --- multi-scale Blender -----------------------------------------------------------------------

def make_blender_scene(root, n=(2, 1, 2), size=16):
    """A tiny NeRF-synthetic scene: RGBA images and transforms_{train,val,test}.json."""
    g = np.random.default_rng(0)
    for split, k in zip(("train", "val", "test"), n):
        (root / split).mkdir(parents=True)
        frames = []
        for i in range(k):
            img = g.integers(0, 256, (size, size, 4), dtype=np.uint8)
            Image.fromarray(img, "RGBA").save(root / split / f"r_{i}.png")
            a = g.normal(size=3)
            R = np.linalg.qr(g.normal(size=(3, 3)))[0]
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = R, 4 * a / np.linalg.norm(a)
            frames.append({"file_path": f"./{split}/r_{i}", "transform_matrix": c2w.tolist()})
        (root / f"transforms_{split}.json").write_text(json.dumps({"camera_angle_x": 0.69, "frames": frames}))


def load_readers(monkeypatch):
    """scene/dataset_readers.py without scene/__init__ and without plyfile or the CUDA modules."""
    from utils.graphics_utils import BasicPointCloud
    stubs = {"plyfile": types.ModuleType("plyfile"), "scene.nt_model": types.ModuleType("scene.nt_model"),
             "scene": types.ModuleType("scene")}
    stubs["plyfile"].PlyData = stubs["plyfile"].PlyElement = None
    stubs["scene.nt_model"].BasicPointCloud = BasicPointCloud
    stubs["scene"].__path__ = [str(ROOT / "scene")]
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "scene.colmap_loader", raising=False)
    spec = importlib.util.spec_from_file_location("dataset_readers_cpu", ROOT / "scene/dataset_readers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.storePly = lambda *a, **k: None
    mod.fetchPly = lambda path: None
    return mod


def converted(tmp_path):
    import convert_blender_data
    raw, out = tmp_path / "nerf_synthetic" / "toy", tmp_path / "ms" / "toy"
    make_blender_scene(raw)
    convert_blender_data.convert_to_nerfdata(str(raw), str(out), 4)
    return raw, out


def test_converter_writes_four_box_downsampled_scales(tmp_path):
    raw, out = converted(tmp_path)
    meta = json.loads((out / "metadata.json").read_text())
    test = meta["test"]
    assert test["label"] == [0, 1, 2, 3] * 2 and test["width"] == [16, 8, 4, 2] * 2
    assert test["focal"][1] == test["focal"][0] / 2 and test["lossmult"][:4] == [1, 4, 16, 64]
    full = np.asarray(Image.open(raw / "test" / "r_1.png"), dtype=np.float64) / 255
    d1 = np.asarray(Image.open(out / test["file_path"][5]), dtype=np.float64) / 255
    box = full.reshape(8, 2, 8, 2, 4).mean((1, 3))
    assert np.abs(d1 - box).max() <= 1 / 255 + 1e-12  # 2×2 box mean, quantized to 8 bits


def test_multiscale_reader_matches_the_blender_reader_at_full_resolution(tmp_path, monkeypatch):
    dr = load_readers(monkeypatch)
    raw, out = converted(tmp_path)
    info = dr.readMultiScaleNerfSyntheticInfo(str(out), True, True)
    assert [c.image_name for c in info.train_cameras] == ["000_d0", "001_d0"]  # full resolution only
    assert [c.image_name for c in info.test_cameras] == [f"00{i}_d{j}" for i in range(2) for j in range(4)]
    ref = dr.readCamerasFromTransforms(str(raw), "transforms_test.json", True)
    for j in range(4):
        cam = info.test_cameras[4 + j]
        assert np.allclose(cam.R, ref[1].R) and np.allclose(cam.T, ref[1].T)
        assert cam.width == 16 >> j and abs(cam.FovX - ref[1].FovX) < 1e-12  # focal and width both halve
    assert np.array_equal(np.asarray(info.test_cameras[4].image), np.asarray(ref[1].image))
    allres = dr.readMultiScaleNerfSyntheticInfo(str(out), True, True, load_allres=True)
    assert len(allres.train_cameras) == 8


# --- per-scale metrics -------------------------------------------------------------------------

def write_pngs(d, imgs):
    d.mkdir(parents=True)
    for i, a in enumerate(imgs):
        Image.fromarray(a).save(d / f"{i:05d}.png")


def test_per_scale_metrics_split_blender_renders_by_label(tmp_path):
    import m6_metrics
    g = np.random.default_rng(1)
    labels = [0, 1, 2, 3] * 2
    gts = [g.integers(0, 256, (16 >> j, 16 >> j, 3), dtype=np.uint8) for j in labels]
    noise = [1, 2, 8, 30]
    rs = [np.clip(a.astype(int) + g.integers(-noise[j], noise[j] + 1, a.shape), 0, 255).astype(np.uint8)
          for a, j in zip(gts, labels)]
    rs[0] = np.clip(rs[0].astype(int) + 1, 0, 255).astype(np.uint8)  # avoid a zero-error image
    write_pngs(tmp_path / "r", rs)
    write_pngs(tmp_path / "g", gts)
    meta = {"test": {"label": labels, "width": [16 >> j for j in labels], "height": [16 >> j for j in labels]}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    lab = m6_metrics.blender_labels(tmp_path / "metadata.json")
    res = m6_metrics.score([("all", tmp_path / "r", tmp_path / "g")], lab, use_lpips=False)
    assert list(res["scales"]) == ["0", "1", "2", "3"] and all(s["n"] == 2 for s in res["scales"].values())
    psnrs = [res["scales"][k]["PSNR"] for k in "0123"]
    assert psnrs == sorted(psnrs, reverse=True)  # more noise at coarser scales here
    from utils.image_utils import psnr
    want = np.mean([float(psnr(torch.tensor(r / 255.).permute(2, 0, 1)[None].float().contiguous(),
                                torch.tensor(gt / 255.).permute(2, 0, 1)[None].float().contiguous()))
                    for r, gt, j in zip(rs, gts, labels) if j == 2])
    assert abs(res["scales"]["2"]["PSNR"] - want) < 1e-4
    assert abs(res["all"]["PSNR"] - np.mean(psnrs)) < 1e-9  # equal counts per scale
    bad = dict(meta)
    bad["test"] = dict(meta["test"], width=[16] * 8)
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="metadata says"):
        m6_metrics.score([("all", tmp_path / "r", tmp_path / "g")], m6_metrics.blender_labels(tmp_path / "bad.json"),
                         use_lpips=False)


def test_per_scale_metrics_zoom_in_pairs(tmp_path):
    import m6_metrics
    g = np.random.default_rng(2)
    for r in (8, 1):
        a = g.integers(0, 256, (3, 24 // r + 4, 3), dtype=np.uint8)
        write_pngs(tmp_path / f"g{r}", [a, a])
        write_pngs(tmp_path / f"r{r}", [255 - a, a ^ 1])
    res = m6_metrics.score([(str(r), tmp_path / f"r{r}", tmp_path / f"g{r}") for r in (8, 1)], use_lpips=False)
    assert list(res["scales"]) == ["8", "1"] and res["all"]["n"] == 4
    (tmp_path / "r1" / "00001.png").unlink()
    with pytest.raises(ValueError, match="renders vs"):
        m6_metrics.score([("1", tmp_path / "r1", tmp_path / "g1")], use_lpips=False)


# --- the Blender download check ----------------------------------------------------------------

def test_blender_manifest_ignores_extras_and_detects_changes(tmp_path):
    import remote_m0
    raw = tmp_path / "nerf_synthetic" / "toy"
    make_blender_scene(raw, n=(1, 1, 1), size=4)
    (raw / "test" / "r_0_depth_0000.png").write_bytes(b"x")

    def build(path, extra=b""):
        with zipfile.ZipFile(path, "w") as z:
            for f in sorted(raw.parent.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(tmp_path).as_posix())
            z.writestr("nerf_synthetic/.DS_Store", b"junk")
            z.writestr("__MACOSX/nerf_synthetic/toy/._r_0.png", b"junk")
            if extra:
                z.writestr("nerf_synthetic/toy/train/r_0.png", extra)
        with zipfile.ZipFile(path) as z:
            return remote_m0.blender_members(z.infolist())

    keep, d1 = build(tmp_path / "a.zip")
    assert sorted(keep) == ["toy/test/r_0.png", "toy/train/r_0.png", "toy/transforms_test.json",
                            "toy/transforms_train.json", "toy/transforms_val.json", "toy/val/r_0.png"]
    _, d2 = build(tmp_path / "b.zip")
    assert d1 == d2
    with pytest.warns(UserWarning):  # zipfile warns about the duplicate name
        _, d3 = build(tmp_path / "c.zip", extra=b"tampered")
    assert d3 != d1


# --- the pipeline's M6 paths, with every subprocess faked ---------------------------------------

def fake_pipeline(tmp_path, monkeypatch):
    import m0_pipeline as mp
    monkeypatch.setattr(mp, "M0", tmp_path / "m0")
    monkeypatch.setattr(mp, "DATA", tmp_path / "data")
    (tmp_path / "data" / "lego").mkdir(parents=True)
    (tmp_path / "data" / "lego" / "metadata.json").write_text("{}")
    monkeypatch.setattr(mp.subprocess, "Popen", lambda *a, **k: types.SimpleNamespace(terminate=lambda: None))
    monkeypatch.setattr(mp, "run_in", lambda cwd, cmd, log, status, phase, scene: fake(cmd, log, cwd))
    calls = []

    def arg(cmd, flag):
        return cmd[cmd.index(flag) + 1] if flag in cmd else None

    def fake(cmd, log, cwd=None):
        calls.append((cwd, cmd))
        Path(log).write_text("")
        script = Path(cmd[1]).name
        model = Path(arg(cmd, "-m")) if "-m" in cmd else None
        if script == "train.py":
            cfg = json.loads(Path(arg(cmd, "--config")).read_text()) if "--config" in cmd else {}
            it = int(arg(cmd, "--iterations") or cfg["iterations"])
            ply = model / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"
            ply.parent.mkdir(parents=True)
            ply.write_bytes(b"ply\nelement vertex 7\nend_header\n")
            (model / "cfg_args").write_text("Namespace()")
        elif script == "render.py":
            it = max(int(p.name.split("_")[1]) for p in (model / "point_cloud").iterdir())
            out = model / "test" / f"ours_{it}"
            r = arg(cmd, "-r")
            names = (f"test_preds_{r or -1}", f"gt_{r or -1}") if cwd else ("renders", "gt")
            for n in names:
                (out / n).mkdir(parents=True)
                (out / n / "00000.png").write_bytes(b"")
        elif script == "m6_metrics.py":
            pairs = [p.split("=", 1)[0] for p, f in zip(cmd[1:], cmd) if f == "--pair"]
            s = {"PSNR": 30.0, "SSIM": 0.9, "LPIPS": 0.1, "n": 1}
            Path(arg(cmd, "--out")).write_text(json.dumps({"scales": {p: s for p in pairs}, "all": s}))
            Path(log).write_text("M6_METRICS=OK\n")
        elif script in ("m0_fps.py", "m6_mip_fps.py"):
            Path(arg(cmd, "--out")).write_text(json.dumps({"fps": 100.0, "size": [8, 8]}))
        return 1.0

    monkeypatch.setattr(mp, "run", lambda cmd, log, status, phase, scene: fake(cmd, log))
    status = types.SimpleNamespace(update=lambda **kw: None)
    return mp, calls, status


def test_pipeline_sss_zoom_in_trains_at_r8_and_renders_four_scales(tmp_path, monkeypatch):
    mp, calls, status = fake_pipeline(tmp_path, monkeypatch)
    monkeypatch.chdir(ROOT)
    s = mp.run_scene("bonsai", False, status, "bonsai_a", {"protocol": "zoomin", "mip_kernel_size": 0.1,
                                                           "mip_filter_3d": True})
    cfg = json.loads((tmp_path / "m0" / "bonsai_a" / "config.json").read_text())
    assert cfg["resolution"] == 8 and "protocol" not in cfg and cfg["cap_max"] == 1300000
    renders = [c for _, c in calls if Path(c[1]).name == "render.py"]
    assert [c[c.index("-r") + 1] for c in renders] == ["8", "4", "2", "1"]
    assert all("--mip_filter_3d" in c and "cpu" in c for c in renders)
    out = tmp_path / "m0" / "bonsai_a" / "model" / "test" / "ours_40000"
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{a}_r{r}" for a in ("renders", "gt") for r in (8, 4, 2, 1))
    assert list(s["per_scale"]) == ["8", "4", "2", "1"] and s["protocol"] == "zoomin"
    fps = next(c for _, c in calls if Path(c[1]).name == "m0_fps.py")
    assert "--only_suffix" not in fps


def test_pipeline_mip_blender_uses_its_scripts_arguments(tmp_path, monkeypatch):
    mp, calls, status = fake_pipeline(tmp_path, monkeypatch)
    monkeypatch.chdir(ROOT)
    s = mp.run_scene("lego", True, status, "lego_mip", {"protocol": "stmt", "method": "mip"})
    cwd, train = calls[0]
    assert cwd == mp.MIP and Path(train[1]).name == "train.py"
    assert {"--eval", "--white_background"} <= set(train) and train[train.index("--kernel_size") + 1] == "0.1"
    assert train[train.index("--iterations") + 1] == "1000"  # dry
    metrics = next(c for _, c in calls if Path(c[1]).name == "m6_metrics.py")
    assert "--metadata" in metrics and any(a.endswith("test_preds_-1:" + str(tmp_path / "m0/lego_mip/model/test/ours_1000/gt_-1"))
                                           for a in metrics)
    fps = next(c for _, c in calls if Path(c[1]).name == "m6_mip_fps.py")
    assert "_d0" in fps and s["method"] == "mip" and s["components"] == 7
    with pytest.raises(mp.PhaseError, match="unsupported"):
        mp.run_scene("lego", True, status, "lego_mip2", {"protocol": "stmt", "method": "mip", "cap_max": 5})


def test_pipeline_sss_blender_uses_the_blender_config(tmp_path, monkeypatch):
    mp, calls, status = fake_pipeline(tmp_path, monkeypatch)
    monkeypatch.chdir(ROOT)
    mp.run_scene("lego", False, status, "lego_sss", {"protocol": "stmt"})
    cfg = json.loads((tmp_path / "m0" / "lego_sss" / "config.json").read_text())
    assert cfg["white_background"] is True and cfg["iterations"] == 30000 and cfg["cap_max"] == 300000
    renders = [c for _, c in calls if Path(c[1]).name == "render.py"]
    assert len(renders) == 1 and "-r" not in renders[0]
    fps = next(c for _, c in calls if Path(c[1]).name == "m0_fps.py")
    assert fps[fps.index("--only_suffix") + 1] == "_d0"
