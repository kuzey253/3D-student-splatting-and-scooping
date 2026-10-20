"""Local checks of the Colab M0 runner pieces that need no VM."""
import importlib.util
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "colab" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            if n.endswith("/"):
                z.writestr(n, "")
            else:
                z.writestr(n, n)


def test_extract_scene_handles_both_archive_layouts(tmp_path):
    remote = load("remote_m0")
    make_zip(tmp_path / "tandt.zip", ["tandt/train/images/a.jpg", "tandt/train/sparse/0/cameras.bin",
                                      "tandt/truck/images/b.jpg", "tandt/truck/sparse/0/cameras.bin",
                                      "db/playroom/images/c.jpg"])
    make_zip(tmp_path / "360.zip", ["garden/", "garden/images/x.JPG", "garden/images_4/x.JPG",
                                    "garden/sparse/0/points3D.bin", "garden/poses_bounds.npy",
                                    "bicycle/images/y.JPG", "bicycle/sparse/0/points3D.bin"])
    assert remote.extract_scene(tmp_path / "tandt.zip", "train", tmp_path / "train") == 2
    assert (tmp_path / "train" / "images" / "a.jpg").exists()
    assert not (tmp_path / "train" / "truck").exists()
    assert remote.extract_scene(tmp_path / "360.zip", "garden", tmp_path / "garden") == 2
    assert sorted(p.relative_to(tmp_path / "garden").as_posix()
                  for p in (tmp_path / "garden").rglob("*") if p.is_file()) == [
        "images/x.JPG", "sparse/0/points3D.bin"]  # no images_4, no poses_bounds


def test_every_config_has_a_known_archive():
    remote = load("remote_m0")
    configs = {p.stem for p in (ROOT / "configs").glob("*.json")}
    assert configs <= set(remote.SCENE_ARCHIVE)


def test_runner_shell_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / "colab" / "run_m0.sh")], check=True)


def test_package_contents(tmp_path):
    """The packaging lines of run_m0.sh: tracked + untracked files, no runbook, no results, no ._"""
    script = (ROOT / "scripts" / "colab" / "run_m0.sh").read_text()
    start = script.index('PKG_DIR=$(mktemp -d)')
    end = script.index('gzip "$PKG_DIR/sss.tar"') + len('gzip "$PKG_DIR/sss.tar"')
    (tmp_path / "out").mkdir()
    (tmp_path / "pkg").mkdir()
    snippet = (f'set -euo pipefail\nROOT="{ROOT}"\nOUT="{tmp_path / "out"}"\n'
               + script[start:end].replace("PKG_DIR=$(mktemp -d)", f'PKG_DIR="{tmp_path / "pkg"}"')
               + "\n")
    subprocess.run(["bash", "-c", snippet], check=True)
    with tarfile.open(tmp_path / "pkg" / "sss.tar.gz") as t:
        names = set(t.getnames())
    assert {"train.py", "mip_sss/design.py", "scripts/colab/m0_pipeline.py", "GIT_STATE.txt",
            "submodules/diff-t-rasterization/rasterize_points.cu"} <= names
    assert not any(n.startswith("results/") or n == "COLAB_GPU_RUNBOOK.md" for n in names)
    assert not any(Path(n).name.startswith("._") for n in names)


def test_remote_scripts_compile():
    for name in ("remote_m0", "m0_pipeline", "m0_fps"):
        subprocess.run([sys.executable, "-m", "py_compile",
                        str(ROOT / "scripts" / "colab" / f"{name}.py")], check=True)
