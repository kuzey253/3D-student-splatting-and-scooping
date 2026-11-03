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
    configs = {p.stem for p in (ROOT / "configs").glob("*.json")} - {"blender"}  # shared by the Blender scenes
    assert configs <= set(remote.SCENE_ARCHIVE)
    assert {s for s, a in remote.SCENE_ARCHIVE.items() if a == "blender"} == set(remote.BLENDER_SCENES)


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


FAKE_COLAB = r'''#!/usr/bin/env python3
"""Fake colab CLI: answers remote steps from a script of status replies, records every call."""
import json, os, sys, tarfile, io
state_dir = os.environ["FAKE_STATE"]
args = sys.argv[1:]
with open(os.path.join(state_dir, "calls.txt"), "a") as f:
    f.write(" ".join(args) + "\n")
envs = dict(a.split("=", 1) for a, k in zip(args[1:], args) if k == "--env")
step = envs.get("STEP")
if args[0] == "exec" and step:
    if step == "status":
        n_file = os.path.join(state_dir, "status_n")
        n = int(open(n_file).read()) if os.path.exists(n_file) else 0
        open(n_file, "w").write(str(n + 1))
        replies = os.environ["FAKE_STATUS"].split()
        r = replies[min(n, len(replies) - 1)]
        if r == "lost":
            print("[colab] Session 'g4' not found.")
            sys.exit(0)
        st, done = r.split(":")
        print(f"M0_STATE={st}\nM0_DONE={done}\nM0_PROGRESS=fake")
    elif step == "collect":
        open(os.path.join(state_dir, "collects"), "a").write("x")
    print(f"M0_{step.upper()}=OK")
elif args[0] == "download":
    k = len(open(os.path.join(state_dir, "collects")).read())
    with tarfile.open(args[-1], "w:gz") as t:
        for i in range(k):
            b = json.dumps({"run": f"r{i}"}).encode()
            ti = tarfile.TarInfo(f"r{i}/summary.json"); ti.size = len(b)
            t.addfile(ti, io.BytesIO(b))
'''

FAKE_ATTACH = r'''#!/usr/bin/env python3
import os, sys
open(os.path.join(os.environ["FAKE_STATE"], "attach.txt"), "a").write(" ".join(sys.argv[1:]) + "\n")
print("[!] The backend reports no active runtimes." if os.environ.get("FAKE_GONE") else
      "[+] Attached 'g4' -> gpu-g4-fake (G4, STANDARD)\n[+] keep-alive daemon pid=12345")
'''


def run_runner(tmp_path, status, gone=False):
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    (bin_dir / "colab").write_text(FAKE_COLAB)
    (bin_dir / "colab").chmod(0o755)
    attach = tmp_path / "attach" / "scripts" / "attach_session.py"
    attach.parent.mkdir(parents=True)
    attach.write_text(FAKE_ATTACH)
    env = dict(__import__("os").environ, PATH=f"{bin_dir}:/usr/bin:/bin", SESSION="g4", POLL="0", RETRY="0",
               OUT=str(tmp_path / "out"), ATTACH=str(attach), FAKE_STATE=str(state), FAKE_STATUS=status,
               SCENES="lego", RUNS="a:lego b:lego c:lego", **({"FAKE_GONE": "1"} if gone else {}))
    p = subprocess.run(["bash", str(ROOT / "scripts" / "colab" / "run_m0.sh")], env=env, capture_output=True,
                       text=True, timeout=120)
    attaches = (state / "attach.txt").read_text().splitlines() if (state / "attach.txt").exists() else []
    return p, tmp_path / "out", attaches


def test_runner_collects_each_finished_run_and_reattaches_a_lost_session(tmp_path):
    p, out, attaches = run_runner(tmp_path, "running:0 running:1 lost lost running:2 done:3")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "collected 1 finished run(s)" in p.stdout and "collected 2 finished run(s)" in p.stdout
    assert len(attaches) == 2 and all("--name g4 --require-large --keep-alive" in a for a in attaches)
    assert "keep-alive daemon pid=12345" in p.stdout
    assert sorted(d.name for d in out.iterdir() if (d / "summary.json").exists()) == ["r0", "r1", "r2"]


def test_runner_stops_with_collected_runs_when_the_vm_is_gone(tmp_path):
    p, out, attaches = run_runner(tmp_path, "running:1 lost", gone=True)
    assert p.returncode == 1 and "the VM is gone" in p.stdout
    assert len(attaches) == 1 and (out / "r0" / "summary.json").exists()
