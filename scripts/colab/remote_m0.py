"""Remote side of scripts/colab/run_m0.sh, executed in the Colab kernel with `colab exec -f`.

The STEP env var selects the step: setup, data, parity, m3, m4, m5, launch, status or collect. Every step ends with a
line `M0_<STEP>=OK` or `M0_<STEP>=FAIL <reason>`, because `colab exec` exits 0 even when remote
code fails: the runner reads these answers, not exit codes.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path("/content")
SSS = ROOT / "sss"
DATA = ROOT / "data"
M0 = ROOT / "m0"
URLS = {
    "tandt": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip",
    "360": "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
}
SCENE_ARCHIVE = {"train": "tandt", "truck": "tandt", "drjohnson": "tandt", "playroom": "tandt",
                 "bicycle": "360", "garden": "360", "stump": "360", "bonsai": "360",
                 "counter": "360", "kitchen": "360", "room": "360"}


def sh(cmd, log=None, **kw):
    """Run a command, echo its tail on failure, return CompletedProcess."""
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if log:
        Path(log).write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode:
        print("\n".join((p.stdout + p.stderr).splitlines()[-60:]))
    return p


def scene_members(names, scene):
    """Zip members for <scene>/images/* and <scene>/sparse/*, wherever the scene folder sits.

    Returns (prefix, members): prefix is the path up to and including '<scene>/'.
    """
    marker = f"{scene}/sparse/"
    hit = next((n for n in names if marker in n), None)
    if hit is None:
        return None, []
    prefix = hit[: hit.index(marker) + len(scene) + 1]
    keep = (prefix + "images/", prefix + "sparse/")
    return prefix, [n for n in names if n.startswith(keep) and not n.endswith("/")]


def extract_scene(zip_path, scene, dest):
    with zipfile.ZipFile(zip_path) as z:
        prefix, members = scene_members(z.namelist(), scene)
        if not members:
            raise RuntimeError(f"{scene} not found in {zip_path.name}")
        for n in members:
            target = dest / n[len(prefix):]
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    return len(members)


def step_setup():
    pkg = ROOT / "sss.tar.gz"
    if not pkg.exists():
        return "FAIL no /content/sss.tar.gz uploaded"
    shutil.rmtree(SSS, ignore_errors=True)
    SSS.mkdir(parents=True)
    with tarfile.open(pkg) as t:
        t.extractall(SSS)
    M0.mkdir(parents=True, exist_ok=True)

    import torch
    cap = torch.cuda.get_device_capability(0)
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": f"{cap[0]}.{cap[1]}",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "nvcc": sh([nvcc, "--version"]).stdout.strip().splitlines()[-1] if Path(nvcc).exists() else None,
        "driver": sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).stdout.strip(),
        "git": (SSS / "GIT_STATE.txt").read_text() if (SSS / "GIT_STATE.txt").exists() else None,
        "setup_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (M0 / "env.json").write_text(json.dumps(env, indent=1))
    print(json.dumps(env, indent=1))
    if env["nvcc"] is None:
        return "FAIL nvcc not found"

    if sh([sys.executable, "-m", "pip", "install", "-q", "plyfile", "tqdm"]).returncode:
        return "FAIL pip install plyfile tqdm"
    build_env = dict(os.environ, TORCH_CUDA_ARCH_LIST=env["capability"], MAX_JOBS="16",
                     PATH=f"{Path(nvcc).parent}:{os.environ['PATH']}")
    t0 = time.time()
    p = sh([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-v",
            "./submodules/diff-t-rasterization", "./submodules/simple-knn"],
           log=M0 / "build.log", cwd=SSS, env=build_env)
    print(f"build: {time.time() - t0:.0f}s, exit {p.returncode}")
    if p.returncode:
        return "FAIL extension build (see /content/m0/build.log)"
    check = ("import torch, diff_t_rasterization; from simple_knn._C import distCUDA2; "
             "print('knn', distCUDA2(torch.rand(64, 3, device='cuda')).shape)")
    p = sh([sys.executable, "-c", check], cwd="/tmp")
    print(p.stdout.strip())
    return "OK" if p.returncode == 0 else "FAIL extension import"


def step_data(scenes):
    DATA.mkdir(parents=True, exist_ok=True)
    for archive in sorted({SCENE_ARCHIVE[s] for s in scenes}):
        z = DATA / f"{archive}.zip"
        if not z.exists():
            t0 = time.time()
            p = sh(["curl", "-L", "--fail", "-sS", "-o", str(z) + ".part", URLS[archive]])
            if p.returncode:
                return f"FAIL download {archive}"
            os.replace(str(z) + ".part", z)
            print(f"{archive}: {z.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s")
    for s in scenes:
        dest = DATA / s
        if not (dest / "sparse").exists():
            n = extract_scene(DATA / f"{SCENE_ARCHIVE[s]}.zip", s, dest)
            print(f"{s}: extracted {n} files")
        n_img = len(list((dest / "images").glob("*")))
        print(f"{s}: {n_img} images, sparse: {sorted(p.name for p in (dest / 'sparse').rglob('*.bin'))}")
        if n_img == 0:
            return f"FAIL no images for {s}"
    print(f"disk free: {shutil.disk_usage('/content').free / 1e9:.0f} GB")
    return "OK"


def pipeline_pid():
    pid_file = M0 / "pipeline.pid"
    if not pid_file.exists():
        return None
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


def step_parity():
    """M2 checks: build the pre-M2 rasterizer separately, render with it, then compare."""
    old_pkg = ROOT / "old_raster.tar.gz"
    if not old_pkg.exists():
        return "FAIL no /content/old_raster.tar.gz uploaded"
    old_src, old_ext = ROOT / "old_raster", ROOT / "ext_old"
    if not (old_ext / "diff_t_rasterization").exists():
        shutil.rmtree(old_src, ignore_errors=True)
        old_src.mkdir()
        with tarfile.open(old_pkg) as t:
            t.extractall(old_src)
        import torch
        cap = torch.cuda.get_device_capability(0)
        env = dict(os.environ, TORCH_CUDA_ARCH_LIST=f"{cap[0]}.{cap[1]}", MAX_JOBS="16")
        t0 = time.time()
        p = sh([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps",
                "--target", str(old_ext), str(old_src / "submodules" / "diff-t-rasterization")],
               log=M0 / "build_old.log", env=env)
        print(f"old build: {time.time() - t0:.0f}s, exit {p.returncode}")
        if p.returncode:
            return "FAIL old rasterizer build (see /content/m0/build_old.log)"
    env = dict(os.environ, PYTHONPATH=str(old_ext))
    p = sh([sys.executable, "scripts/colab/m2_scene.py", "--out", str(M0 / "m2_old.pt")],
           cwd=SSS, env=env)
    print(p.stdout.strip())
    if p.returncode:
        return "FAIL rendering with the old build"
    p = sh([sys.executable, "scripts/colab/m2_checks.py"], cwd=SSS, log=M0 / "m2_checks.log")
    print("\n".join(l for l in p.stdout.splitlines() if l.startswith(("{", "M2_"))))
    verdict = next((l for l in p.stdout.splitlines() if l.startswith("M2_CHECKS=")), "")
    return "OK" if verdict.startswith("M2_CHECKS=OK") else f"FAIL {verdict or 'checks crashed'}"


def step_m3():
    """M3 checks (scripts/colab/m3_checks.py); needs the train scene's data."""
    p = sh([sys.executable, "scripts/colab/m3_checks.py"], cwd=SSS, log=M0 / "m3_checks.log")
    print("\n".join(l for l in p.stdout.splitlines() if l.startswith(("{", "M3_"))))
    verdict = next((l for l in p.stdout.splitlines() if l.startswith("M3_CHECKS=")), "")
    return "OK" if verdict.startswith("M3_CHECKS=OK") else f"FAIL {verdict or 'checks crashed'}"


def step_m4():
    """M4 checks (scripts/colab/m4_checks.py): Option B in CUDA vs the float64 reference."""
    p = sh([sys.executable, "scripts/colab/m4_checks.py"], cwd=SSS, log=M0 / "m4_checks.log")
    print("\n".join(l for l in p.stdout.splitlines() if l.startswith(("{", "M4_"))))
    verdict = next((l for l in p.stdout.splitlines() if l.startswith("M4_CHECKS=")), "")
    return "OK" if verdict.startswith("M4_CHECKS=OK") else f"FAIL {verdict or 'checks crashed'}"


def step_m5():
    """M5 checks (scripts/colab/m5_checks.py): the exact-ν-gradient flag vs the float64 reference."""
    p = sh([sys.executable, "scripts/colab/m5_checks.py"], cwd=SSS, log=M0 / "m5_checks.log")
    print("\n".join(l for l in p.stdout.splitlines() if l.startswith(("{", "M5_"))))
    verdict = next((l for l in p.stdout.splitlines() if l.startswith("M5_CHECKS=")), "")
    return "OK" if verdict.startswith("M5_CHECKS=OK") else f"FAIL {verdict or 'checks crashed'}"


def step_launch(scenes, runs, dry):
    if pipeline_pid():
        return "FAIL a pipeline is already running"
    for f in ("status.json", "pipeline.log"):
        (M0 / f).unlink(missing_ok=True)
    cmd = [sys.executable, "scripts/colab/m0_pipeline.py"]
    if runs:
        cmd += ["--runs", *runs]
    else:
        cmd += ["--scenes", *scenes]
    if dry:
        cmd.append("--dry")
    log = open(M0 / "pipeline.log", "w")
    p = subprocess.Popen(cmd, cwd=SSS, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (M0 / "pipeline.pid").write_text(str(p.pid))
    time.sleep(5)
    print(f"pipeline pid {p.pid}: {' '.join(cmd)}")
    return "OK" if pipeline_pid() or (M0 / "status.json").exists() else "FAIL pipeline exited at start"


def step_status():
    status_file = M0 / "status.json"
    st = json.loads(status_file.read_text()) if status_file.exists() else {}
    alive = pipeline_pid() is not None
    state = st.get("state", "starting" if alive else "missing")
    if state == "running" and not alive:
        state = "failed"
        st["error"] = "pipeline process is gone without a final status"
    now = time.time()
    elapsed = now - st["t0"] if "t0" in st else 0
    in_phase = now - st["phase_t0"] if st.get("phase_t0") else 0
    print(f"M0_STATE={state}")
    print(f"M0_PROGRESS=scene {st.get('scene')} phase {st.get('phase')} ({in_phase / 60:.1f} min) "
          f"done {st.get('done', [])} total {elapsed / 60:.1f} min")
    if st.get("error"):
        print(f"M0_ERROR={st['error']}")
    tail = st.get("log")
    if tail and Path(tail).exists():
        lines = Path(tail).read_text(errors="replace").replace("\r", "\n").splitlines()
        print("\n".join(l for l in lines[-4:] if l.strip()))
    print(sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"]).stdout.strip())
    return "OK"


def step_collect():
    out = ROOT / "m0_results.tar.gz"
    keep = ("summary.json", "config.json", "results.json", "per_view.json", "fps.json", "cfg_args",
            "cameras.json", "mem.csv", ".log", ".md", "stats.json", "m2_checks.json", "m3_checks.json", "m4_checks.json",
            "m5_checks.json",
            "nan_capture.pt", "nan_capture_scene.pt")
    with tarfile.open(out, "w:gz") as t:
        for p in sorted(M0.rglob("*")):
            if p.is_file() and (p.name.endswith(keep) or p.parent.name == "stats"):
                if p.stat().st_size < 200e6:
                    t.add(p, arcname=str(p.relative_to(M0)))
    print(f"{out}: {out.stat().st_size / 1e6:.1f} MB")
    return "OK"


def main():
    step = os.environ.get("STEP", "")
    runs = os.environ.get("RUNS", "").split()
    scenes = os.environ.get("SCENES", "train garden").split()
    scenes += [r.split(":")[1] for r in runs if ":" in r and r.split(":")[1] not in scenes]
    dry = os.environ.get("DRY", "0") == "1"
    unknown = [s for s in scenes if s not in SCENE_ARCHIVE]
    if unknown:
        print(f"M0_{step.upper()}=FAIL unknown scenes {unknown}")
        return
    try:
        result = {"setup": step_setup, "data": lambda: step_data(scenes),
                  "parity": step_parity, "m3": step_m3, "m4": step_m4, "m5": step_m5, "launch": lambda: step_launch(scenes, runs, dry),
                  "status": step_status,
                  "collect": step_collect}[step]()
    except KeyError:
        result = f"FAIL unknown STEP '{step}'"
    except Exception as e:  # report, never crash silently
        result = f"FAIL {type(e).__name__}: {e}"
    print(f"M0_{step.upper()}={result}")


if __name__ == "__main__":
    main()
