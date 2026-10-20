"""M0 pipeline, run detached on the GPU VM from the SSS root by scripts/colab/remote_m0.py.

For each scene: train SSS with its shipped config, render and score the test views, time
rendering, and compute component statistics. Every phase checks its output files rather than
trusting exit codes (metrics.py, for one, swallows its own errors). Progress goes to
/content/m0/status.json; each scene ends with /content/m0/<scene>/summary.json.

--dry trains for 1000 iterations with the component cap lowered to 100k (still past SSS's first
relocations at iteration 600), to check the whole chain cheaply.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

M0 = Path("/content/m0")
DATA = Path("/content/data")
DRY_ITERATIONS, DRY_CAP = 1000, 100_000


class PhaseError(RuntimeError):
    pass


class Status:
    def __init__(self):
        self.t0 = time.time()
        self.st = {"state": "running", "done": [], "error": None, "t0": self.t0}

    def update(self, **kw):
        if "phase" in kw:
            kw["phase_t0"] = time.time()
        self.st.update(kw, elapsed_s=time.time() - self.t0)
        tmp = M0 / "status.json.tmp"
        tmp.write_text(json.dumps(self.st, indent=1))
        tmp.replace(M0 / "status.json")


def run(cmd, log, status, phase, scene):
    status.update(scene=scene, phase=phase, log=str(log))
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise PhaseError(f"{scene}/{phase} exited with {rc} (see {log})")
    return time.time() - t0


def ply_vertices(path):
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
            if line.startswith(b"end_header"):
                break
    return None


def run_scene(scene, dry, status):
    d = M0 / scene
    model = d / "model"
    d.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(f"configs/{scene}.json").read_text())
    if dry:
        cfg["iterations"] = DRY_ITERATIONS
        cfg["cap_max"] = min(cfg["cap_max"], DRY_CAP)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    iters = cfg["iterations"]
    times = {}

    mem = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                            "-lms", "500"], stdout=open(d / "mem.csv", "w"))
    try:
        times["train_s"] = run([sys.executable, "train.py", "-s", str(DATA / scene), "-m", str(model),
                                "--config", str(d / "config.json")], d / "train.log", status, "train",
                               scene)
    finally:
        mem.terminate()
    ply = model / "point_cloud" / f"iteration_{iters}" / "point_cloud.ply"
    if not ply.exists():
        raise PhaseError(f"{scene}: no {ply}")
    vals = [int(x) for x in (d / "mem.csv").read_text().split() if x.strip().isdigit()]

    times["render_s"] = run([sys.executable, "render.py", "-m", str(model), "--skip_train"],
                            d / "render.log", status, "render", scene)
    renders = model / "test" / f"ours_{iters}" / "renders"
    n_renders = len(list(renders.glob("*.png")))
    if n_renders == 0 or n_renders != len(list((renders.parent / "gt").glob("*.png"))):
        raise PhaseError(f"{scene}: {n_renders} renders in {renders}")

    times["metrics_s"] = run([sys.executable, "metrics.py", "-m", str(model)], d / "metrics.log",
                             status, "metrics", scene)
    results = model / "results.json"
    if not results.exists():
        raise PhaseError(f"{scene}: metrics.py wrote no results.json (see metrics.log)")
    scores = json.loads(results.read_text())[f"ours_{iters}"]
    for name in ("results.json", "per_view.json", "cfg_args", "cameras.json"):
        if (model / name).exists():
            (d / name).write_bytes((model / name).read_bytes())

    times["fps_s"] = run([sys.executable, "scripts/colab/m0_fps.py", "-m", str(model),
                          "--out", str(d / "fps.json")], d / "fps.log", status, "fps", scene)
    fps = json.loads((d / "fps.json").read_text())

    times["stats_s"] = run([sys.executable, "scripts/m0_component_stats.py", str(model),
                            "--out", str(d / "stats")], d / "stats.log", status, "stats", scene)

    summary = {"scene": scene, "dry": dry, "iterations": iters, "config": cfg,
               "components": ply_vertices(ply), "psnr": scores["PSNR"], "ssim": scores["SSIM"],
               "lpips": scores["LPIPS"], "fps": fps["fps"], "render_size": fps["size"],
               "test_views": n_renders, "peak_gpu_mem_mib": max(vals) if vals else None,
               **{k: round(v, 1) for k, v in times.items()}}
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    M0.mkdir(parents=True, exist_ok=True)
    status = Status()
    try:
        for scene in args.scenes:
            summary = run_scene(scene, args.dry, status)
            print(json.dumps(summary), flush=True)
            status.update(done=status.st["done"] + [scene])
        status.update(state="done", phase=None)
    except Exception as e:
        status.update(state="failed", error=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
