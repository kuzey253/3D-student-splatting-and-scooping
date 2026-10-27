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


def parse_run(spec):
    """'name:scene[:key=value,key=value]' → (name, scene, overrides). Values are JSON when they parse."""
    parts = spec.split(":", 2)
    name, scene = parts[0], parts[1] if len(parts) > 1 else parts[0]
    overrides = {}
    for kv in (parts[2].split(",") if len(parts) > 2 and parts[2] else []):
        k, v = kv.split("=", 1)
        try:
            overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            overrides[k] = v
    return name, scene, overrides


def run_scene(scene, dry, status, name=None, overrides=None):
    name = name or scene
    d = M0 / name
    model = d / "model"
    d.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(f"configs/{scene}.json").read_text())
    cfg.update(overrides or {})
    if dry:
        cfg["iterations"] = min(cfg["iterations"], DRY_ITERATIONS)
        cfg["cap_max"] = min(cfg["cap_max"], DRY_CAP)
    # render.py and the FPS script would otherwise take the filter from their own default (0)
    pipe_flags = ["--mip_kernel_size", str(cfg["mip_kernel_size"])] if "mip_kernel_size" in cfg else []
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    iters = cfg["iterations"]
    times = {}

    mem = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                            "-lms", "500"], stdout=open(d / "mem.csv", "w"))
    try:
        times["train_s"] = run([sys.executable, "train.py", "-s", str(DATA / scene), "-m", str(model),
                                "--config", str(d / "config.json")], d / "train.log", status, "train",
                               name)
    finally:
        mem.terminate()
    ply = model / "point_cloud" / f"iteration_{iters}" / "point_cloud.ply"
    if not ply.exists():
        raise PhaseError(f"{scene}: no {ply}")
    vals = [int(x) for x in (d / "mem.csv").read_text().split() if x.strip().isdigit()]

    times["render_s"] = run([sys.executable, "render.py", "-m", str(model), "--skip_train", *pipe_flags],
                            d / "render.log", status, "render", name)
    renders = model / "test" / f"ours_{iters}" / "renders"
    n_renders = len(list(renders.glob("*.png")))
    if n_renders == 0 or n_renders != len(list((renders.parent / "gt").glob("*.png"))):
        raise PhaseError(f"{scene}: {n_renders} renders in {renders}")

    times["metrics_s"] = run([sys.executable, "metrics.py", "-m", str(model)], d / "metrics.log",
                             status, "metrics", name)
    results = model / "results.json"
    if not results.exists():
        raise PhaseError(f"{scene}: metrics.py wrote no results.json (see metrics.log)")
    scores = json.loads(results.read_text())[f"ours_{iters}"]
    for name in ("results.json", "per_view.json", "cfg_args", "cameras.json"):
        if (model / name).exists():
            (d / name).write_bytes((model / name).read_bytes())

    times["fps_s"] = run([sys.executable, "scripts/colab/m0_fps.py", "-m", str(model),
                          "--out", str(d / "fps.json"), *pipe_flags], d / "fps.log", status, "fps", name)
    fps = json.loads((d / "fps.json").read_text())

    times["stats_s"] = run([sys.executable, "scripts/m0_component_stats.py", str(model),
                            "--out", str(d / "stats")], d / "stats.log", status, "stats", name)

    summary = {"run": name, "scene": scene, "dry": dry, "iterations": iters, "config": cfg,
               "components": ply_vertices(ply), "psnr": scores["PSNR"], "ssim": scores["SSIM"],
               "lpips": scores["LPIPS"], "fps": fps["fps"], "render_size": fps["size"],
               "test_views": n_renders, "peak_gpu_mem_mib": max(vals) if vals else None,
               **{k: round(v, 1) for k, v in times.items()}}
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=[])
    ap.add_argument("--runs", nargs="+", default=[],
                    help="name:scene[:key=value,...] with config overrides, e.g. "
                         "train_mip:train:iterations=7000,mip_kernel_size=0.1")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    M0.mkdir(parents=True, exist_ok=True)
    status = Status()
    try:
        for name, scene, overrides in [(s, s, {}) for s in args.scenes] + [parse_run(r) for r in args.runs]:
            summary = run_scene(scene, args.dry, status, name, overrides)
            print(json.dumps(summary), flush=True)
            status.update(done=status.st["done"] + [name])
        status.update(state="done", phase=None)
    except Exception as e:
        status.update(state="failed", error=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
