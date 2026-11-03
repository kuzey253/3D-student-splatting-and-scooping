"""M0 pipeline, run detached on the GPU VM from the SSS root by scripts/colab/remote_m0.py.

For each scene: train SSS with its shipped config, render and score the test views, time
rendering, and compute component statistics. Every phase checks its output files rather than
trusting exit codes (metrics.py, for one, swallows its own errors). Progress goes to
/content/m0/status.json; each scene ends with /content/m0/<scene>/summary.json.

--dry trains for 1000 iterations with the component cap lowered to 100k (still past SSS's first
relocations at iteration 600), to check the whole chain cheaply.

M6 (run overrides, not passed to train.py):
  - protocol=stmt: multi-scale Blender, trained on the full-resolution images, tested at all four
    scales (Mip-Splatting's single-scale-train/multi-scale-test).
  - protocol=zoomin: trained at -r 8 (config resolution 8), rendered at -r 8, 4, 2, 1.
  - method=mip: the Mip-Splatting baseline (/content/mip-splatting, built by remote_m0.py STEP=mip),
    run with its own scripts' arguments; scored by the same per-scale metrics.
Both protocols write per_scale.json (scripts/m6_metrics.py).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

M0 = Path("/content/m0")
DATA = Path("/content/data")
MIP = Path("/content/mip-splatting")
DRY_ITERATIONS, DRY_CAP = 1000, 100_000
ZOOM_SCALES = (8, 4, 2, 1)
MIP_ITERATIONS = 30_000


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


def scene_config(scene):
    path = Path(f"configs/{scene}.json")
    if not path.exists() and (DATA / scene / "metadata.json").exists():
        path = Path("configs/blender.json")  # M6: one config for the multi-scale Blender scenes
    return json.loads(path.read_text())


def run_scene(scene, dry, status, name=None, overrides=None):
    name = name or scene
    d = M0 / name
    model = d / "model"
    d.mkdir(parents=True, exist_ok=True)
    overrides = dict(overrides or {})
    protocol, method = overrides.pop("protocol", None), overrides.pop("method", "sss")
    if method == "mip":
        return run_mip(scene, dry, status, name, protocol, overrides)
    cfg = scene_config(scene)
    cfg.update(overrides)
    if protocol == "zoomin":
        cfg["resolution"] = 8
    if dry:
        cfg["iterations"] = min(cfg["iterations"], DRY_ITERATIONS)
        cfg["cap_max"] = min(cfg["cap_max"], DRY_CAP)
    # render.py and the FPS script would otherwise take the filter from their own default (0)
    pipe_flags = ["--mip_kernel_size", str(cfg["mip_kernel_size"])] if "mip_kernel_size" in cfg else []
    if cfg.get("mip_option_b"):
        pipe_flags += ["--mip_option_b"]
    if cfg.get("mip_filter_3d"):
        pipe_flags += ["--mip_filter_3d", "--mip_filter_3d_variance",
                       str(cfg.get("mip_filter_3d_variance", 0.2))]
    # M4 diagnostic: the rasterizer saves the state at the first NaN gradient into the run directory
    if cfg.pop("nan_capture", False):
        os.environ["MIP_NAN_CAPTURE"] = str(d)
    else:
        os.environ.pop("MIP_NAN_CAPTURE", None)
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

    out = model / "test" / f"ours_{iters}"
    if protocol is None:
        times["render_s"] = run([sys.executable, "render.py", "-m", str(model), "--skip_train", *pipe_flags],
                                d / "render.log", status, "render", name)
        renders = out / "renders"
        n_renders = len(list(renders.glob("*.png")))
        if n_renders == 0 or n_renders != len(list((renders.parent / "gt").glob("*.png"))):
            raise PhaseError(f"{scene}: {n_renders} renders in {renders}")

        times["metrics_s"] = run([sys.executable, "metrics.py", "-m", str(model)], d / "metrics.log",
                                 status, "metrics", name)
        results = model / "results.json"
        if not results.exists():
            raise PhaseError(f"{scene}: metrics.py wrote no results.json (see metrics.log)")
        scores = json.loads(results.read_text())[f"ours_{iters}"]
    else:
        pairs, t_render = [], 0.0
        for r in (ZOOM_SCALES if protocol == "zoomin" else (None,)):
            res = ["-r", str(r), "--data_device", "cpu"] if r else []
            t_render += run([sys.executable, "render.py", "-m", str(model), "--skip_train", *res, *pipe_flags],
                            d / f"render{'_r' + str(r) if r else ''}.log", status, f"render{r or ''}", name)
            if r:  # render.py always writes renders/ and gt/; keep each scale apart
                for sub in ("renders", "gt"):
                    (out / sub).rename(out / f"{sub}_r{r}")
                pairs.append((str(r), out / f"renders_r{r}", out / f"gt_r{r}"))
            else:
                pairs.append(("all", out / "renders", out / "gt"))
        times["render_s"] = t_render
        scores, n_renders = per_scale_metrics(scene, protocol, pairs, d, status, name, times)
    for fname in ("results.json", "per_view.json", "cfg_args", "cameras.json"):
        if (model / fname).exists():
            (d / fname).write_bytes((model / fname).read_bytes())

    only = ["--only_suffix", "_d0"] if protocol == "stmt" else []
    times["fps_s"] = run([sys.executable, "scripts/colab/m0_fps.py", "-m", str(model),
                          "--out", str(d / "fps.json"), *pipe_flags, *only], d / "fps.log", status, "fps", name)
    fps = json.loads((d / "fps.json").read_text())

    times["stats_s"] = run([sys.executable, "scripts/m0_component_stats.py", str(model),
                            "--out", str(d / "stats")], d / "stats.log", status, "stats", name)

    summary = {"run": name, "scene": scene, "dry": dry, "iterations": iters, "config": cfg,
               "components": ply_vertices(ply), "psnr": scores["PSNR"], "ssim": scores["SSIM"],
               "lpips": scores["LPIPS"], "fps": fps["fps"], "render_size": fps["size"],
               "test_views": n_renders, "peak_gpu_mem_mib": max(vals) if vals else None,
               **{k: round(v, 1) for k, v in times.items()}}
    if protocol:
        summary.update(method="sss", protocol=protocol,
                       per_scale=json.loads((d / "per_scale.json").read_text())["scales"])
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def per_scale_metrics(scene, protocol, pairs, d, status, name, times):
    """scripts/m6_metrics.py on (label, renders, gt) pairs; returns (mean over all images, image count)."""
    for _, rdir, gdir in pairs:
        n = len(list(Path(rdir).glob("*.png")))
        if n == 0 or n != len(list(Path(gdir).glob("*.png"))):
            raise PhaseError(f"{scene}: {n} renders in {rdir}")
    cmd = [sys.executable, "scripts/m6_metrics.py", "--out", str(d / "per_scale.json")]
    if protocol == "stmt":
        cmd += ["--metadata", str(DATA / scene / "metadata.json")]
    for label, rdir, gdir in pairs:
        cmd += ["--pair", f"{label}={rdir}:{gdir}"]
    times["metrics_s"] = run(cmd, d / "metrics.log", status, "metrics", name)
    if "M6_METRICS=OK" not in (d / "metrics.log").read_text():
        raise PhaseError(f"{scene}: per-scale metrics failed (see metrics.log)")
    res = json.loads((d / "per_scale.json").read_text())
    return res["all"], res["all"]["n"]


def run_mip(scene, dry, status, name, protocol, overrides):
    """Mip-Splatting dda02ab, with the arguments of its scripts/run_nerf_synthetic_stmt.py and
    run_mipnerf360_stmt.py (kernel size 0.1, 30k iterations), scored by scripts/m6_metrics.py."""
    if protocol not in ("stmt", "zoomin"):
        raise PhaseError(f"{name}: method=mip needs protocol=stmt or zoomin")
    d = M0 / name
    model = d / "model"
    iters = DRY_ITERATIONS if dry else int(overrides.pop("iterations", MIP_ITERATIONS))
    ks = str(overrides.pop("kernel_size", 0.1))
    if overrides:
        raise PhaseError(f"{name}: unsupported Mip-Splatting overrides {sorted(overrides)}")
    train_args = ["-s", str(DATA / scene), "-m", str(model), "--eval", "--kernel_size", ks,
                  "--iterations", str(iters), "--save_iterations", str(iters), "--test_iterations", str(iters)]
    train_args += ["--white_background"] if protocol == "stmt" else ["-r", "8"]
    cfg = {"method": "mip", "protocol": protocol, "ref": "dda02ab", "train_args": train_args}
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    times = {}
    mem = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                            "-lms", "500"], stdout=open(d / "mem.csv", "w"))
    try:
        times["train_s"] = run_in(MIP, [sys.executable, "train.py", *train_args], d / "train.log", status, "train", name)
    finally:
        mem.terminate()
    ply = model / "point_cloud" / f"iteration_{iters}" / "point_cloud.ply"
    if not ply.exists():
        raise PhaseError(f"{name}: no {ply}")
    vals = [int(x) for x in (d / "mem.csv").read_text().split() if x.strip().isdigit()]

    out = model / "test" / f"ours_{iters}"
    pairs, t_render = [], 0.0
    for r in (ZOOM_SCALES if protocol == "zoomin" else (-1,)):
        res = ["-r", str(r), "--data_device", "cpu"] if r > 0 else []
        t_render += run_in(MIP, [sys.executable, "render.py", "-m", str(model), "--skip_train", *res],
                           d / f"render{'_r' + str(r) if r > 0 else ''}.log", status, f"render{r if r > 0 else ''}", name)
        pairs.append((str(r) if r > 0 else "all", out / f"test_preds_{r}", out / f"gt_{r}"))
    times["render_s"] = t_render
    scores, n_renders = per_scale_metrics(scene, protocol, pairs, d, status, name, times)
    for fname in ("cfg_args", "cameras.json"):
        if (model / fname).exists():
            (d / fname).write_bytes((model / fname).read_bytes())

    only = ["--only_suffix", "_d0"] if protocol == "stmt" else []
    times["fps_s"] = run_in(MIP, [sys.executable, str(Path.cwd() / "scripts/colab/m6_mip_fps.py"), "-m", str(model),
                                  "--out", str(d / "fps.json"), *only], d / "fps.log", status, "fps", name)
    fps = json.loads((d / "fps.json").read_text())
    summary = {"run": name, "scene": scene, "dry": dry, "iterations": iters, "config": cfg, "method": "mip",
               "protocol": protocol, "components": ply_vertices(ply), "psnr": scores["PSNR"],
               "ssim": scores["SSIM"], "lpips": scores["LPIPS"], "fps": fps["fps"], "render_size": fps["size"],
               "test_views": n_renders, "peak_gpu_mem_mib": max(vals) if vals else None,
               "per_scale": json.loads((d / "per_scale.json").read_text())["scales"],
               **{k: round(v, 1) for k, v in times.items()}}
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def run_in(cwd, cmd, log, status, phase, scene):
    """run() with another working directory (the Mip-Splatting checkout)."""
    here = Path.cwd()
    os.chdir(cwd)
    try:
        return run(cmd, log, status, phase, scene)
    finally:
        os.chdir(here)


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
