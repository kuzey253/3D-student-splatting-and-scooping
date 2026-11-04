"""Collect the numbers behind results/report/index.html into one JSON (M6 report).

Every value comes from a result file under results/ (paths are recorded next to each block), so the
report reproduces from the repository:

    python3 scripts/m6_report_data.py --out results/report/report_data.json \
        --template results/report/template.html --html results/report/index.html
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def stats_md(path):
    """Parse scripts/m0_component_stats.py's markdown: ν summary, ν histogram, per-scale table, ρ histogram."""
    t = path.read_text()
    out = {"source": str(path.relative_to(ROOT))}
    m = re.search(r"Median ([0-9.e+]+)\. At the lower bound 1: ([0-9.]+)%; at the upper bound 1e4: ([0-9.]+)%"
                  r".*?Negative opacity: ([0-9.]+)%", t, re.S)
    out.update(nu_median=float(m.group(1)), nu_at_1=float(m.group(2)), nu_at_max=float(m.group(3)),
               negative=float(m.group(4)))
    sec = t.split("## Learned ν")[1].split("## Per scale")[0]
    out["nu_hist"] = [[b, int(n)] for b, n in re.findall(r"^\| (\[[^|]+\)) \| (\d+) \|", sec, re.M)]
    sec = t.split("## Per scale")[1].split("## ρ")[0]
    out["per_scale"] = [{"scale": s, "pairs": int(p), "optb_share": float(sh), "cost": float(c)}
                        for s, p, sh, c in re.findall(r"^\| (1/\d) \| (\d+) \| ([0-9.]+)% \| ([0-9.]+)×", sec, re.M)]
    sec = t.split("## ρ")[1]
    out["rho_hist"] = [[b, [int(x) for x in rest.split("|")]]
                       for b, rest in re.findall(r"^\| (\[[^|]+\)) \| ([\d |]+?) \|\s*$", sec, re.M)]
    return out


def summary(path):
    s = json.loads(path.read_text())
    keep = ("run", "scene", "iterations", "components", "psnr", "ssim", "lpips", "fps", "render_size",
            "peak_gpu_mem_mib", "train_s", "render_s", "per_scale", "method", "protocol", "config")
    d = {k: s[k] for k in keep if k in s}
    d["source"] = str(path.relative_to(ROOT))
    return d


def quad():
    rows = list(csv.DictReader(open(R / "m1/quad_study.csv")))
    pick = {"README (fixed K=8, Gaussian > 30)": "README as written (fixed, K=8, Gaussian above ν=30)",
            "Gauss K=8": "gauss K=8 cv=none",
            "adaptive_tail τ=1/255 K=8 cv=A": "adaptive_tail τ=1/255 K=8 cv=A",
            "adaptive_tail τ=1/255 K=12 cv=A (chosen)": "adaptive_tail τ=1/255 K=12 cv=A",
            "adaptive_tail τ=1e-4 K=32 cv=A": "adaptive_tail τ=1e-4 K=32 cv=A"}
    series = {}
    for label, cand in pick.items():
        rs = sorted((r for r in rows if r["candidate"] == cand), key=lambda r: float(r["nu"]))
        series[label] = {"nu": [float(r["nu"]) for r in rs], "max_abs": [float(r["max_abs"]) for r in rs],
                         "core_rel": [float(r["core_rel"]) for r in rs]}
    # every candidate's worst max_abs over ν, with its per-pixel cost class (K)
    worst = {}
    for r in rows:
        worst[r["candidate"]] = max(worst.get(r["candidate"], 0), float(r["max_abs"]))
    scatter = []
    for c, w in worst.items():
        k = re.search(r"K=(\d+)", c)
        cv = "cv=A" in c and "A_prime" not in c
        scatter.append({"candidate": c, "K": int(k.group(1)) if k else None, "worst_max_abs": w,
                        "cv": "A" if cv else ("A′" if "A_prime" in c else "none")})
    return {"source": "results/m1/quad_study.csv", "series": series, "scatter": scatter}


def shortcut():
    rows = list(csv.DictReader(open(R / "m1/shortcut_study.csv")))
    nus = sorted({float(r["nu"]) for r in rows})
    return {"source": "results/m1/shortcut_study.csv",
            "series": {str(nu): {"rho": [float(r["rho"]) for r in rows if float(r["nu"]) == nu],
                                 "A": [float(r["A"]) for r in rows if float(r["nu"]) == nu]} for nu in nus}}


def drift():
    t = (R / "m5/relocation_drift.md").read_text()
    rows = []
    for line in t.splitlines():
        m = re.match(r"^\| ([a-z -]+) \(([^)]+)\) \| (\d+) \| ([0-9.]+) \| (\d) \| ([0-9.]+)(†?) \| (.+) \|$", line)
        if m:
            vals = [float(re.sub(r"[*†]", "", v.strip())) for v in m.group(8).split("|")]
            rows.append({"shape": m.group(1), "nu": int(m.group(3)), "o": float(m.group(4)), "N": int(m.group(5)),
                         "sss": vals[0], "raw_fast": vals[1], "eff_fast": vals[2], "raw_exact": vals[3],
                         "eff_exact": vals[4]})
    return {"source": "results/m5/relocation_drift.md", "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", help="results/report/template.html: also write the report with the data inlined")
    ap.add_argument("--html", help="output page for --template: a standalone HTML file")
    ap.add_argument("--fragment", help="also write the page without the document skeleton (for publishing as an Artifact)")
    a = ap.parse_args()
    data = {"quad": quad(), "shortcut": shortcut(), "drift": drift()}
    data["m0"] = {s: {"summary": summary(R / f"m0/run_20261020_181819/{s}/summary.json"),
                      "stats": stats_md(R / f"m0/run_20261020_181819/{s}/stats/component_stats.md")}
                  for s in ("train", "garden")}
    train_runs = {
        "M2 filter off": "m2/run_20261027_152224/train_s0",
        "M2 2D filter": "m2/run_20261027_152224/train_s01",
        "M3 filter off": "m3/run_20261027_160410/train_s0",
        "M3 3D filter": "m3/run_20261027_160410/train_3d",
        "M3 2D+3D (Option A)": "m3/run_20261027_160410/train_s01_3d",
        "M4 Option A (s1)": "m4/run_20261027_172201/train_s01_3d",
        "M4 Option B + 3D (s1)": "m4/run_20261027_172201/train_s01_3d_ob",
        "M4 Option B + 3D (s8)": "m4/run_20261102_141838/train_s01_3d_ob",
        "M4 Option B, σ3=0 (a)": "m4/run_20261102_141838/ob_a",
        "M4 Option B, σ3=0 (b)": "m4/run_20261102_141838/ob_b",
        "M4 Option B, σ3=0 (c)": "m4/run_20261102_141838/ob_c",
        "M5 Option A + exact ν grad": "m5/run_20261103_134858/train_s01_3d_exnu",
        "M5 Option B + exact ν grad": "m5/run_20261103_134858/train_s01_3d_ob_exnu",
        "M5 Option A + eff. reloc + filt. reg": "m5/run_20261103_134858/train_s01_3d_m5",
        "M5 Option B + eff. reloc + filt. reg": "m5/run_20261103_134858/train_s01_3d_ob_m5",
    }
    data["train7k"] = {}
    for label, d in train_runs.items():
        e = {"summary": summary(R / d / "summary.json")}
        st = R / d / "stats/component_stats.md"
        if st.exists():
            e["stats"] = stats_md(st)
        data["train7k"][label] = e
    pilot = "m6/run_20261103_175727"
    data["pilot"] = {r: {"summary": summary(R / pilot / r / "summary.json")} for r in ("lego_sss", "lego_mip", "lego_a")}
    for r in ("lego_sss", "lego_a"):
        data["pilot"][r]["stats"] = stats_md(R / pilot / r / "stats/component_stats.md")
    data["pilot_dry"] = {}
    for d in ("m6/dry_20261103_155521", "m6/dry_20261103_173138"):
        data["pilot_dry"][d] = {r: summary(R / d / r / "summary.json") for r in ("lego_sss", "lego_mip", "lego_a", "lego_b")}
    lb = (R / pilot / "lego_b/train.log").read_text().replace("\r", "\n")
    it = re.findall(r"\| (\d+)/30000 \[([0-9:]+)<([0-9:]+), *([0-9.]+)it/s", lb)
    data["pilot_lego_b_last"] = {"source": f"results/{pilot}/lego_b/train.log", "iteration": int(it[-1][0]),
                                 "elapsed": it[-1][1], "remaining_est": it[-1][2], "it_per_s": float(it[-1][3])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}: {len(json.dumps(data)) / 1e3:.0f} kB")
    if a.template:
        blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        page = Path(a.template).read_text()
        assert page.count("__REPORT_DATA__") == 1
        body = page.replace("__REPORT_DATA__", blob)
        head = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
        Path(a.html).write_text(head + body + "\n</html>\n")
        print(f"wrote {a.html}")
        if a.fragment:
            Path(a.fragment).parent.mkdir(parents=True, exist_ok=True)
            Path(a.fragment).write_text(body)
            print(f"wrote {a.fragment}")


if __name__ == "__main__":
    main()
