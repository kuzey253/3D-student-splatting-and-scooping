#!/usr/bin/env bash
# M0 on a Colab GPU (COLAB_GPU_RUNBOOK.md, PLAN.md): package this working tree, set up the VM,
# train the SSS baselines, and bank the small results locally under results/m0/<tag>/.
#
# Start the session yourself, only after the user's go-ahead:  colab new --gpu G4 --session g4
# Then, dry run first:   SESSION=g4 DRY=1 bash scripts/colab/run_m0.sh
#            real run:   SESSION=g4 bash scripts/colab/run_m0.sh
# This runner never starts or stops the VM. Afterwards: colab stop -s g4, and kill keep-alive helpers.
#
# Env: SESSION (required), DRY (0/1), SCENES (default "train garden"), RUNS (named runs with config
# overrides, "name:scene:key=value,..." space-separated; replaces SCENES for training), PARITY=1
# (M2 checks: also build the rasterizer at OLD_REF, default 19b1b53, and compare), M3CHECKS=1 (M3
# checks, after the data step; stops before training if they fail), M4CHECKS=1 (M4 Option B checks,
# likewise), M5CHECKS=1 (M5 exact-ν-gradient flag checks, likewise), MIP=1 (M6: fetch and build
# Mip-Splatting dda02ab for method=mip runs, after the data step), ATTACH (the user's
# attach_session.py, used to re-attach a lost session), ATTACH_ARGS (its runtime filter; default
# --require-large, i.e. >= 80 GB VRAM such as the G4; empty for a 40 GB A100), POLL (seconds
# between status checks; default 30 dry, 300 real). Remote steps answer with M0_<STEP>=OK|FAIL
# lines, which this script checks; `colab exec` itself exits 0 even when the remote code fails.
set -euo pipefail

SESSION=${SESSION:?set SESSION, e.g. SESSION=g4}
DRY=${DRY:-0}
SCENES=${SCENES:-train garden}
RUNS=${RUNS:-}
PARITY=${PARITY:-0}
M3CHECKS=${M3CHECKS:-0}
M4CHECKS=${M4CHECKS:-0}
M5CHECKS=${M5CHECKS:-0}
MIP=${MIP:-0}
ATTACH=${ATTACH:-$HOME/Desktop/Diffusion_Experiment/scripts/attach_session.py}
ATTACH_ARGS=${ATTACH_ARGS---require-large}
OLD_REF=${OLD_REF:-19b1b53}
POLL=${POLL:-$([ "$DRY" = 1 ] && echo 30 || echo 300)}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TAG=$([ "$DRY" = 1 ] && echo dry || echo run)_$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-$ROOT/results/m0/$TAG}
RETRY=${RETRY:-60}
REMOTE="$ROOT/scripts/colab/remote_m0.py"
mkdir -p "$OUT"

log() { echo "[$(date +%H:%M:%S)] $*"; }
quiet() { grep -v -E "new version of Colab CLI|colab update --install|uv tool install|enable_update_check" || true; }

try_step() {  # try_step <name> <timeout_s>: run one remote step; return 0 iff it answered OK
  local name=$1 timeout=$2 out="$OUT/$1.log"
  colab exec -s "$SESSION" --timeout "$timeout" --env "STEP=$name" --env "DRY=$DRY" \
    --env "SCENES=$SCENES" --env "RUNS=$RUNS" -f "$REMOTE" 2>&1 | quiet > "$out"
  local upper
  upper=$(echo "$name" | tr '[:lower:]' '[:upper:]')
  grep -q "^M0_${upper}=OK" "$out"
}

step() {  # step <name> <timeout_s>: as try_step, but stop the runner if the step fails
  if ! try_step "$1" "$2"; then
    log "step $1 did not answer OK:"
    grep "^M0_$(echo "$1" | tr '[:lower:]' '[:upper:]')=" "$OUT/$1.log" || tail -20 "$OUT/$1.log"
    exit 1
  fi
}

log "output: $OUT"
log "warming the connection"
for _ in 1 2; do colab exec -s "$SESSION" --timeout 120 <<< 'print("warm")' 2>&1 | quiet > /dev/null || true; done

log "packaging the working tree"
PKG_DIR=$(mktemp -d)
{ echo "head $(git -C "$ROOT" rev-parse HEAD)"; echo "status:"; git -C "$ROOT" status --porcelain; } \
  > "$PKG_DIR/GIT_STATE.txt"
cp "$PKG_DIR/GIT_STATE.txt" "$OUT/"
(cd "$ROOT" && { git ls-files; git ls-files --others --exclude-standard; } \
  | grep -v -E '^(COLAB_GPU_RUNBOOK\.md|results/)' | sort -u > "$PKG_DIR/files.txt")
# COPYFILE_DISABLE: no macOS ._ resource files in the archive
(cd "$ROOT" && COPYFILE_DISABLE=1 tar cf "$PKG_DIR/sss.tar" -T "$PKG_DIR/files.txt")
COPYFILE_DISABLE=1 tar rf "$PKG_DIR/sss.tar" -C "$PKG_DIR" GIT_STATE.txt
gzip "$PKG_DIR/sss.tar"
colab upload -s "$SESSION" "$PKG_DIR/sss.tar.gz" /content/sss.tar.gz 2>&1 | quiet
if [ "$PARITY" = 1 ]; then
  git -C "$ROOT" archive --format=tar.gz -o "$PKG_DIR/old_raster.tar.gz" "$OLD_REF" submodules/diff-t-rasterization
  colab upload -s "$SESSION" "$PKG_DIR/old_raster.tar.gz" /content/old_raster.tar.gz 2>&1 | quiet
fi

log "setup: unpack, record versions, build the CUDA extensions"
step setup 3600
grep -E '"(gpu|torch|torch_cuda|nvcc|capability)"|^build:' "$OUT/setup.log" || true
if [ "$PARITY" = 1 ]; then
  log "parity: build the rasterizer at $OLD_REF, CUDA vs float64 reference, filter-off unchanged"
  if ! try_step parity 3600; then
    log "parity checks did not pass; stopping before training:"
    grep -E "^M0_PARITY=|^M2_" "$OUT/parity.log" || tail -20 "$OUT/parity.log"
    exit 1
  else
    grep -E "^M2_" "$OUT/parity.log" || true
  fi
fi
log "data: download and extract $SCENES ${RUNS:+(runs: $RUNS)}"
step data 3600
grep -E "images|GB" "$OUT/data.log" || true
if [ "$M3CHECKS" = 1 ]; then
  log "m3: 3D filter wiring vs float64 reference, filter values on real cameras"
  if ! try_step m3 1800; then
    log "M3 checks did not pass; stopping before training:"
    grep -E "^M0_M3=|^M3_" "$OUT/m3.log" || tail -20 "$OUT/m3.log"
    exit 1
  fi
  grep -E "^M3_" "$OUT/m3.log" || true
fi
if [ "$M4CHECKS" = 1 ]; then
  log "m4: Option B in CUDA vs float64 reference, render() wiring"
  if ! try_step m4 1800; then
    log "M4 checks did not pass; stopping before training:"
    grep -E "^M0_M4=|^M4_" "$OUT/m4.log" || tail -20 "$OUT/m4.log"
    exit 1
  fi
  grep -E "^M4_" "$OUT/m4.log" || true
fi
if [ "$M5CHECKS" = 1 ]; then
  log "m5: exact-ν-gradient flag vs float64 reference"
  if ! try_step m5 1200; then
    log "M5 checks did not pass; stopping before training:"
    grep -E "^M0_M5=|^M5_" "$OUT/m5.log" || tail -20 "$OUT/m5.log"
    exit 1
  fi
  grep -E "^M5_" "$OUT/m5.log" || true
fi
if [ "$MIP" = 1 ]; then
  log "mip: fetch Mip-Splatting dda02ab, patch the build, build its rasterizer"
  step mip 3600
  grep -E "^mip" "$OUT/mip.log" || true
fi
log "launch"
step launch 120

fetch_results() {  # collect on the VM, download, extract; return 0 iff all three worked
  try_step collect 600 || return 1
  colab download -s "$SESSION" /content/m0_results.tar.gz "$OUT/m0_results.tar.gz" 2>&1 | quiet
  [ -s "$OUT/m0_results.tar.gz" ] && tar xzf "$OUT/m0_results.tar.gz" -C "$OUT"
}

reattach() {  # the CLI sometimes loses the session while the VM runs on (NOTES.md 58); re-attach it
  local out="$OUT/reattach_$(date +%H%M%S).log"
  (cd "$(dirname "$ATTACH")/.." && python3 "$ATTACH" --name "$SESSION" $ATTACH_ARGS --keep-alive) 2>&1 \
    | quiet > "$out" || true
  grep -E "Attached|keep-alive daemon pid|no active runtimes" "$out" | sed 's/^/    /' || true
  if grep -q "no active runtimes" "$out"; then
    log "the VM is gone; runs collected so far are in $OUT"
    exit 1
  fi
}

# Training runs detached on the VM; each status check is a separate short exec, so no command is
# ever sent into a streaming job. A failed check re-attaches the session (keep-alive helpers it
# starts must be killed afterwards) and retries after a minute; the pipeline keeps running on the
# VM regardless. Each finished run is collected as soon as a status check reports it, so a lost VM
# does not take finished runs with it.
state=running
misses=0
collected=0
while [ "$state" = running ] || [ "$state" = starting ]; do
  sleep "$([ "$misses" -gt 0 ] && echo "$RETRY" || echo "$POLL")"
  if ! try_step status 120; then
    misses=$((misses + 1))
    log "status check failed ($misses in a row); re-attaching"
    reattach
    if [ "$misses" -ge 6 ]; then
      log "giving up on status checks; the pipeline may still be running on the VM"
      exit 1
    fi
    continue
  fi
  misses=0
  state=$(grep -o '^M0_STATE=[a-z]*' "$OUT/status.log" | cut -d= -f2)
  log "$state $(grep '^M0_PROGRESS=' "$OUT/status.log" | cut -d= -f2-)"
  done_n=$(grep -o '^M0_DONE=[0-9]*' "$OUT/status.log" | cut -d= -f2)
  if [ "$state" = running ] && [ "${done_n:-0}" -gt "$collected" ]; then
    if fetch_results; then
      collected=$done_n
      log "collected $collected finished run(s)"
    else
      log "collecting finished runs failed; will retry"
    fi
  fi
done
grep '^M0_ERROR=' "$OUT/status.log" || true

log "collect"
fetch_results || { log "final collect failed; runs collected so far are in $OUT"; exit 1; }
for s in "$OUT"/*/summary.json; do [ -f "$s" ] && cat "$s"; done

if [ "$state" != done ]; then
  log "pipeline ended in state '$state'; results so far are in $OUT"
  exit 1
fi
log "done. Results in $OUT. Stop the VM when finished: colab stop -s $SESSION"
