#!/usr/bin/env bash
# M0 on a Colab GPU (COLAB_GPU_RUNBOOK.md, PLAN.md): package this working tree, set up the VM,
# train the SSS baselines, and bank the small results locally under results/m0/<tag>/.
#
# Start the session yourself, only after the user's go-ahead:  colab new --gpu G4 --session g4
# Then, dry run first:   SESSION=g4 DRY=1 bash scripts/colab/run_m0.sh
#            real run:   SESSION=g4 bash scripts/colab/run_m0.sh
# This runner never starts or stops the VM. Afterwards: colab stop -s g4, and kill keep-alive helpers.
#
# Env: SESSION (required), DRY (0/1), SCENES (default "train garden"), POLL (seconds between status
# checks; default 30 dry, 300 real). Remote steps answer with M0_<STEP>=OK|FAIL lines, which this
# script checks; `colab exec` itself exits 0 even when the remote code fails.
set -euo pipefail

SESSION=${SESSION:?set SESSION, e.g. SESSION=g4}
DRY=${DRY:-0}
SCENES=${SCENES:-train garden}
POLL=${POLL:-$([ "$DRY" = 1 ] && echo 30 || echo 300)}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TAG=$([ "$DRY" = 1 ] && echo dry || echo run)_$(date +%Y%m%d_%H%M%S)
OUT="$ROOT/results/m0/$TAG"
REMOTE="$ROOT/scripts/colab/remote_m0.py"
mkdir -p "$OUT"

log() { echo "[$(date +%H:%M:%S)] $*"; }
quiet() { grep -v -E "new version of Colab CLI|colab update --install|uv tool install|enable_update_check" || true; }

try_step() {  # try_step <name> <timeout_s>: run one remote step; return 0 iff it answered OK
  local name=$1 timeout=$2 out="$OUT/$1.log"
  colab exec -s "$SESSION" --timeout "$timeout" --env "STEP=$name" --env "DRY=$DRY" \
    --env "SCENES=$SCENES" -f "$REMOTE" 2>&1 | quiet > "$out"
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

log "setup: unpack, record versions, build the CUDA extensions"
step setup 3600
grep -E '"(gpu|torch|torch_cuda|nvcc|capability)"|^build:' "$OUT/setup.log" || true
log "data: download and extract $SCENES"
step data 3600
grep -E "images|GB" "$OUT/data.log" || true
log "launch"
step launch 120

# Training runs detached on the VM; each status check is a separate short exec, so no command is
# ever sent into a streaming job. A failed check (dropped connection) is retried; the pipeline
# keeps running on the VM regardless.
state=running
misses=0
while [ "$state" = running ] || [ "$state" = starting ]; do
  sleep "$POLL"
  if ! try_step status 120; then
    misses=$((misses + 1))
    log "status check failed ($misses in a row); retrying"
    if [ "$misses" -ge 6 ]; then
      log "giving up on status checks; the pipeline may still be running on the VM"
      exit 1
    fi
    continue
  fi
  misses=0
  state=$(grep -o '^M0_STATE=[a-z]*' "$OUT/status.log" | cut -d= -f2)
  log "$state $(grep '^M0_PROGRESS=' "$OUT/status.log" | cut -d= -f2-)"
done
grep '^M0_ERROR=' "$OUT/status.log" || true

log "collect"
step collect 600
colab download -s "$SESSION" /content/m0_results.tar.gz "$OUT/m0_results.tar.gz" 2>&1 | quiet
tar xzf "$OUT/m0_results.tar.gz" -C "$OUT"
for s in "$OUT"/*/summary.json; do [ -f "$s" ] && cat "$s"; done

if [ "$state" != done ]; then
  log "pipeline ended in state '$state'; results so far are in $OUT"
  exit 1
fi
log "done. Results in $OUT. Stop the VM when finished: colab stop -s $SESSION"
