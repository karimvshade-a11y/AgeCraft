#!/usr/bin/env bash
# Unattended, budget-boxed, RESUMABLE training on a rented GPU.
#
# WHAT WENT WRONG LAST TIME
# -------------------------
# 4.5 hours of generation survived only because the raw files happened to still
# be sitting on a network volume. The upload ran once, at the very end, with a
# read-only token -- so it failed, and the pod terminated on schedule with
# everything still inside it. A single upload at the end is not a backup, it is
# a bet, and the odds are worse than they look because the end of the run is
# exactly when you are least able to intervene.
#
# So: the checkpoint leaves this machine every CKPT_EVERY_MIN minutes, starting
# from the first one that exists. Kill the pod at any point after that and you
# lose minutes, not hours.
#
# And because the run RESUMES from whatever checkpoint is on HuggingFace,
# "the budget ran out" and "training is paused" become the same thing. Top up,
# run this again, it carries on from where it stopped. That is what makes a
# small balance workable: you are buying hours, not buying a finished model.
#
# COST
#   RTX 4090 community cloud ~= $0.34/hr. MAX_HOURS is your maximum loss.
#   MAX_HOURS=7 is about $2.40 of GPU plus a little disk.
#
# USAGE
#   export HF_TOKEN=hf_...        # WRITE token. A read token is what broke it before.
#   export MAX_HOURS=7            # your maximum loss, in hours
#   cd /workspace/agecraft
#   nohup bash scripts/run_training.sh > /workspace/train.log 2>&1 &
#   # then close the browser. Losing your connection costs nothing.

set -uo pipefail

MAX_HOURS="${MAX_HOURS:-7}"
RESERVE_MIN="${RESERVE_MIN:-25}"        # held back for ONNX export + final upload
CKPT_EVERY_MIN="${CKPT_EVERY_MIN:-30}"  # how much work you are willing to lose
CONFIG="${CONFIG:-configs/runpod4090.yaml}"
REPO_DIR="${REPO_DIR:-/workspace/agecraft}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
LOG_FILE="${LOG_FILE:-/workspace/train.log}"

export HF_DATASET="${HF_DATASET:-Abdelkarim40/agecraft-data}"
export HF_MODEL="${HF_MODEL:-Abdelkarim40/agecraft-weights}"
export HF_HOME=/workspace/hf
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src"
export WEIGHTS="$REPO_DIR/weights"

cd "$REPO_DIR"
mkdir -p "$WEIGHTS"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

push() {  # push <local file> <path in repo>
    python scripts/hf_push.py --file "$1" --path-in-repo "$2" \
        --repo "$HF_MODEL" --repo-type model
}

terminate_pod() {
    log "TERMINATING POD ${RUNPOD_POD_ID:-unknown}"
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
        runpodctl remove pod "$RUNPOD_POD_ID" \
            || log "runpodctl FAILED -- TERMINATE MANUALLY IN THE CONSOLE"
    else
        log "runpodctl unavailable -- TERMINATE MANUALLY IN THE CONSOLE"
    fi
}

save_everything() {
    [ -f "$WEIGHTS/last.pt" ] && push "$WEIGHTS/last.pt" "last.pt"
    [ -f "$REPO_DIR/dist/agecraft.onnx" ] && push "$REPO_DIR/dist/agecraft.onnx" "agecraft.onnx"
    # Newer torch exporters put the weights in a sidecar next to the graph.
    # The .onnx alone is a few hundred KB of topology and no model at all.
    [ -f "$REPO_DIR/dist/agecraft.onnx.data" ] && \
        push "$REPO_DIR/dist/agecraft.onnx.data" "agecraft.onnx.data"
    [ -f "$LOG_FILE" ] && push "$LOG_FILE" "train.log"
    return 0
}

# ---- LAYER 1: WATCHDOG ------------------------------------------------------
# Hard deadline, independent of everything below. Fires even if training hangs,
# the uploader dies, or the connection to you is long gone.
(
    sleep $(( MAX_HOURS * 3600 ))
    echo "[WATCHDOG] ${MAX_HOURS}h deadline hit. Saving what exists, then killing pod."
    cd "$REPO_DIR"
    save_everything || true
    terminate_pod
) &
WATCHDOG=$!
log "watchdog armed: hard kill in ${MAX_HOURS}h"

# ---- deps -------------------------------------------------------------------
# onnxscript is not optional on torch >= 2.6: without it the ONNX export dies
# at the very end of the run, after the GPU money is already spent.
pip install -q "huggingface_hub>=0.23" pyyaml onnx onnxruntime onnxscript 2>&1 | tail -1

python - << 'PY' || { log "GPU CHECK FAILED"; terminate_pod; exit 1; }
import torch
assert torch.cuda.is_available(), "NO CUDA -- wrong template"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}  {p.total_memory/1e9:.1f}GB  bf16={p.major >= 8}")
PY

# ---- PROVE WE CAN WRITE TO HF BEFORE SPENDING ANYTHING ----------------------
# This is the exact check that failed silently last time, except last time the
# token was read-only and nobody found out until the run was already over.
[ -z "${HF_TOKEN:-}" ] && { log "ERROR: HF_TOKEN not set"; terminate_pod; exit 1; }

python - << 'PY' || { log "HF WRITE TEST FAILED -- aborting before spending"; terminate_pod; exit 1; }
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
who = api.whoami()
role = who.get("auth", {}).get("accessToken", {}).get("role")
assert role == "write", f"token role is {role!r}, needs to be 'write'"
repo = os.environ["HF_MODEL"]
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=b"ok", path_in_repo="ping.txt",
                repo_id=repo, repo_type="model")
print(f"HF write OK -> {repo}")
PY

# ---- data -------------------------------------------------------------------
python scripts/prepare_dataset.py --dataset "$HF_DATASET" --root "$DATA_ROOT" \
    || { log "DATASET PREP FAILED"; terminate_pod; exit 1; }

# ---- resume from wherever the last run stopped ------------------------------
python - << 'PY'
import os, shutil
from pathlib import Path
dest = Path(os.environ["WEIGHTS"]) / "last.pt"
if dest.exists():
    print("local checkpoint present -- resuming from it")
else:
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=os.environ["HF_MODEL"], filename="last.pt",
                            token=os.environ["HF_TOKEN"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        print("pulled previous checkpoint from HF -- resuming")
    except Exception as e:
        print(f"no previous checkpoint ({type(e).__name__}) -- starting fresh")
PY

# ---- LAYER 2: STREAM CHECKPOINTS OUT WHILE TRAINING RUNS --------------------
uploader() {
    local last_seen=""
    while true; do
        sleep $(( CKPT_EVERY_MIN * 60 ))
        [ -f "$WEIGHTS/last.pt" ] || continue
        local m
        m=$(stat -c %Y "$WEIGHTS/last.pt" 2>/dev/null || echo "")
        [ -n "$m" ] && [ "$m" != "$last_seen" ] || continue
        if push "$WEIGHTS/last.pt" "last.pt" >/dev/null; then
            last_seen="$m"
            log "checkpoint pushed to $HF_MODEL"
        else
            log "checkpoint push failed -- retrying next cycle"
        fi
    done
}
uploader &
UPLOADER=$!
log "checkpoint uploader running every ${CKPT_EVERY_MIN}min"

# ---- train ------------------------------------------------------------------
TRAIN_SECONDS=$(( MAX_HOURS * 3600 - RESERVE_MIN * 60 ))
RESUME=()
[ -f "$WEIGHTS/last.pt" ] && RESUME=(--resume "$WEIGHTS/last.pt")

log "training up to $(( TRAIN_SECONDS / 60 ))min (${RESERVE_MIN}min reserved for export+upload)"
python -m agecraft.train --config "$CONFIG" --max-seconds "$TRAIN_SECONDS" "${RESUME[@]}"
TRAIN_RC=$?
log "training exited rc=$TRAIN_RC"

kill $UPLOADER 2>/dev/null

# ---- export + LAYER 3: SAVE, THEN SELF-KILL ---------------------------------
if [ -f "$WEIGHTS/last.pt" ]; then
    log "exporting ONNX"
    python -m agecraft.export_onnx --model "$WEIGHTS/last.pt" \
        --out "$REPO_DIR/dist/agecraft.onnx" \
        || log "ONNX export failed -- weights are still safe, export locally later"
else
    log "WARNING: no checkpoint on disk; nothing trained?"
fi

log "final upload"
save_everything || log "FINAL UPLOAD FAILED -- do not terminate, rescue manually"

kill $WATCHDOG 2>/dev/null
terminate_pod
