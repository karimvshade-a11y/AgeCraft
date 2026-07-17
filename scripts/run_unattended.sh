#!/usr/bin/env bash
# Unattended, self-terminating data generation.
#
# WHY THIS EXISTS
# ---------------
# A pod does not stop when your PC does. It is a machine in a datacentre and it
# bills per second until someone clicks Terminate. A power cut at home left one
# running for 24.5 hours and burned $8.57 doing nothing.
#
# The lesson is not "remember to terminate". Humans forget, lose power, fall
# asleep. The lesson is: THE JOB MUST KILL ITS OWN MACHINE.
#
# Three layers here, each independent:
#   1. WATCHDOG   -- hard kill after MAX_HOURS no matter what. Even if the job
#                    hangs, even if you vanish. This is the one that matters.
#   2. UPLOAD     -- results go to a private HF dataset BEFORE termination, so
#                    the pod dying costs you nothing.
#   3. SELF-KILL  -- terminate on completion, success or failure.
#
# Layer 1 is the seatbelt. Set MAX_HOURS to something you can afford to lose,
# because that IS your maximum loss.
#
# USAGE:
#   export HF_TOKEN=hf_...
#   export HF_DATASET=yourname/agecraft-data     # private, auto-created
#   export MAX_HOURS=8
#   nohup bash scripts/run_unattended.sh > /workspace/run.log 2>&1 &
#   # now you can close the browser, lose power, whatever. It handles itself.

set -uo pipefail

MAX_HOURS="${MAX_HOURS:-10}"
LIMIT="${LIMIT:-1200}"
SIZE="${SIZE:-768}"
ANCHORS="${ANCHORS:-20,35,50,65,80}"
MODE="${MODE:-offload}"
QUANT="${QUANT:-}"
HF_DATASET="${HF_DATASET:-Abdelkarim40/agecraft-data}"

export HF_HOME=/workspace/hf
export PYTHONUNBUFFERED=1
export PYTHONPATH=/workspace/agecraft/src
cd /workspace/agecraft

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

terminate_pod() {
    log "TERMINATING POD ${RUNPOD_POD_ID:-unknown}"
    if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
        runpodctl remove pod "$RUNPOD_POD_ID"
    else
        log "runpodctl unavailable -- TERMINATE MANUALLY IN THE CONSOLE"
    fi
}

# ---- LAYER 1: WATCHDOG -----------------------------------------------------
# Detached subshell with a hard deadline. Survives the main job hanging,
# crashing, or waiting on a prompt that will never be answered.
(
    sleep $(( MAX_HOURS * 3600 ))
    echo "[WATCHDOG] ${MAX_HOURS}h deadline hit. Killing pod."
    if [ -n "${HF_DATASET:-}" ]; then
        cd /workspace/agecraft
        python scripts/upload_results.py --dataset "$HF_DATASET" || true
    fi
    terminate_pod
) &
WATCHDOG=$!
log "watchdog armed: hard kill in ${MAX_HOURS}h"

# ---- deps ------------------------------------------------------------------
pip install -q "diffusers>=0.30" "transformers>=4.40" "accelerate>=0.30" \
               sentencepiece protobuf pyyaml requests 2>&1 | tail -1
pip install -q --no-deps facenet-pytorch 2>&1 | tail -1
[ -n "$QUANT" ] && pip install -q "bitsandbytes>=0.43" 2>&1 | tail -1

python -c "
import torch; assert torch.cuda.is_available(), 'NO CUDA'
p = torch.cuda.get_device_properties(0)
print(f'GPU: {p.name} {p.total_memory/1e9:.1f}GB')
" || { terminate_pod; exit 1; }

[ -z "${HF_TOKEN:-}" ] && { log "ERROR: no HF_TOKEN"; terminate_pod; exit 1; }

# ---- PROVE UPLOAD WORKS BEFORE SPENDING ANYTHING ---------------------------
python - << 'PY' || { log "UPLOAD TEST FAILED -- aborting before spending"; terminate_pod; exit 1; }
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
ds = os.environ.get("HF_DATASET","Abdelkarim40/agecraft-data")
api.create_repo(ds, repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj=b"ping", path_in_repo="ping.txt",
                repo_id=ds, repo_type="dataset")
print(f"upload test OK -> {ds}")
PY

# ---- the job ---------------------------------------------------------------
run_job() {
    [ -f data/prompts.jsonl ] || \
        python -m agecraft.prompts.generate_prompts \
            --out data/prompts.jsonl -n 1200 --no-llm --seed 1337 || return 1

    local qflag=()
    [ -n "$QUANT" ] && qflag=(--quantize "$QUANT")

    log "generating $LIMIT identities x $(echo $ANCHORS | tr ',' ' ' | wc -w) ages"
    python -m agecraft.data.generate_pairs \
        --prompts data/prompts.jsonl --out /workspace/data/raw \
        --limit "$LIMIT" --size "$SIZE" --steps 4 \
        --mode "$MODE" --dtype bf16 --anchors "$ANCHORS" "${qflag[@]}" || return 1

    log "uploading RAW data (before filtering -- late failures cannot lose it)"
    tar -czf /workspace/raw_backup.tgz -C /workspace/data raw
    python - << 'PY' || log "raw upload failed (continuing)"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
ds = os.environ.get("HF_DATASET","Abdelkarim40/agecraft-data")
api.upload_file(path_or_fileobj="/workspace/raw_backup.tgz",
                path_in_repo="raw_backup.tgz", repo_id=ds, repo_type="dataset")
print("raw backup uploaded")
PY

    log "filtering"
    python -m agecraft.data.filter_pairs \
        --raw /workspace/data/raw --out /workspace/data/processed || return 1
}

run_job
JOB_RC=$?
log "job finished rc=$JOB_RC"

# ---- LAYER 2: UPLOAD BEFORE DYING ------------------------------------------
if [ -n "$HF_DATASET" ]; then
    log "uploading survivors to $HF_DATASET"
    python scripts/upload_results.py --dataset "$HF_DATASET" || \
        log "UPLOAD FAILED -- data will be lost on terminate!"
else
    log "WARNING: no HF_DATASET set. Data dies with the pod."
    log "Sleeping 30min so you can scp manually, then terminating."
    sleep 1800
fi

# ---- LAYER 3: SELF-KILL ----------------------------------------------------
kill $WATCHDOG 2>/dev/null
terminate_pod
