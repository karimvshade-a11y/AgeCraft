#!/usr/bin/env bash
# AgeCraft on a rented RTX 4090 (RunPod Community Cloud, ~$0.34/hr).
#
# TOTAL EXPECTED COST: ~$1-2 for data, ~$3-4 if you also train here.
# You will deposit $10 minimum. You'll have change left.
#
# ---------------------------------------------------------------------------
# POD SETUP -- get these right or you'll waste money debugging
# ---------------------------------------------------------------------------
#   Template          : "RunPod PyTorch 2.4" (or any recent PyTorch template)
#   GPU               : RTX 4090 (Community Cloud, $0.34/hr)
#   Container Disk    : 30 GB
#   Volume Disk       : 80 GB  mounted at /workspace   <-- NOT the default 20GB
#
# THE DISK TRAP: FLUX.1-schnell is ~24GB of weights. On the default 20GB volume
# the download dies at ~90% with a confusing error after you've already paid for
# 15 minutes. 80GB gives room for weights (24GB) + dataset (~6GB) + slack.
#
# THE BILLING TRAP: RunPod bills per second while the pod EXISTS, not while it
# computes. A stopped pod still bills for storage. When you're done:
# download your data, then TERMINATE the pod (not "stop").
#
# ---------------------------------------------------------------------------
# USAGE: paste into the RunPod web terminal, or scp this file and run it.
# ---------------------------------------------------------------------------

set -euo pipefail

# Keep HF cache on the persistent volume so a pod restart doesn't re-download
# 24GB. This single line has saved more money than any other in this file.
export HF_HOME=/workspace/hf
export PYTHONUNBUFFERED=1
mkdir -p /workspace/hf

cd /workspace

# --- get the code -----------------------------------------------------------
if [ ! -d agecraft ]; then
  # Option A: clone from your GitHub
  #   git clone https://github.com/YOU/agecraft.git
  # Option B: from your machine, before running this:
  #   scp -P <port> -r agecraft root@<host>:/workspace/
  echo "ERROR: /workspace/agecraft not found. Clone or scp it first."
  exit 1
fi
cd agecraft

# --- deps -------------------------------------------------------------------
pip install -q --upgrade pip

# Generation stack. Safe to resolve normally.
pip install -q "diffusers>=0.30" "transformers>=4.40" "accelerate>=0.30" \
               sentencepiece protobuf pyyaml requests

# facenet-pytorch MUST be --no-deps. Its setup.py declares torch<=2.3.0,
# torchvision<=0.18.0, numpy<2.0.0, Pillow<10.3.0 -- all stale pins. Without
# --no-deps pip will DOWNGRADE the pod's PyTorch to 2.3 mid-install, silently
# breaking CUDA and costing you an hour of confused debugging on the clock.
# The library itself works fine on current torch; only its metadata is wrong.
pip install -q --no-deps facenet-pytorch

# Prove the downgrade didn't happen before we spend money.
python - << 'PY'
import torch, torchvision, numpy, PIL
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} "
      f"| numpy {numpy.__version__} | pillow {PIL.__version__}")
assert torch.cuda.is_available(), "CUDA GONE -- something downgraded torch. Stop."
from facenet_pytorch import MTCNN, InceptionResnetV1  # noqa: F401
print("facenet imports fine on current torch (its pins were lying) ✓")
PY

# --- sanity: fail fast and loud --------------------------------------------
python - << 'PY'
import torch, sys
assert torch.cuda.is_available(), "NO CUDA. Wrong template. Terminate and restart."
p = torch.cuda.get_device_properties(0)
gb = p.total_memory / 1e9
print(f"GPU: {p.name}  {gb:.1f}GB  sm_{p.major}{p.minor}")
if gb < 20:
    print(f"WARNING: {gb:.1f}GB is not a 4090. Use --quantize nf4 or terminate.")
    sys.exit(1)
print("bf16 supported:", p.major >= 8)
PY

df -h /workspace | tail -1 | awk '{print "volume free: "$4}'

export PYTHONPATH=/workspace/agecraft/src

# --- stage 1: prompts (seconds, no GPU) ------------------------------------
python -m agecraft.prompts.generate_prompts \
    --out data/prompts.jsonl -n 1200 --no-llm --seed 1337

# ===========================================================================
# STAGE 2A: SMOKE TEST FIRST. 50 identities, ~5 minutes, ~$0.03.
#
# DO NOT SKIP THIS. It downloads the 24GB of weights (one-time, ~5 min on
# datacenter bandwidth) and proves the seed-lock works BEFORE you commit to
# the full run. If the keep rate is bad, you've spent 3 cents finding out
# instead of a dollar.
# ===========================================================================
echo "=== SMOKE TEST: 50 identities ==="
time python -m agecraft.data.generate_pairs \
    --prompts data/prompts.jsonl \
    --out /workspace/data/raw \
    --limit 50 \
    --size 768 --steps 4 \
    --mode offload \
    --dtype bf16 \
    --anchors 20,35,50,65,80

python -m agecraft.data.filter_pairs \
    --raw /workspace/data/raw \
    --out /workspace/data/processed

cat << 'EOF'

============================================================
STOP AND READ THE KEEP RATE ABOVE.

  >20%   -> works. Uncomment STAGE 2B below and re-run.
  5-20%  -> works, wasteful. Consider tightening AGE_PHRASES.
  <5%    -> STOP. Seed-locking is too weak on its own.
            Terminate the pod (you've spent ~$0.05) and we
            rethink before spending more.

Then LOOK at the images with your own eyes:
  /workspace/data/raw/id_000000/*.png
The metric can pass sequences a human would reject.
============================================================

EOF

# ===========================================================================
# STAGE 2B: FULL RUN. Uncomment only after the smoke test looks good.
# 1200 identities x 5 ages = 6000 images @ ~1.2s = ~2h = ~$0.70
# ===========================================================================
# python -m agecraft.data.generate_pairs \
#     --prompts data/prompts.jsonl \
#     --out /workspace/data/raw \
#     --size 768 --steps 4 --mode offload --dtype bf16 \
#     --anchors 20,35,50,65,80
#
# python -m agecraft.data.filter_pairs \
#     --raw /workspace/data/raw --out /workspace/data/processed

# ===========================================================================
# STAGE 3 (OPTIONAL): train here instead of on your 1650.
#
#   Here  : base=64, crop=512, ~8h, ~$2.80 -> the good model
#   Home  : base=32, crop=256, ~10h overnight, $0 -> the softer model
#
# For under three dollars I'd train here and download only the weights
# (140MB) instead of the whole dataset. Your call.
# ===========================================================================
# python -m agecraft.train --config configs/base.yaml
# python -m agecraft.export_onnx --model weights/last.pt --out dist/agecraft.onnx

# ===========================================================================
# STAGE 4: GET YOUR STUFF OFF THE POD, THEN TERMINATE IT.
#
# Only the survivors matter -- filter first, then tar only what passed.
# Full raw set is ~6GB; keepers are ~1.5GB.
# ===========================================================================
cat << 'EOF'

To download (run FROM YOUR WINDOWS MACHINE, not the pod):

  # dataset only (if training at home):
  scp -P <port> root@<host>:/workspace/data/processed/pairs.jsonl .
  tar -czf keepers.tgz -C /workspace/data raw    # run on pod first
  scp -P <port> root@<host>:/workspace/keepers.tgz .

  # or weights only (if you trained on the pod) -- 140MB:
  scp -P <port> root@<host>:/workspace/agecraft/weights/last.pt .
  scp -P <port> root@<host>:/workspace/agecraft/dist/agecraft.onnx .

THEN TERMINATE THE POD. Not "stop" -- terminate. A stopped pod still
bills you for the 80GB volume.

EOF
