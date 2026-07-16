# Kaggle notebook: generate the AgeCraft dataset on a free T4.
#
# HOW TO USE
#   1. kaggle.com -> Code -> New Notebook
#   2. Settings -> Accelerator -> "GPU T4 x2"   (NOT P100: Pascal has no fp16
#      tensor cores and bitsandbytes NF4 is unreliable there)
#   3. Settings -> Persistence -> "Files only"  <- CRITICAL, else you lose
#      everything when the 12h session dies
#   4. Paste this whole file into one cell and run.
#   5. When the session dies (it will), just re-run. It resumes.
#
# QUOTA MATH (~30 GPU-h/week, 12h max session):
#   768px, 5 anchors, NF4 on T4  ~= 20 s/image
#   1200 identities x 5 ages     = 6,000 images = ~33 hours = ~1.5 weeks
#   Cut to --limit 400 for a first real dataset in one 12h session.
#
# If 1.5 weeks of babysitting sounds bad, renting a 4090 for ~3h costs ~$1.50
# and does the whole thing. See README.

import os
import subprocess
import sys

# ---------------------------------------------------------------- setup
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "diffusers>=0.30", "transformers>=4.40", "accelerate>=0.30",
                "bitsandbytes>=0.43", "sentencepiece", "protobuf"], check=True)

REPO = "/kaggle/working/agecraft"
if not os.path.exists(REPO):
    # Option A: clone your repo (push it to GitHub first)
    # subprocess.run(["git", "clone", "https://github.com/YOU/agecraft", REPO])
    #
    # Option B: upload the repo as a Kaggle Dataset and copy it:
    subprocess.run(["cp", "-r", "/kaggle/input/agecraft-src", REPO], check=True)

sys.path.insert(0, f"{REPO}/src")
os.chdir(REPO)

# ---------------------------------------------------------------- sanity
import torch

print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory/1e9:.1f}GB, sm_{p.major}{p.minor}")
    if p.major < 8:
        print("pre-Ampere -> using fp16 (no bf16 support on T4/P100)")
DTYPE = "fp16" if torch.cuda.get_device_properties(0).major < 8 else "bf16"

# ---------------------------------------------------------------- stage 1
# Prompts: no GPU needed, no Ollama on Kaggle -> templates.
# The template path is deterministic and produces a perfectly balanced grid;
# Qwen only adds phrasing variety, which is nice-to-have, not load-bearing.
if not os.path.exists("data/prompts.jsonl"):
    subprocess.run([sys.executable, "-m", "agecraft.prompts.generate_prompts",
                    "--out", "data/prompts.jsonl", "-n", "1200",
                    "--no-llm", "--seed", "1337"], check=True)

# ---------------------------------------------------------------- stage 2
# Persist to /kaggle/working so "Files only" persistence keeps it across
# sessions. 20GB limit: 6000 PNGs at 768px is ~4-5GB. Fits.
OUT = "/kaggle/working/data/raw"

subprocess.run([sys.executable, "-m", "agecraft.data.generate_pairs",
                "--prompts", "data/prompts.jsonl",
                "--out", OUT,
                "--size", "768",
                "--steps", "4",
                "--mode", "offload",
                "--quantize", "nf4",
                "--dtype", DTYPE,
                "--anchors", "20,35,50,65,80",   # 5 not 7: 30% cheaper
                "--limit", "400"],               # one session's worth
               check=True)

# ---------------------------------------------------------------- stage 3
# Filtering is cheap (small models) -- run it in the same session so you can
# read the keep rate before spending another 12h.
# --no-deps is mandatory: facenet-pytorch pins torch<=2.3.0 and would
# downgrade Kaggle's PyTorch mid-session, killing CUDA. Its pins are stale;
# the library runs fine on current torch.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                "facenet-pytorch"], check=True)
subprocess.run([sys.executable, "-m", "agecraft.data.filter_pairs",
                "--raw", OUT,
                "--out", "/kaggle/working/data/processed"], check=True)

print("""
================= READ THE KEEP RATE ABOVE =================
 >20%  -> seed-locking works. Raise --limit, keep going.
 5-20% -> works but wasteful. Tighten AGE_PHRASES first.
 <5%   -> seed-locking too weak alone. Stop and rethink
          before burning more quota.

Also: open data/raw/id_000000/ and LOOK at the 5 images.
The metric can pass sequences that look wrong to a human.
============================================================
""")
