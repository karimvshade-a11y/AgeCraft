# AgeCraft

Local, private, **sellable** face re-aging with per-region control.

Everything runs on your machine. No API. No upload. No subscription to anyone.

---

## The thesis, in three lines

1. **Predict a delta, not a face.** `output = input + delta`. Identity,
   background, glasses and earrings survive by construction.
2. **The delta is an image, so you can take it apart.** Mask it by face region
   → separate sliders for wrinkles / gray hair / eye bags / jawline. Nobody
   else ships this, because nobody else's architecture allows it.
3. **Train on Apache-2.0 synthetic data so you can legally sell it.** The whole
   open re-aging lineage (StyleGAN → FFHQ → SAM → every FRAN reimplementation)
   is non-commercial. That's the gap. See [PROVENANCE.md](PROVENANCE.md) — it's
   the most important file here.

## Pipeline

```
prompts.jsonl        Qwen/Ollama phrases a balanced identity grid  (offline, once)
    │
    ▼
data/raw             FLUX.1-schnell, seed+latent locked, 7 ages each
    │                (Apache 2.0 — no ID adapter, no InsightFace)
    ▼
data/processed       ruthless QC: identity gate + age-monotonicity gate
    │                (expect to discard 60-80%. that's the design working.)
    ▼
weights/last.pt      delta U-Net, 34.5M params
    │
    ▼
dist/agecraft.onnx   ships everywhere: CPU / CUDA / DirectML / CoreML
```

## Quickstart

```bash
pip install -e ".[train]"
ollama pull qwen2.5:14b-instruct     # optional; templates work without it

dvc repro                            # runs the whole chain
```

Or stage by stage:

```bash
python -m agecraft.prompts.generate_prompts --out data/prompts.jsonl -n 4000
python -m agecraft.data.generate_pairs --prompts data/prompts.jsonl --out data/raw
python -m agecraft.data.filter_pairs --raw data/raw --out data/processed
python -m agecraft.train --config configs/base.yaml
python -m agecraft.export_onnx --model weights/last.pt --out dist/agecraft.onnx
```

Inference, with the feature that sells it:

```bash
python -m agecraft.inference --model weights/last.pt --image me.jpg \
    --src-age 34 --tgt-age 70 \
    --sliders wrinkles=1.0,gray_hair=0.2,eye_bags=0.5 \
    --out out.png --export-layer delta.png
```

`--export-layer` writes the delta as a plain PNG (128 = neutral) that
composites in Photoshop or Resolve as a Linear Light layer at any opacity.
That's the non-destructive workflow VFX shops actually pay for.

## Cost

| | |
|---|---|
| Software | $0 — every dependency is free and permissively licensed |
| Training data | $0 — synthesised locally |
| Inference | $0 — runs on the user's machine |
| Your GPU, ~2-4 nights | the only real cost |

## Status

**Verified working:**
- delta U-Net — zero-init identity, bounded output, odd/non-square shapes, gradients ✅
- region sliders — isolation, linearity, feathering, background protection ✅
- prompt grid — balanced <0.5% across all axes, deterministic, 4000/4000 valid ✅
- FLUX latent packing — round-trip exact, seed-lock verified ✅
- ONNX export — opset 18 clean, dynamic shapes verified at a non-traced size ✅

**Not built yet:**
- PySide6 desktop app (the shell around `inference.py`)
- BiSeNet parse-map integration (sliders currently need a parse map passed in;
  without one they degrade to a single global opacity)
- video mode + optical-flow delta smoothing
- consent gate / C2PA output marking — **required before you ship**, see
  PROVENANCE.md § Safety

**Untested:** the FLUX generation and training loops have not been run against
real weights here — no GPU in this environment. The shape math, latent packing
and export are verified; the yield rate of seed-locking is the open empirical
question. Run 50 identities first and look at the keep rate before committing
a full night.

## Tuning the one number that matters

`filter.id_thresh` (params.yaml, default 0.55) is the dial between data volume
and identity fidelity. Too low → the model learns that aging changes bone
structure, and faces morph into strangers. Too high → you throw away everything
and have no data. Check `data/processed/rejected.jsonl` and the per-ancestry
keep-rate report that `filter_pairs.py` prints. **Uneven keep rates across
ancestry mean a biased model** — that report exists because a balanced input
grid is worthless if the filter silently rejects one group harder.
