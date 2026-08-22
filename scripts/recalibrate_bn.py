"""Rebuild a checkpoint's BatchNorm running statistics from real data.

WHY THIS EXISTS
---------------
BatchNorm keeps running_mean/running_var, updated by the forward pass rather
than by the optimiser. One fp16 overflow anywhere in a step writes NaN into
them, and nothing puts it back: GradScaler skips the optimiser step on
non-finite gradients, but the statistics have already been written.

The failure then hides for the rest of the run. train() mode normalises with
batch statistics, so the loss stays healthy and the weights keep learning
normally. Only eval() mode reads the running statistics -- so the first sign of
trouble is an exported model whose every output is NaN, hours later.

The weights are usually untouched. Running statistics are not learned, they are
measured, so they can simply be measured again:

    python scripts/recalibrate_bn.py --model last.pt --pairs pairs.jsonl --out fixed.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from agecraft.data.dataset import AgePairDataset
from agecraft.models.fran import FRANGenerator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop", type=int, default=None, help="default: the training crop")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu")
    cfg = ck.get("cfg", {})
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    crop = args.crop or cfg.get("crop", 256)

    bad = [k for k, v in ck["G"].items()
           if torch.is_floating_point(v) and "running_" not in k and not torch.isfinite(v).all()]
    if bad:
        print(f"ERROR: {len(bad)} learned weight tensors are non-finite, e.g. {bad[:3]}")
        print("Recalibration cannot help -- the training run itself is lost.")
        return 1

    G = FRANGenerator(base=cfg.get("base_channels", 64)).to(device)
    G.load_state_dict(ck["G"])

    n_bn = 0
    for mod in G.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None  # cumulative average: every batch weighs equally
            n_bn += 1
    print(f"reset {n_bn} BatchNorm layers; measuring over {args.batches} batches "
          f"at {crop}px on {device}")

    ds = AgePairDataset(args.pairs, crop=crop)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    drop_last=True)

    G.train()  # train() is the point: this is what updates the statistics
    seen = 0
    with torch.no_grad():
        while seen < args.batches:
            for b in dl:
                if seen >= args.batches:
                    break
                src = b["src"].to(device)
                sa = b["src_age"].to(device)
                ta = b["tgt_age"].to(device)
                # Mirror the mixture the training loop actually fed these
                # statistics: two ageing passes per step (one inside the
                # discriminator step, one in the generator step) and one
                # identity pass. Calibrating on ageing alone leaves a constant
                # non-zero delta at src_age == tgt_age -- exactly the no-op the
                # identity-cycle loss exists to guarantee.
                G.reage(src, sa, ta)
                G.reage(src, sa, ta)
                G.reage(src, sa, sa)
                seen += 1
                if seen % 20 == 0:
                    print(f"  {seen}/{args.batches}")

    still_bad = [k for k, v in G.state_dict().items()
                 if torch.is_floating_point(v) and not torch.isfinite(v).all()]
    if still_bad:
        print(f"ERROR: {len(still_bad)} tensors still non-finite after recalibration")
        return 1

    ck["G"] = {k: v.cpu() for k, v in G.state_dict().items()}
    ck["bn_recalibrated"] = True
    torch.save(ck, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
