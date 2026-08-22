"""Stage 4 — train the delta net.

Loss = L1 + perceptual + adversarial, plus one addition of my own:

  IDENTITY-CYCLE LOSS. Re-aging X from age A to age A must produce delta == 0.
  We inject these no-op samples into every batch and penalise any non-zero
  delta. It costs nothing, and it anchors the model's notion of "no change",
  which is what keeps the sliders linear and well-behaved near zero. Without
  it, delta*0.5 does not reliably mean "half as old" and your headline feature
  -- the disentangled sliders -- feels mushy.

PERCEPTUAL LOSS / LICENCE WARNING
---------------------------------
LPIPS is the default choice and it uses a VGG or AlexNet backbone whose weights
have their own provenance questions. Used as a *training loss* the weights are
not embedded in or distributed with our model, which is a much weaker exposure
than shipping them -- but it is not zero exposure, and I am not a lawyer.

`--perceptual none` trains with L1 + GAN only. It converges slower and the skin
texture is a little softer, but the lineage is unambiguous. Start there if you
want to be able to ship without a legal review; switch on LPIPS only once
counsel has signed off. This flag exists precisely so the decision is yours and
reversible, not baked in.

Usage:
    python -m agecraft.train --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data.dataset import AgePairDataset
from .models.discriminator import PatchDiscriminator
from .models.fran import FRANGenerator


def build_perceptual(kind: str, device: str):
    if kind == "none":
        return None
    if kind == "lpips":
        import lpips
        return lpips.LPIPS(net="vgg").to(device).eval()
    raise ValueError(f"unknown perceptual loss: {kind}")


def hinge_d(real_logits, fake_logits):
    return (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()) * 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop cleanly after this long, checkpointing first")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    ds = AgePairDataset(cfg["pairs_file"], crop=cfg["crop"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    print(f"{len(ds)} pairs, {len(dl)} steps/epoch")

    G = FRANGenerator(base=cfg["base_channels"]).to(device)
    D = PatchDiscriminator().to(device)
    P = build_perceptual(cfg["perceptual"], device)

    optG = torch.optim.AdamW(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    optD = torch.optim.AdamW(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler(enabled=cfg["amp"])

    start_epoch = 0
    if args.resume and args.resume.exists():
        ck = torch.load(args.resume, map_location=device)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
        start_epoch = ck["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")

    w = cfg["loss_weights"]
    step = 0
    deadline = time.time() + args.max_seconds if args.max_seconds else None

    def save_ckpt(ep: int) -> None:
        # Written aside and moved into place: run_training.sh uploads last.pt
        # while training continues, and a half-written file would ship as a
        # corrupt checkpoint.
        tmp = out / "last.pt.tmp"
        torch.save({"G": G.state_dict(), "D": D.state_dict(),
                    "optG": optG.state_dict(), "optD": optD.state_dict(),
                    "epoch": ep, "cfg": cfg}, tmp)
        tmp.replace(out / "last.pt")

    for epoch in range(start_epoch, cfg["epochs"]):
        G.train(); D.train()
        t0 = time.time()

        for batch in dl:
            src = batch["src"].to(device, non_blocking=True)
            tgt = batch["tgt"].to(device, non_blocking=True)
            sa = batch["src_age"].to(device)
            ta = batch["tgt_age"].to(device)

            # ---------------- discriminator
            with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                with torch.no_grad():
                    fake, _ = G.reage(src, sa, ta)
                d_loss = hinge_d(D(tgt, ta), D(fake.detach(), ta))
            optD.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(optD)

            # ---------------- generator
            with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                fake, delta = G.reage(src, sa, ta)

                l1 = F.l1_loss(fake, tgt)
                adv = -D(fake, ta).mean()
                perc = P(fake, tgt).mean() if P is not None else torch.zeros((), device=device)

                # identity-cycle: A -> A must be a no-op
                _, null_delta = G.reage(src, sa, sa)
                idc = null_delta.abs().mean()

                # mild delta sparsity: prefer the smallest edit that works.
                # keeps backgrounds pristine and makes the exported layer clean.
                sparse = delta.abs().mean()

                g_loss = (w["l1"] * l1 + w["adv"] * adv + w["perceptual"] * perc
                          + w["identity_cycle"] * idc + w["delta_sparsity"] * sparse)

            optG.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(optG)
            scaler.update()

            if step % cfg["log_every"] == 0:
                print(f"e{epoch} s{step}  G {g_loss.item():.4f}  D {d_loss.item():.4f}  "
                      f"l1 {l1.item():.4f}  idc {idc.item():.5f}  "
                      f"|delta| {delta.abs().mean().item():.4f}")
            step += 1

            if deadline is not None and time.time() > deadline:
                # Resume picks up at epoch+1, so the tail of this epoch is
                # dropped rather than replayed -- the data is shuffled, and it
                # keeps loader state out of the checkpoint.
                save_ckpt(epoch)
                print(f"time budget reached in epoch {epoch} at step {step}; "
                      f"checkpointed and stopping")
                return 0

        save_ckpt(epoch)
        if (epoch + 1) % cfg["save_every"] == 0:
            torch.save({"G": G.state_dict(), "cfg": cfg}, out / f"G_e{epoch:03d}.pt")
        print(f"epoch {epoch} done in {time.time()-t0:.0f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
