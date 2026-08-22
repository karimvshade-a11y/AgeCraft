"""Stage 5 — export to ONNX for shipping.

Ship ONNX Runtime, not PyTorch. One artifact runs on CPU, CUDA, DirectML
(Windows AMD/Intel) and CoreML (Mac), and your users never install CUDA. It
also cuts the installer from ~2.5GB to ~150MB, which matters more for
conversion than any feature you'll build.

Dynamic H/W axes are essential: the model is fully convolutional and users will
throw 4K portraits at it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .models.fran import FRANGenerator


class ExportWrapper(torch.nn.Module):
    """Folds age-channel construction into the graph so the app just passes
    scalars and never has to replicate the normalisation convention."""

    def __init__(self, g: FRANGenerator):
        super().__init__()
        self.g = g

    def forward(self, img: torch.Tensor, src_age: torch.Tensor,
                tgt_age: torch.Tensor) -> torch.Tensor:
        return self.g(self.g.build_input(img, src_age, tgt_age))  # delta only


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    # 18, not 17: the ReflectionPad in BlurPool2d trips the opset-17 version
    # converter ("No Adapter To Version 17 for Pad"). It recovers via fallback,
    # but 18 exports clean and DirectML/CoreML both support it.
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu")
    cfg = ck.get("cfg", {})

    # A NaN here is worth naming precisely, because the two cases have very
    # different prognoses. NaN in the learned weights means the training run is
    # lost. NaN confined to BatchNorm running stats means the weights are fine
    # and only the eval-mode statistics need recalibrating -- and it hides,
    # because train() mode uses batch statistics and so the loss looks healthy
    # right up until you export.
    bad_w = [k for k, v in ck["G"].items()
             if torch.is_floating_point(v) and "running_" not in k
             and not torch.isfinite(v).all()]
    bad_bn = [k for k, v in ck["G"].items()
              if torch.is_floating_point(v) and "running_" in k
              and not torch.isfinite(v).all()]
    if bad_w:
        print(f"ERROR: {len(bad_w)} weight tensors contain NaN/Inf, e.g. {bad_w[:3]}")
        return 1
    if bad_bn:
        print(f"ERROR: {len(bad_bn)} BatchNorm running stats contain NaN/Inf, "
              f"e.g. {bad_bn[:3]}")
        print("The learned weights are intact. Recalibrate the statistics with "
              "scripts/recalibrate_bn.py, then export again.")
        return 1

    G = FRANGenerator(base=cfg.get("base_channels", 64))
    G.load_state_dict(ck["G"])
    G.eval()

    wrapper = ExportWrapper(G).eval()
    dummy = (torch.randn(1, 3, 512, 512),
             torch.tensor([30.0]), torch.tensor([70.0]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, str(args.out),
        input_names=["image", "src_age", "tgt_age"],
        output_names=["delta"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "src_age": {0: "batch"},
            "tgt_age": {0: "batch"},
            "delta": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"wrote {args.out}")

    # Verify at a DIFFERENT resolution than we traced at -- this is the check
    # that actually catches broken dynamic axes.
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        img = np.random.randn(1, 3, 384, 640).astype(np.float32)
        d = sess.run(None, {"image": img,
                            "src_age": np.array([30.0], np.float32),
                            "tgt_age": np.array([70.0], np.float32)})[0]
        assert d.shape == img.shape, f"shape mismatch {d.shape} vs {img.shape}"
        print(f"verified at 384x640 -> delta {d.shape}, "
              f"range [{d.min():.3f}, {d.max():.3f}]")
        if not np.isfinite(d).all():
            # This check used to only print. A run once exported a model whose
            # every output was NaN, said so on this line, and shipped anyway.
            print("ERROR: exported model produces non-finite output. "
                  "Not shipping this file.")
            args.out.unlink(missing_ok=True)
            return 1
    except ImportError:
        print("onnxruntime not installed; skipped verification")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
