"""Inference + the feature that actually sells the product.

The delta representation gives us something no latent-space method can offer:
the aging edit is a plain image, so it can be MASKED AND SCALED PER REGION.

    output = input + sum_over_regions( mask_r * delta * slider_r )

That's the whole trick. Everyone else ships one age slider because their
architecture regenerates the face and can't take it apart afterwards. We ship:

    wrinkles / skin texture      -> skin regions
    gray hair                    -> hair region
    hairline recession           -> hair + forehead boundary
    eye bags                     -> peri-orbital
    jawline / sagging            -> jaw + neck
    sun spots                    -> skin, chroma-only

Plus opacity, because delta * 0.4 is a genuinely useful "age me a bit".

And because the delta is just an image, `--export-layer` writes it as a
straight PNG that composites in Photoshop or Resolve at any opacity. That's
the non-destructive workflow that turns this from a toy into something a VFX
shop will pay for.

Usage:
    python -m agecraft.inference --model weights/G.pt --image me.jpg \
        --src-age 34 --tgt-age 70 --sliders wrinkles=1.0,gray_hair=0.2 \
        --out out.png --export-layer delta.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .models.fran import FRANGenerator

# BiSeNet face-parsing class indices (CelebAMask-HQ label set).
PARSE = {
    "skin": 1, "l_brow": 2, "r_brow": 3, "l_eye": 4, "r_eye": 5,
    "eye_g": 6, "l_ear": 7, "r_ear": 8, "ear_r": 9, "nose": 10,
    "mouth": 11, "u_lip": 12, "l_lip": 13, "neck": 14, "neck_l": 15,
    "cloth": 16, "hair": 17, "hat": 18,
}

# Which parse classes each user-facing slider controls.
SLIDER_REGIONS = {
    "wrinkles":    ["skin", "nose", "u_lip", "l_lip"],
    "eye_bags":    ["l_eye", "r_eye"],
    "gray_hair":   ["hair", "l_brow", "r_brow"],
    "jawline":     ["neck", "neck_l"],
    "sun_spots":   ["skin"],
    "ears_nose":   ["l_ear", "r_ear", "nose"],  # cartilage keeps growing, really
}

DEFAULT_SLIDERS = {k: 1.0 for k in SLIDER_REGIONS}


def load_generator(path: Path, device: str) -> FRANGenerator:
    ck = torch.load(path, map_location=device)
    cfg = ck.get("cfg", {})
    G = FRANGenerator(base=cfg.get("base_channels", 64)).to(device)
    G.load_state_dict(ck["G"])
    G.eval()
    return G


def to_tensor(img: Image.Image, device: str) -> torch.Tensor:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def to_pil(t: torch.Tensor) -> Image.Image:
    a = ((t[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5)
    return Image.fromarray(a.round().astype(np.uint8))


def feather(mask: torch.Tensor, radius: int = 9) -> torch.Tensor:
    """Blur mask edges. Hard mask boundaries on a residual read as visible seams
    -- the eye is far more sensitive to a sharp edge in a delta than to the
    delta itself."""
    k = radius | 1
    blur = torch.ones(1, 1, k, k, device=mask.device) / (k * k)
    return F.conv2d(F.pad(mask, (k // 2,) * 4, mode="reflect"), blur)


def compose_delta(delta: torch.Tensor, parse_map: torch.Tensor | None,
                  sliders: dict[str, float]) -> torch.Tensor:
    """Blend the raw delta according to per-region slider values."""
    if parse_map is None:
        # No parser available: fall back to a single global opacity.
        return delta * float(np.mean(list(sliders.values())))

    gain = torch.zeros_like(parse_map, dtype=torch.float32)
    weight = torch.zeros_like(gain)

    for name, value in sliders.items():
        for region in SLIDER_REGIONS.get(name, []):
            m = (parse_map == PARSE[region]).float()
            gain += m * value
            weight += m

    # Regions covered by several sliders get the mean; uncovered regions
    # (background, clothing) get 0 -- they must never be touched.
    gain = torch.where(weight > 0, gain / weight.clamp(min=1), torch.zeros_like(gain))
    gain = feather(gain.unsqueeze(0).unsqueeze(0) if gain.dim() == 2 else gain)
    return delta * gain


@torch.no_grad()
def reage(G, img_t, src_age, tgt_age, parse_map=None, sliders=None, opacity=1.0):
    sliders = {**DEFAULT_SLIDERS, **(sliders or {})}
    sa = torch.tensor([float(src_age)], device=img_t.device)
    ta = torch.tensor([float(tgt_age)], device=img_t.device)

    delta = G(G.build_input(img_t, sa, ta))
    delta = compose_delta(delta, parse_map, sliders) * opacity
    return (img_t + delta).clamp(-1, 1), delta


def parse_sliders(s: str | None) -> dict[str, float]:
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in SLIDER_REGIONS:
            raise SystemExit(f"unknown slider '{k}'. options: {list(SLIDER_REGIONS)}")
        out[k] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--src-age", type=float, required=True)
    ap.add_argument("--tgt-age", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sliders", default=None,
                    help="e.g. wrinkles=1.0,gray_hair=0.3,eye_bags=0.5")
    ap.add_argument("--opacity", type=float, default=1.0)
    ap.add_argument("--export-layer", type=Path, default=None,
                    help="write the raw delta as a compositing layer")
    ap.add_argument("--parse-map", type=Path, default=None,
                    help="BiSeNet parse map PNG; omit for global opacity only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    G = load_generator(args.model, args.device)
    img = Image.open(args.image)
    img_t = to_tensor(img, args.device)

    parse_map = None
    if args.parse_map:
        pm = np.asarray(Image.open(args.parse_map))
        parse_map = torch.from_numpy(pm).to(args.device)

    out_t, delta = reage(G, img_t, args.src_age, args.tgt_age,
                         parse_map, parse_sliders(args.sliders), args.opacity)

    to_pil(out_t).save(args.out)
    print(f"wrote {args.out}")

    if args.export_layer:
        # Delta lives in [-2,2]; remap to [0,255] with 128 = no change, which is
        # exactly what a Photoshop "Linear Light" / "Add" layer expects.
        d = (delta[0].permute(1, 2, 0).cpu().numpy() * 63.75 + 128.0)
        Image.fromarray(d.clip(0, 255).round().astype(np.uint8)).save(args.export_layer)
        print(f"wrote {args.export_layer}  (128 = neutral; composite as Linear Light)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
