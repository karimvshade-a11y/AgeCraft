"""PatchGAN discriminator (Isola et al., pix2pix).

Judges NxN patches rather than the whole image. For re-aging this is exactly
what we want: the question is "does this skin patch look like it belongs to a
70-year-old", not "is this a real photo". Patch-level judgement is also what
pushes the model to synthesise pore/wrinkle texture instead of the blurry
average that L1 alone converges to.

Conditioned on the target age so the critic can call out under- and
over-aging, not just realism.

Training-time only. Never shipped, never exported to ONNX.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .fran import AGE_NORM


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 4, base: int = 64, n_layers: int = 3):
        # in_ch = 3 (RGB) + 1 (target age channel)
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, base, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        ch = base
        for i in range(1, n_layers):
            nxt = min(base * (2**i), 512)
            layers += [
                nn.Conv2d(ch, nxt, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(nxt),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = nxt

        nxt = min(ch * 2, 512)
        layers += [
            nn.Conv2d(ch, nxt, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nxt),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nxt, 1, 4, stride=1, padding=1),  # -> (B,1,h,w) logits
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, img: torch.Tensor, tgt_age: torch.Tensor) -> torch.Tensor:
        b, _, h, w = img.shape
        age = tgt_age.view(b, 1, 1, 1).expand(b, 1, h, w) / AGE_NORM
        return self.model(torch.cat([img, age.to(img.dtype)], dim=1))
