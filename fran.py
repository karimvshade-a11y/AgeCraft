"""FRAN-style re-aging network.

Architecture follows the public description of Disney Research's
"Production-Ready Face Re-Aging for Visual Effects" (Zoss et al., SIGGRAPH Asia
2022): a U-Net that consumes an RGB image plus two constant age channels and
predicts a *residual* (delta) rather than a new image.

    output = clamp(input + delta, -1, 1)

Why the residual matters (this is the whole product thesis):
  * identity / background / earrings / glasses survive by construction
  * the delta is a first-class artifact -> export as a compositing layer
  * the delta is spatially localisable -> per-region sliders for free
  * the delta is temporally stable -> video without a video model

The architecture is a published description and is not itself encumbered.
The *weights* are what carry licence risk, which is why we train our own.
See PROVENANCE.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ages are normalised to [0, 1] by this constant before being painted into
# constant input channels. Keep in sync with inference + ONNX export.
AGE_NORM = 100.0


class BlurPool2d(nn.Module):
    """Anti-aliased downsampling (Zhang, ICML 2019).

    Plain max-pooling aliases, and aliasing in a *residual* model shows up as
    shimmering wrinkles when you run it over video frames. Blur-then-pool is
    the cheapest fix and is what FRAN specifies.
    """

    def __init__(self, channels: int, filt_size: int = 3, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.channels = channels

        if filt_size == 3:
            a = torch.tensor([1.0, 2.0, 1.0])
        elif filt_size == 5:
            a = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        else:
            raise ValueError(f"unsupported filt_size={filt_size}")

        filt = a[:, None] * a[None, :]
        filt = filt / filt.sum()
        # One filter per channel, applied depthwise.
        self.register_buffer("filt", filt[None, None].repeat(channels, 1, 1, 1))
        self.pad = nn.ReflectionPad2d(filt_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(x, kernel_size=2, stride=1)  # max first, stride 1
        x = self.pad(x)
        return F.conv2d(x, self.filt, stride=self.stride, groups=self.channels)


class ConvBlock(nn.Module):
    """(3x3 conv -> BN -> ReLU) x 2, the standard U-Net double conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FRANGenerator(nn.Module):
    """U-Net predicting a 3-channel aging delta.

    Input  : (B, 5, H, W) = RGB in [-1, 1] + src_age chan + tgt_age chan
    Output : (B, 3, H, W) delta in roughly [-2, 2] (tanh-scaled)

    The network is fully convolutional, so it trains on 512px crops and runs at
    whatever resolution you hand it at inference (subject to VRAM).
    """

    def __init__(self, base: int = 64, depth: int = 4, delta_scale: float = 1.0):
        super().__init__()
        self.depth = depth
        self.delta_scale = delta_scale

        chans = [base * (2**i) for i in range(depth)]  # 64,128,256,512

        # ---- encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = 5  # RGB + 2 age channels
        for c in chans:
            self.downs.append(ConvBlock(in_ch, c))
            self.pools.append(BlurPool2d(c))
            in_ch = c

        # ---- bottleneck
        self.bottleneck = ConvBlock(chans[-1], chans[-1] * 2)

        # ---- decoder (bilinear upsample + conv, avoids checkerboard artifacts
        #      that transposed conv bakes into a residual)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        in_ch = chans[-1] * 2
        for c in reversed(chans):
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_ch, c, 3, padding=1),
                )
            )
            self.up_convs.append(ConvBlock(c * 2, c))  # *2 for skip concat
            in_ch = c

        self.head = nn.Conv2d(chans[0], 3, 1)

        # Zero-init the head so the model starts as a perfect identity function
        # (delta == 0). Training then only has to learn the *difference*, which
        # is dramatically more stable than learning to reconstruct the face.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def build_input(
        img: torch.Tensor, src_age: torch.Tensor, tgt_age: torch.Tensor
    ) -> torch.Tensor:
        """Paint scalar ages into constant channels and concat with RGB.

        img      : (B, 3, H, W) in [-1, 1]
        src_age  : (B,) or (B, 1) in years
        tgt_age  : (B,) or (B, 1) in years
        """
        b, _, h, w = img.shape
        src = src_age.view(b, 1, 1, 1).expand(b, 1, h, w) / AGE_NORM
        tgt = tgt_age.view(b, 1, 1, 1).expand(b, 1, h, w) / AGE_NORM
        return torch.cat([img, src.to(img.dtype), tgt.to(img.dtype)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is the 5-channel tensor from build_input(). Returns the delta."""
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            # Guard against odd input dims losing a pixel on the way down.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = conv(torch.cat([x, skip], dim=1))

        return torch.tanh(self.head(x)) * self.delta_scale

    def reage(
        self, img: torch.Tensor, src_age: torch.Tensor, tgt_age: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: returns (re-aged image, delta)."""
        delta = self.forward(self.build_input(img, src_age, tgt_age))
        return (img + delta).clamp(-1.0, 1.0), delta
