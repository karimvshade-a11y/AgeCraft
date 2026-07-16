"""Paired dataset for delta-net training.

Critical detail: src and tgt come from the same seed-locked latent, so they are
already pixel-aligned. Any augmentation MUST be applied identically to both, or
the model learns to predict a delta that includes a spatial shift -- which
looks exactly like ghosting in the output.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def to_tensor(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # -> [-1, 1]
    return torch.from_numpy(a).permute(2, 0, 1)


class AgePairDataset(Dataset):
    def __init__(self, pairs_file: Path, crop: int = 512, train: bool = True):
        self.pairs = [json.loads(l) for l in Path(pairs_file).read_text().splitlines()
                      if l.strip()]
        self.crop = crop
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        p = self.pairs[i]
        src = Image.open(p["src_path"]).convert("RGB")
        tgt = Image.open(p["tgt_path"]).convert("RGB")
        assert src.size == tgt.size, f"size mismatch in {p['identity']}"

        if self.train:
            w, h = src.size
            c = min(self.crop, w, h)
            x = random.randint(0, w - c)
            y = random.randint(0, h - c)
            box = (x, y, x + c, y + c)
            src, tgt = src.crop(box), tgt.crop(box)  # SAME box -- non-negotiable

            if random.random() < 0.5:
                src = src.transpose(Image.FLIP_LEFT_RIGHT)
                tgt = tgt.transpose(Image.FLIP_LEFT_RIGHT)  # SAME flip
        else:
            src = src.resize((self.crop, self.crop), Image.LANCZOS)
            tgt = tgt.resize((self.crop, self.crop), Image.LANCZOS)

        return {
            "src": to_tensor(src),
            "tgt": to_tensor(tgt),
            "src_age": torch.tensor(float(p["src_age"])),
            "tgt_age": torch.tensor(float(p["tgt_age"])),
        }
