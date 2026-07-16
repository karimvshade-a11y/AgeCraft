"""Smoke tests. Run `pytest -q` before you spend money on a GPU.

These are the tests I ran while building the scaffold. They live here now so
you can verify the model yourself rather than taking my word for it.

They all run on CPU in a few seconds. Nothing here needs a GPU, FLUX, or a
dataset -- this is purely "is the code sane".
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from agecraft.data.filter_pairs import spearman
from agecraft.inference import PARSE, compose_delta, feather, parse_sliders
from agecraft.models.discriminator import PatchDiscriminator
from agecraft.models.fran import BlurPool2d, FRANGenerator
from agecraft.prompts.generate_prompts import (
    build_grid,
    fallback_prompt,
    validate,
)


# ---------------------------------------------------------------- model
def test_build_input_paints_age_channels():
    G = FRANGenerator(base=16)
    img = torch.randn(2, 3, 64, 64)
    x = G.build_input(img, torch.tensor([30.0, 25.0]), torch.tensor([70.0, 60.0]))
    assert x.shape == (2, 5, 64, 64)
    assert x[0, 3].mean().item() == pytest.approx(0.30)
    assert x[0, 4].mean().item() == pytest.approx(0.70)


def test_zero_init_head_is_exact_identity():
    """At init the model MUST be a perfect no-op. This is what makes training
    stable -- it only ever learns the difference from the input."""
    G = FRANGenerator(base=16)
    img = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    out, delta = G.reage(img, torch.tensor([30.0]), torch.tensor([70.0]))
    assert delta.abs().max().item() == 0.0
    assert torch.allclose(out, img)


def test_delta_is_bounded():
    G = FRANGenerator(base=16)
    torch.nn.init.normal_(G.head.weight, std=0.5)
    d = G(G.build_input(torch.randn(1, 3, 64, 64),
                        torch.tensor([20.0]), torch.tensor([80.0])))
    assert d.abs().max().item() <= 1.0 + 1e-5


@pytest.mark.parametrize("h,w", [(96, 160), (127, 193), (65, 65), (64, 64)])
def test_odd_and_nonsquare_shapes_survive_the_unet(h, w):
    """Classic U-Net bug: odd dims lose a pixel on the way down and the skip
    concat explodes. The model must run at whatever size a user hands it."""
    G = FRANGenerator(base=16)
    img = torch.randn(1, 3, h, w)
    d = G(G.build_input(img, torch.tensor([20.0]), torch.tensor([80.0])))
    assert d.shape == (1, 3, h, w)


def test_gradients_flow():
    G = FRANGenerator(base=16)
    img = torch.randn(1, 3, 64, 64)
    out, _ = G.reage(img, torch.tensor([30.0]), torch.tensor([70.0]))
    torch.nn.functional.l1_loss(out, torch.randn_like(out)).backward()
    total = sum(p.grad.abs().sum().item() for p in G.parameters()
                if p.grad is not None)
    assert total > 0


def test_blurpool_halves_spatial_dims():
    assert BlurPool2d(8)(torch.randn(1, 8, 64, 64)).shape == (1, 8, 32, 32)


def test_discriminator_outputs_patch_logits():
    logits = PatchDiscriminator()(torch.randn(2, 3, 128, 128), torch.tensor([70.0, 40.0]))
    assert logits.ndim == 4 and logits.shape[1] == 1


# ---------------------------------------------------------------- sliders
def _parse_map(h=64, w=64):
    pm = torch.zeros(h, w, dtype=torch.long)
    pm[:20, :] = PARSE["hair"]
    pm[20:50, :] = PARSE["skin"]
    pm[50:, :] = PARSE["cloth"]
    return pm


def test_background_and_cloth_are_never_touched():
    """The delta must be zero outside face regions. If this fails, backgrounds
    get modified and the whole 'identity preserved by construction' claim dies."""
    out = compose_delta(torch.ones(1, 3, 64, 64) * 0.5, _parse_map(),
                        {"wrinkles": 1.0, "gray_hair": 1.0})
    assert out[0, :, 55:, :].abs().max().item() < 1e-4


def test_sliders_isolate_regions():
    out = compose_delta(torch.ones(1, 3, 64, 64) * 0.5, _parse_map(),
                        {"wrinkles": 1.0, "gray_hair": 0.0})
    assert out[0, :, :15, :].abs().mean().item() < 0.02   # hair suppressed
    assert out[0, :, 30:45, :].abs().mean().item() > 0.4  # skin untouched


def test_slider_scaling_is_linear():
    """delta*0.5 must mean 'half as much'. The identity-cycle loss in training
    is what protects this at the model level; this checks the compositing."""
    vals = [compose_delta(torch.ones(1, 3, 64, 64) * 0.5, _parse_map(),
                          {"wrinkles": s})[0, :, 30:45, :].abs().mean().item()
            for s in [0.0, 0.25, 0.5, 0.75, 1.0]]
    diffs = np.diff(vals)
    assert np.allclose(diffs, diffs[0], atol=1e-3)


def test_no_parse_map_falls_back_to_global_opacity():
    out = compose_delta(torch.ones(1, 3, 32, 32) * 0.5, None,
                        {"wrinkles": 0.5, "gray_hair": 0.5})
    assert out.abs().mean().item() == pytest.approx(0.25, abs=1e-4)


def test_feather_softens_mask_edges():
    m = torch.zeros(1, 1, 32, 32)
    m[:, :, :16, :] = 1.0
    assert 0.0 < feather(m, 9)[0, 0, 15, 16].item() < 1.0


def test_unknown_slider_rejected():
    with pytest.raises(SystemExit):
        parse_sliders("bogus=1.0")


# ---------------------------------------------------------------- prompts
def test_banned_tokens_match_on_word_boundaries_only():
    """REGRESSION TEST. Naive substring matching false-positives on
    'aver(age) build' and 'g(old)en hour backlight' -- both of which are in our
    own attribute grid. That bug silently rejected 1297/4000 valid prompts and
    stripped an entire lighting condition out of the LLM-phrased set."""
    assert validate("head-and-shoulders portrait of an average build man lit by "
                    "golden hour backlight, 85mm lens, sharp focus on the eyes")
    assert validate("portrait of a woman with shoulder-length hair, average "
                    "build, soft window light, 85mm lens, sharp focus on eyes")


@pytest.mark.parametrize("bad", [
    "portrait of a young woman, 85mm lens, shallow depth of field, sharp focus here",
    "portrait of an elderly man with deep wrinkles, 85mm lens, shallow depth, focus",
    "portrait of a 40 year old man, 85mm lens, shallow depth of field, sharp focus",
    "portrait of a teenager, 85mm lens, shallow depth of field, sharp focus on eyes",
    "portrait of an aged face, 85mm lens, shallow depth of field, sharp focus eyes",
])
def test_age_leaking_prompts_are_rejected(bad):
    assert not validate(bad)


def test_grid_is_demographically_balanced():
    """The whole point of the grid. If this drifts, the model inherits bias."""
    from collections import Counter
    specs = build_grid(2000, random.Random(1337))
    for axis in ["ancestry", "skin_tone", "sex"]:
        c = Counter(getattr(s, axis) for s in specs)
        assert max(c.values()) / min(c.values()) < 1.05, f"{axis} imbalanced"


def test_grid_is_deterministic():
    a = build_grid(500, random.Random(1337))
    b = build_grid(500, random.Random(1337))
    assert [s.seed for s in a] == [s.seed for s in b]


def test_grid_seeds_are_unique():
    specs = build_grid(2000, random.Random(1337))
    assert len({s.seed for s in specs}) == len(specs)


def test_every_template_prompt_passes_validation():
    specs = build_grid(2000, random.Random(1337))
    assert [s for s in specs if not validate(fallback_prompt(s))] == []


# ---------------------------------------------------------------- filter
def test_spearman_matches_scipy_semantics():
    assert spearman(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 3, 4])) == pytest.approx(1.0)
    assert spearman(np.array([1.0, 2, 3, 4]), np.array([4.0, 3, 2, 1])) == pytest.approx(-1.0)


def test_spearman_flags_flat_age_estimates():
    """If FLUX ignores the age phrase and returns the same apparent age for
    every anchor, rho collapses and the sequence gets rejected."""
    ages = np.array([20.0, 35, 50, 65, 80])
    flat = np.array([45.0, 44, 46, 45, 44])
    assert spearman(ages, flat) < 0.85


# ---------------------------------------------------------------- latents
def test_flux_latent_packing_roundtrip_is_exact():
    """If this is lossy, the seed-lock is a lie and identity won't hold."""
    lh = lw = 96
    latents = torch.randn((1, 16, lh, lw),
                          generator=torch.Generator().manual_seed(42))
    packed = latents.view(1, 16, lh // 2, 2, lw // 2, 2)
    packed = packed.permute(0, 2, 4, 1, 3, 5).reshape(1, (lh // 2) * (lw // 2), 64)
    un = packed.view(1, lh // 2, lw // 2, 16, 2, 2)
    un = un.permute(0, 3, 1, 4, 2, 5).reshape(1, 16, lh, lw)
    assert torch.allclose(un, latents)


def test_generator_state_advances_hence_we_clone():
    """Documents WHY generate_pairs.py builds the latent once and clones it:
    reusing a Generator gives a different draw each call, so each age would
    drift to a different person."""
    g = torch.Generator().manual_seed(7)
    assert not torch.equal(torch.randn((1, 4), generator=g),
                           torch.randn((1, 4), generator=g))
