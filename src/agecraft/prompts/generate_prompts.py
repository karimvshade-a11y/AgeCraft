"""Stage 1 — build the identity prompt matrix with a local LLM (Qwen/Ollama).

This is the ONE place an LLM belongs in this pipeline, and it runs exactly
once. The output (prompts.jsonl) is committed and versioned. Ollama never
ships with the product and never touches training or inference.

Why bother instead of hand-writing prompts?
  Every open re-aging model inherits FFHQ's demographic skew and is visibly
  worse on darker skin tones and non-Western faces. We're building the dataset
  from scratch, so we get to fix that on purpose. "Works on everyone" is a real
  selling point against FaceApp -- but only if the identity matrix is balanced
  by construction rather than by hope.

Strategy: we do NOT ask the LLM to freestyle. We enumerate a balanced
factorial grid ourselves (that part must be deterministic and auditable), and
use Qwen only to turn each cell into varied, natural photographic phrasing.
The LLM is a phrasing engine, not a decision-maker.

Usage:
    ollama pull qwen2.5:14b-instruct
    python -m agecraft.prompts.generate_prompts --out data/prompts.jsonl -n 4000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

# --- The balanced grid. Deterministic, auditable, ours. ----------------------
# Keep these lists roughly equal in length so the product is near-uniform.

ANCESTRY = [
    "West African", "East African", "North African",
    "South Asian", "East Asian", "Southeast Asian", "Central Asian",
    "Northern European", "Southern European", "Eastern European",
    "Middle Eastern", "Indigenous American", "Pacific Islander",
    "Afro-Caribbean", "Mestizo Latin American",
]

SKIN_TONE = [  # Fitzpatrick-ish, stated explicitly so FLUX doesn't default pale
    "very fair skin", "fair skin", "light olive skin", "olive skin",
    "tan brown skin", "deep brown skin", "very deep brown skin",
]

SEX = ["man", "woman"]

LIGHTING = [
    "soft window light", "harsh direct midday sun", "overcast diffuse light",
    "warm tungsten indoor light", "cool fluorescent office light",
    "golden hour backlight", "on-camera flash", "studio softbox key light",
    "dim ambient evening light", "mixed daylight and shade",
]

POSE = [
    "straight-on to camera", "three-quarter turn", "slight head tilt",
    "chin slightly raised", "chin slightly lowered", "profile-leaning angle",
]

EXPRESSION = [
    "neutral relaxed expression", "faint closed-mouth smile",
    "broad smile showing teeth", "serious focused expression",
    "eyebrows slightly raised", "tired soft expression",
]

FEATURES = [
    "clean shaven", "short stubble", "full beard", "moustache only",
    "wearing thin wire-frame glasses", "wearing thick-rimmed glasses",
    "wearing small stud earrings", "wearing a headscarf",
    "shaved head", "long hair tied back", "short cropped hair",
    "curly voluminous hair", "straight shoulder-length hair",
    "visible facial mole", "freckled complexion", "no distinguishing accessories",
]

BUILD = ["slender build", "average build", "fuller face", "heavyset build"]

SYSTEM = """You write terse photographic prompts for a text-to-image model.
Given attribute tags, return ONE photorealistic portrait prompt.

Hard rules:
- Describe an ADULT. Never imply a child, teen, or minor.
- Do NOT mention age, years, young, old, elderly, youthful, or wrinkles.
  Age is injected separately downstream; if you mention it you corrupt the data.
- Do NOT name any real or public person.
- Include: framing (head-and-shoulders portrait), the given attributes,
  lighting, camera realism cues (85mm lens, shallow depth of field).
- 40 words maximum. No preamble, no quotes, no markdown.

Respond with JSON only: {"prompt": "..."}"""


@dataclass
class IdentitySpec:
    id: str
    ancestry: str
    skin_tone: str
    sex: str
    lighting: str
    pose: str
    expression: str
    features: str
    build: str
    prompt: str = ""
    seed: int = 0


def build_grid(n: int, rng: random.Random) -> list[IdentitySpec]:
    """Latin-hypercube-ish sampling over the attribute space.

    We cycle each axis independently rather than taking a random product, which
    guarantees near-exact marginal balance on every axis even when n is far
    smaller than the full factorial (which would be ~10^7 cells).
    """
    axes = {
        "ancestry": ANCESTRY, "skin_tone": SKIN_TONE, "sex": SEX,
        "lighting": LIGHTING, "pose": POSE, "expression": EXPRESSION,
        "features": FEATURES, "build": BUILD,
    }
    cycles = {}
    for name, values in axes.items():
        pool = []
        while len(pool) < n:
            chunk = list(values)
            rng.shuffle(chunk)
            pool.extend(chunk)
        cycles[name] = pool[:n]

    specs = []
    for i in range(n):
        specs.append(
            IdentitySpec(
                id=f"id_{i:06d}",
                seed=rng.randrange(0, 2**31 - 1),
                **{name: cycles[name][i] for name in axes},
            )
        )
    return specs


def phrase_with_qwen(spec: IdentitySpec, model: str, timeout: int = 120) -> str:
    tags = (
        f"{spec.ancestry} {spec.sex}, {spec.skin_tone}, {spec.build}, "
        f"{spec.features}, {spec.expression}, {spec.pose}, {spec.lighting}"
    )
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.8, "seed": spec.seed},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Attributes: {tags}"},
            ],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    return json.loads(content)["prompt"].strip()


def fallback_prompt(spec: IdentitySpec) -> str:
    """Deterministic template if Ollama is unavailable. The pipeline must never
    hard-depend on the LLM -- it's a convenience, not infrastructure."""
    return (
        f"head-and-shoulders photorealistic portrait of a {spec.ancestry} "
        f"{spec.sex} with {spec.skin_tone}, {spec.build}, {spec.features}, "
        f"{spec.expression}, {spec.pose}, lit by {spec.lighting}, "
        f"85mm lens, shallow depth of field, sharp focus on eyes"
    )


# Age-related tokens that must not appear: age is injected downstream by
# generate_pairs.py, and a prompt that mentions it corrupts the age signal.
#
# NOTE: these MUST be matched on word boundaries. Naive substring matching
# false-positives on "aver(age) build" and "g(old)en hour backlight" -- both of
# which are in our own attribute grid. That bug silently rejects ~32% of valid
# prompts and systematically strips one entire lighting condition out of the
# LLM-phrased set, quietly destroying the balance the grid exists to guarantee.
# The trailing \w* still catches inflections: aged, older, wrinkles, teenager.
BANNED = ("child", "children", "kid", "teen", "young", "old", "elderly",
          "youthful", "wrinkle", "age", "year", "adolescent", "baby", "infant",
          "toddler", "boy", "girl", "senior", "geriatric")

_BANNED_RE = re.compile(r"\b(" + "|".join(BANNED) + r")\w*\b", re.IGNORECASE)


def banned_hits(prompt: str) -> list[str]:
    return _BANNED_RE.findall(prompt)


def validate(prompt: str) -> bool:
    if banned_hits(prompt):
        return False
    return 10 <= len(prompt.split()) <= 60


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("-n", type=int, default=4000, help="number of identities")
    ap.add_argument("--model", default="qwen2.5:14b-instruct")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-llm", action="store_true", help="use templates only")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    specs = build_grid(args.n, rng)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_llm = n_fallback = 0

    with args.out.open("w") as f:
        for i, spec in enumerate(specs):
            prompt = ""
            if not args.no_llm:
                try:
                    prompt = phrase_with_qwen(spec, args.model)
                    if not validate(prompt):
                        prompt = ""  # LLM leaked a banned token; drop to template
                except Exception as e:
                    if i == 0:
                        print(f"[warn] Ollama unreachable ({e}); using templates",
                              file=sys.stderr)
                    prompt = ""

            if prompt:
                n_llm += 1
            else:
                prompt = fallback_prompt(spec)
                n_fallback += 1

            spec.prompt = prompt
            f.write(json.dumps(asdict(spec)) + "\n")

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{args.n}", file=sys.stderr)

    print(f"wrote {args.out}  (llm={n_llm}, template={n_fallback})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
