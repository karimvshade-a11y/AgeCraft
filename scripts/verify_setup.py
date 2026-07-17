"""Verify the repo is intact before you spend money on a GPU.

Run this FIRST, every time you move the code somewhere new:

    python scripts/verify_setup.py

Catches the failure that actually happens: files downloaded individually from
a chat window land flat in one folder, the package structure is gone, and you
find out 20 minutes into a paid pod.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    ".gitignore",
    "PROVENANCE.md",
    "README.md",
    "pyproject.toml",
    "params.yaml",
    "dvc.yaml",
    "configs/base.yaml",
    "configs/gtx1650.yaml",
    "scripts/runpod_setup.sh",
    "scripts/run_unattended.sh",
    "scripts/upload_results.py",
    "scripts/analyze_thresholds.py",
    "notebooks/kaggle_generate.py",
    "tests/test_smoke.py",
    "src/agecraft/__init__.py",
    "src/agecraft/train.py",
    "src/agecraft/inference.py",
    "src/agecraft/export_onnx.py",
    "src/agecraft/models/__init__.py",
    "src/agecraft/models/fran.py",
    "src/agecraft/models/discriminator.py",
    "src/agecraft/data/__init__.py",
    "src/agecraft/data/dataset.py",
    "src/agecraft/data/generate_pairs.py",
    "src/agecraft/data/filter_pairs.py",
    "src/agecraft/prompts/__init__.py",
    "src/agecraft/prompts/generate_prompts.py",
]


def main() -> int:
    print(f"checking {ROOT}\n")

    problems = 0
    missing = [f for f in REQUIRED if not (ROOT / f).exists()]

    # The specific failure mode: right filenames, wrong (flat) location.
    flat = []
    for f in missing:
        name = Path(f).name
        if (ROOT / name).exists():
            flat.append((name, f))

    if flat:
        print("PROBLEM: files are FLAT in the repo root, not in their packages.")
        print("This happens when files are downloaded one-by-one from chat.")
        print("Python cannot import a package that isn't laid out on disk.\n")
        for name, should in flat:
            print(f"  {name:28s} should be at  {should}")
        print("\nFIX: re-download the repo ZIP and extract it, or move the files.")
        problems += 1

    truly_missing = [f for f in missing if not (ROOT / Path(f).name).exists()]
    if truly_missing:
        print("\nMISSING ENTIRELY (not anywhere in the repo):")
        for f in truly_missing:
            note = ""
            if f == ".gitignore":
                note = "  <-- DANGEROUS: you'll commit 6GB of PNGs to GitHub"
            elif f in ("src/agecraft/data/dataset.py",
                       "src/agecraft/models/discriminator.py"):
                note = "  <-- train.py cannot run without this"
            elif f == "pyproject.toml":
                note = "  <-- `pip install -e .` will fail"
            elif f == "configs/base.yaml":
                note = "  <-- the config for the full-quality model"
            print(f"  ✗ {f}{note}")
        problems += 1

    # --- STALE DUPLICATES -------------------------------------------------
    # Files that belong in a package but ALSO sit at the root. Python may
    # import the wrong one, and if they came from an older download they're
    # silently out of date. This is what let a broken repo reach GitHub.
    pkg_names = {Path(f).name for f in REQUIRED if f.startswith("src/")}
    dupes = [n for n in pkg_names
             if (ROOT / n).exists() and n != "__init__.py"]
    if dupes:
        print("\nSTALE DUPLICATES at repo root (these shadow the real modules):")
        for n in sorted(dupes):
            print(f"  ⚠ {n}")
        print("\nThese are leftovers from an older flat download. They are")
        print("probably OUT OF DATE. Delete them:")
        print("  " + " ".join(f"del {n}" for n in sorted(dupes)) + "   (Windows)")
        print("  rm " + " ".join(sorted(dupes)) + "   (Linux)")
        problems += 1

    # --- GITIGNORE ANCHORING ----------------------------------------------
    # `data/` (unanchored) matches src/agecraft/data/ at any depth and will
    # silently drop a quarter of the package on push. Must be `/data/`.
    gi = ROOT / ".gitignore"
    if gi.exists():
        bad = []
        for line in gi.read_text().splitlines():
            s = line.strip()
            if s in ("data/", "weights/", "dist/", "data", "weights", "dist"):
                bad.append(s)
        if bad:
            print("\nGITIGNORE BUG: unanchored patterns.")
            for b in bad:
                print(f"  ⚠ '{b}' also matches src/agecraft/{b.rstrip('/')}/ "
                      f"-- use '/{b.rstrip('/')}/' instead")
            print("\nThis silently excludes package code from git. Fix before pushing.")
            problems += 1

    if problems:
        print(f"\n{len(REQUIRED) - len(missing)}/{len(REQUIRED)} files present. NOT READY.")
        return 1

    print(f"all {len(REQUIRED)} files present ✓")

    # Structure is right -- now prove it actually imports.
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from agecraft.data.dataset import AgePairDataset  # noqa: F401
        from agecraft.data.filter_pairs import spearman  # noqa: F401
        from agecraft.data.generate_pairs import AGE_PHRASES  # noqa: F401
        from agecraft.inference import compose_delta  # noqa: F401
        from agecraft.models.discriminator import PatchDiscriminator  # noqa: F401
        from agecraft.models.fran import FRANGenerator  # noqa: F401
        from agecraft.prompts.generate_prompts import build_grid  # noqa: F401
    except ImportError as e:
        print(f"\nIMPORT FAILED: {e}")
        print("Files are in place but something won't load. Install deps:")
        print("  pip install -e \".[train]\"")
        return 1

    print("all modules import ✓")
    print("\nREADY. Next:  pytest -q     (then deploy the pod)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
