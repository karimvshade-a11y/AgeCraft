"""Push filtered results to a private HuggingFace dataset.

The point: once this runs, the pod is disposable. It can be terminated, run out
of credit, or catch fire, and your dataset survives. HF private datasets are
free and effectively unlimited for this size.

Uploads ONLY sequences that passed QC -- typically ~72%, so ~1.5GB rather than
the full ~7GB of raw output. The rejects aren't worth the bandwidth.

    python scripts/upload_results.py --dataset yourname/agecraft-data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. yourname/agecraft-data")
    ap.add_argument("--processed", type=Path, default=Path("/workspace/data/processed"))
    ap.add_argument("--raw", type=Path, default=Path("/workspace/data/raw"))
    ap.add_argument("--public", action="store_true", help="default is PRIVATE")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        return 1

    kept_file = args.processed / "kept.jsonl"
    if not kept_file.exists():
        print(f"ERROR: {kept_file} missing -- did the filter run?")
        return 1

    kept = [json.loads(l) for l in kept_file.read_text().splitlines() if l.strip()]
    if not kept:
        print("ERROR: zero sequences passed QC. Nothing to upload.")
        return 1
    print(f"{len(kept)} sequences passed QC")

    # Tar only the survivors. Rejects are not worth the bandwidth.
    tgz = Path("/workspace/agecraft_dataset.tgz")
    print(f"packing -> {tgz}")
    with tarfile.open(tgz, "w:gz") as tar:
        for rec in kept:
            for p in rec["frames"].values():
                p = Path(p)
                if p.exists():
                    tar.add(p, arcname=f"raw/{p.parent.name}/{p.name}")
        for name in ("pairs.jsonl", "kept.jsonl", "rejected.jsonl"):
            f = args.processed / name
            if f.exists():
                tar.add(f, arcname=f"processed/{name}")
        pj = Path("data/prompts.jsonl")
        if pj.exists():
            tar.add(pj, arcname="prompts.jsonl")

    size_gb = tgz.stat().st_size / 1e9
    print(f"packed {size_gb:.2f} GB")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.create_repo(args.dataset, repo_type="dataset",
                        private=not args.public, exist_ok=True)
    except Exception as e:
        print(f"create_repo: {e}")

    print(f"uploading to {args.dataset} ({'public' if args.public else 'PRIVATE'})")
    api.upload_file(
        path_or_fileobj=str(tgz),
        path_in_repo="agecraft_dataset.tgz",
        repo_id=args.dataset,
        repo_type="dataset",
    )

    # A README so future-you knows what this is and where it came from.
    readme = f"""---
license: apache-2.0
tags: [synthetic, face-aging]
---
# AgeCraft synthetic re-aging dataset

{len(kept)} identity sequences that passed QC, {size_gb:.2f} GB.

Generated with FLUX.1-schnell (Apache 2.0) from prompts authored in the
AgeCraft repo. Identity held by seed+latent locking -- no ID adapter, no
InsightFace, no FFHQ, no StyleGAN. See PROVENANCE.md in the source repo.

QC: adjacent-frame identity chain >= 0.55 (FaceNet cosine), age-monotonic.

Every pixel derives from an Apache-2.0 model and our own prompts.
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=args.dataset,
        repo_type="dataset",
    )

    print(f"\nDONE -> https://huggingface.co/datasets/{args.dataset}")
    print("The pod is now disposable. Terminate it freely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
