"""Fetch the QC'd dataset from HuggingFace and point pairs.jsonl at it.

THE PATH TRAP
-------------
pairs.jsonl stores absolute paths, written by whichever pod generated the data
(/workspace/data/raw/id_000123/...). A different pod may unpack anywhere, and a
stale path does not fail at startup -- it fails in the dataloader, several
minutes into a paid GPU run, once per worker, in a stack trace that looks like
a code bug. So every path is rewritten against the real extract location and
checked on disk BEFORE training is allowed to start.

    python scripts/prepare_dataset.py --root /workspace/data
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Abdelkarim40/agecraft-data")
    ap.add_argument("--file", default="agecraft_dataset.tgz")
    ap.add_argument("--root", type=Path, default=Path("/workspace/data"))
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep the .tgz; by default it is deleted after extract")
    args = ap.parse_args()

    root: Path = args.root
    pairs_file = root / "processed" / "pairs.jsonl"

    if pairs_file.exists():
        print(f"{pairs_file} already present -- skipping download")
    else:
        from huggingface_hub import hf_hub_download

        print(f"downloading {args.file} from {args.dataset}")
        tgz = hf_hub_download(repo_id=args.dataset, filename=args.file,
                              repo_type="dataset",
                              token=os.environ.get("HF_TOKEN"))
        print(f"extracting -> {root}")
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tar:
            try:
                tar.extractall(root, filter="data")
            except TypeError:  # filter= is 3.12+
                tar.extractall(root)
        if not args.keep_archive:
            # archive + hub cache + extracted tree together outgrow a modest
            # container disk, and the archive is the copy we no longer need
            Path(tgz).unlink(missing_ok=True)

    if not pairs_file.exists():
        print(f"ERROR: {pairs_file} missing after extract")
        return 1

    raw = root / "raw"
    pairs = [json.loads(l) for l in pairs_file.read_text().splitlines() if l.strip()]

    usable, missing = [], 0
    for p in pairs:
        for key in ("src_path", "tgt_path"):
            old = Path(p[key])
            p[key] = str(raw / old.parent.name / old.name)
        if Path(p["src_path"]).exists() and Path(p["tgt_path"]).exists():
            usable.append(p)
        else:
            missing += 1

    if not usable:
        print(f"ERROR: none of the {len(pairs)} pairs resolve under {raw}")
        return 1

    pairs_file.write_text("\n".join(json.dumps(p) for p in usable) + "\n")

    identities = {p["identity"] for p in usable}
    print(f"{len(usable)} pairs across {len(identities)} identities, rooted at {raw}")
    if missing:
        print(f"WARNING: dropped {missing} pairs with missing image files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
