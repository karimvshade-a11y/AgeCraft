"""Push one file to a HuggingFace repo.

Called repeatedly by run_training.sh, which is the whole point: a checkpoint
that only leaves the machine once, at the end, is a bet rather than a backup.

    python scripts/hf_push.py --file weights/last.pt --path-in-repo last.pt \
        --repo yourname/agecraft-weights
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--path-in-repo", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-type", default="model")
    ap.add_argument("--public", action="store_true", help="default is PRIVATE")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type=args.repo_type,
                    private=not args.public, exist_ok=True)
    api.upload_file(path_or_fileobj=args.file, path_in_repo=args.path_in_repo,
                    repo_id=args.repo, repo_type=args.repo_type)
    print(f"pushed {args.file} -> {args.repo}/{args.path_in_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
