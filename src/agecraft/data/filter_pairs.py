"""Stage 3 — ruthless QC. This is where the yield loss happens, by design.

Two independent gates, both numeric, both logged:

  1. IDENTITY GATE. Every frame in a sequence must embed close to the sequence
     centroid. If FLUX drifted to a different person when we changed the age
     phrase, the whole sequence dies. No partial credit -- a sequence with one
     bad frame teaches the model that aging changes bone structure.

  2. AGE GATE. An age estimator must agree, monotonically, that the frames are
     ordered by age. If FLUX ignored "80 year old" and gave us a 45-year-old,
     that pair teaches the model to under-age. Monotonicity is a much stronger
     signal than absolute accuracy -- estimators are biased but consistently so.

Deliberately NOT using a VLM here. "Does this look like the same person" is a
question a 20MB embedding model answers better, faster, and deterministically
than a 14B LLM, and it returns a *number* we can put in the audit log. An LLM's
vibes are not a defensible provenance record.

Licence note: the QC models below are used offline to SELECT data. They are not
distributed and their weights are not embedded in our model. That is a weaker
exposure than shipping them -- but "weaker" is not "none", and I'm not a
lawyer. PROVENANCE.md tracks the choice; get it reviewed before you sell.
The interface is pluggable specifically so you can swap in whatever your
counsel is comfortable with.

Usage:
    python -m agecraft.data.filter_pairs --raw data/raw --out data/processed
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class IdentityEmbedder:
    """FaceNet (facenet-pytorch, MIT code). Swap freely -- interface is
    `embed(PIL.Image) -> unit-norm np.ndarray`."""

    def __init__(self, device: str = "cuda"):
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.device = device
        self.mtcnn = MTCNN(image_size=160, margin=20, device=device, post_process=True)
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    @torch.no_grad()
    def embed(self, img: Image.Image) -> np.ndarray | None:
        face = self.mtcnn(img)
        if face is None:
            return None  # no detectable face -> sequence is garbage anyway
        v = self.net(face.unsqueeze(0).to(self.device))[0].cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-8)


class AgeEstimator:
    """ViT age classifier fine-tuned on FairFace (CC BY 4.0 dataset).

    Returns an expected age from the bucket distribution -- softer and more
    monotone-friendly than argmax.
    """

    BUCKET_MIDPOINTS = {
        "0-2": 1, "3-9": 6, "10-19": 15, "20-29": 25, "30-39": 35,
        "40-49": 45, "50-59": 55, "60-69": 65, "more than 70": 78,
    }

    def __init__(self, device: str = "cuda",
                 model_id: str = "nateraw/vit-age-classifier"):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.net = AutoModelForImageClassification.from_pretrained(model_id)
        self.net.eval().to(device)
        self.mids = np.array(
            [self.BUCKET_MIDPOINTS.get(self.net.config.id2label[i], 40.0)
             for i in range(self.net.config.num_labels)]
        )

    @torch.no_grad()
    def estimate(self, img: Image.Image) -> float:
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        probs = self.net(**inputs).logits.softmax(-1)[0].cpu().numpy()
        return float((probs * self.mids).sum())


def check_sequence(rec: dict, ident: IdentityEmbedder, ager: AgeEstimator,
                   id_thresh: float, spearman_min: float) -> dict:
    ages = sorted(int(a) for a in rec["frames"])
    imgs = {a: Image.open(rec["frames"][str(a)]).convert("RGB") for a in ages}

    # --- gate 1: identity
    embs = {}
    for a, im in imgs.items():
        e = ident.embed(im)
        if e is None:
            return {"pass": False, "reason": f"no face detected at age {a}"}
        embs[a] = e

    centroid = np.mean(list(embs.values()), axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-8
    sims = {a: float(np.dot(embs[a], centroid)) for a in ages}
    worst = min(sims.values())

    # Also check the hardest pair directly (youngest vs oldest); centroid
    # similarity can hide a slow drift across the sequence.
    pair_sim = float(np.dot(embs[ages[0]], embs[ages[-1]]))

    if worst < id_thresh or pair_sim < id_thresh:
        return {"pass": False, "reason": "identity drift",
                "worst_sim": worst, "endpoint_sim": pair_sim}

    # --- gate 2: age monotonicity
    est = {a: ager.estimate(im) for a, im in imgs.items()}
    rho = spearman(np.array(ages, dtype=float),
                   np.array([est[a] for a in ages]))
    if rho < spearman_min:
        return {"pass": False, "reason": "age not monotonic",
                "spearman": rho, "estimated": est}

    span = est[ages[-1]] - est[ages[0]]
    if span < 25.0:  # asked for 20->80 and got less than 25 years of change
        return {"pass": False, "reason": "insufficient age span",
                "span": span, "estimated": est}

    return {"pass": True, "worst_sim": worst, "endpoint_sim": pair_sim,
            "spearman": rho, "span": span, "estimated": est}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without pulling in scipy."""
    def rank(v):
        order = v.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum()) + 1e-8
    return float((rx * ry).sum() / denom)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--id-thresh", type=float, default=0.55,
                    help="cosine sim floor; FaceNet/vggface2 same-person is "
                         "typically >0.6. Raise it if the model learns to "
                         "change bone structure.")
    ap.add_argument("--spearman-min", type=float, default=0.85)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    recs = [json.loads(l) for l in (args.raw / "manifest.jsonl").read_text().splitlines()
            if l.strip()]

    ident = IdentityEmbedder(args.device)
    ager = AgeEstimator(args.device)

    args.out.mkdir(parents=True, exist_ok=True)
    kept, rejected = [], []

    for i, rec in enumerate(recs):
        result = check_sequence(rec, ident, ager, args.id_thresh, args.spearman_min)
        entry = {"id": rec["id"], **result}
        if result["pass"]:
            kept.append({**rec, "qc": result})
        else:
            rejected.append(entry)
        if (i + 1) % 50 == 0:
            rate = len(kept) / (i + 1)
            print(f"  {i+1}/{len(recs)}  keep rate {rate:.1%}")

    # Expand surviving sequences into ordered training pairs (both directions --
    # the model must learn de-aging as well as aging).
    pairs = []
    for rec in kept:
        ages = sorted(int(a) for a in rec["frames"])
        for a, b in itertools.permutations(ages, 2):
            pairs.append({
                "identity": rec["id"],
                "src_path": rec["frames"][str(a)],
                "tgt_path": rec["frames"][str(b)],
                "src_age": a,
                "tgt_age": b,
            })

    (args.out / "pairs.jsonl").write_text(
        "\n".join(json.dumps(p) for p in pairs) + "\n")
    (args.out / "kept.jsonl").write_text(
        "\n".join(json.dumps(k) for k in kept) + "\n")
    (args.out / "rejected.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rejected) + "\n")

    # Balance report -- the whole point of the balanced grid is wasted if the
    # filter silently rejects one demographic harder than another. THIS IS THE
    # NUMBER TO WATCH. Uneven keep rates mean a biased model.
    from collections import Counter
    keep_by = Counter(k["ancestry"] for k in kept)
    all_by = Counter(r.get("ancestry", "?") for r in recs)
    print("\nkeep rate by ancestry (watch for imbalance):")
    for anc in sorted(all_by):
        tot = all_by[anc]
        print(f"  {anc:28s} {keep_by.get(anc,0):4d}/{tot:4d}  "
              f"{keep_by.get(anc,0)/max(tot,1):.1%}")

    print(f"\nkept {len(kept)}/{len(recs)} sequences -> {len(pairs)} training pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
