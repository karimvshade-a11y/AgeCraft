"""Re-analyse ALREADY-GENERATED sequences to set identity thresholds empirically.

Why this exists
---------------
The first filter used an endpoint check: cosine(age_20, age_80) >= 0.55.
On 50 real sequences that gate had a MAXIMUM of 0.540. It never passed once.
It wasn't strict, it was impossible -- FaceNet similarity between a person at
20 and the same person at 80 is genuinely ~0.3, because that is what ageing
does to a face. The gate was rejecting sequences FOR AGEING SUCCESSFULLY.

The fix is not a lower endpoint bar. Endpoints are the wrong measurement.

ADJACENT frames are the right one. Consecutive anchors are ~15 years apart,
where face recognition is reliable, and true identity drift shows up as ONE
broken link in the chain rather than a slowly sagging endpoint number. A
sequence that ages correctly has high adjacent similarity and low endpoint
similarity -- exactly the pattern the old gate punished.

This script re-embeds existing PNGs (no regeneration, no FLUX) and prints the
distributions so thresholds are set from data instead of my guesses.

    python scripts/analyze_thresholds.py --raw /workspace/data/raw
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def hist(xs, lo=-0.1, hi=1.0, bins=22, width=44):
    if not xs:
        return
    counts = [0] * bins
    for x in xs:
        i = int((x - lo) / (hi - lo) * bins)
        counts[min(max(i, 0), bins - 1)] += 1
    top = max(counts) or 1
    for i, c in enumerate(counts):
        edge = lo + (hi - lo) * i / bins
        bar = "#" * int(c / top * width)
        print(f"  {edge:+.2f} |{bar:<{width}}| {c}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from facenet_pytorch import MTCNN, InceptionResnetV1

    mtcnn = MTCNN(image_size=160, margin=20, device=args.device, post_process=True)
    net = InceptionResnetV1(pretrained="vggface2").eval().to(args.device)

    @torch.no_grad()
    def embed(p):
        f = mtcnn(Image.open(p).convert("RGB"))
        if f is None:
            return None
        v = net(f.unsqueeze(0).to(args.device))[0].cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-8)

    recs = [json.loads(l) for l in (args.raw / "manifest.jsonl").read_text().splitlines()
            if l.strip()]
    print(f"re-embedding {len(recs)} sequences (no FLUX, no regeneration)\n")

    adjacent, endpoint, centroid, no_face = [], [], [], []
    per_seq = []

    for i, rec in enumerate(recs):
        ages = sorted(int(a) for a in rec["frames"])
        embs, missing = {}, []
        for a in ages:
            e = embed(rec["frames"][str(a)])
            if e is None:
                missing.append(a)
            else:
                embs[a] = e
        if missing:
            no_face.append((rec["id"], missing))
            continue

        adj = [float(np.dot(embs[ages[j]], embs[ages[j + 1]]))
               for j in range(len(ages) - 1)]
        end = float(np.dot(embs[ages[0]], embs[ages[-1]]))
        c = np.mean([embs[a] for a in ages], axis=0)
        c /= np.linalg.norm(c) + 1e-8
        cen = min(float(np.dot(embs[a], c)) for a in ages)

        adjacent.append(min(adj))
        endpoint.append(end)
        centroid.append(cen)
        per_seq.append({"id": rec["id"], "min_adj": min(adj),
                        "endpoint": end, "centroid": cen, "adj": adj})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(recs)}")

    print(f"\nusable: {len(per_seq)}  |  no-face: {len(no_face)}")
    for sid, ages in no_face:
        print(f"    {sid}: no face at {ages}")

    for name, xs in [("MIN ADJACENT (the right metric)", adjacent),
                     ("ENDPOINT 20-vs-80 (the broken one)", endpoint),
                     ("MIN CENTROID", centroid)]:
        print(f"\n{'='*58}\n{name}\n{'='*58}")
        print(f"  min {min(xs):.3f} | p10 {pct(xs,10):.3f} | med {st.median(xs):.3f} "
              f"| p90 {pct(xs,90):.3f} | max {max(xs):.3f}")
        hist(xs)

    print(f"\n{'='*58}\nKEEP RATE vs MIN-ADJACENT THRESHOLD\n{'='*58}")
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        n = sum(1 for x in adjacent if x >= t)
        star = "  <-- " if 0.40 <= t <= 0.55 else ""
        print(f"  min_adj >= {t:.2f}  ->  {n:3d}/{len(recs)} = {n/len(recs):5.1%}{star}")

    print(f"\n{'='*58}\nSANITY: does endpoint correlate with adjacent?\n{'='*58}")
    if len(adjacent) > 2:
        a = np.array(adjacent); e = np.array(endpoint)
        r = float(np.corrcoef(a, e)[0, 1])
        print(f"  pearson r = {r:+.3f}")
        if abs(r) < 0.4:
            print("  -> weakly related. Endpoint sim is NOT measuring identity")
            print("     drift; it's mostly measuring HOW MUCH THE FACE AGED.")
            print("     Confirms the endpoint gate was structurally wrong.")

    print(f"\n{'='*58}\nWORST 5 BY MIN-ADJACENT (look at these by eye)\n{'='*58}")
    for s in sorted(per_seq, key=lambda x: x["min_adj"])[:5]:
        print(f"  {s['id']}  min_adj={s['min_adj']:.3f}  "
              f"chain={[round(x,2) for x in s['adj']]}")
    print(f"\n{'='*58}\nBEST 5 (these should look great)\n{'='*58}")
    for s in sorted(per_seq, key=lambda x: -x["min_adj"])[:5]:
        print(f"  {s['id']}  min_adj={s['min_adj']:.3f}  "
              f"endpoint={s['endpoint']:.3f}  chain={[round(x,2) for x in s['adj']]}")

    out = args.raw.parent / "threshold_analysis.json"
    out.write_text(json.dumps(per_seq, indent=2))
    print(f"\nwrote {out}")
    print("\nPick the threshold where the histogram has a natural gap -- that gap")
    print("is the boundary between 'aged correctly' and 'became someone else'.")
    print("Then eyeball the WORST survivors to confirm before the full run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
