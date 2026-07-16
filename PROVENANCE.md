# Provenance & Licence Ledger

**This is the most commercially important file in the repo.** The model
architecture is public. The code is a weekend. The thing that makes this
sellable when competitors aren't is a defensible claim that our weights derive
only from permissively-licensed assets.

Keep this current. Every time you add a dependency that touches training data
or weights, add a row. If you can't fill in the licence column, don't add the
dependency.

> I'm not a lawyer and this is not legal advice. This ledger is engineering
> documentation intended to make a lawyer's job cheap and fast. Get the ⚠️ rows
> reviewed before you take money.

---

## Why this exists

The entire open re-aging lineage is legally poisoned for commercial use:

| Asset | Licence | Sellable? |
|---|---|---|
| NVIDIA StyleGAN / StyleGAN2 / StyleGAN3 | CC BY-NC 4.0 | ❌ No |
| FFHQ dataset | CC BY-NC-SA 4.0 (collection) | ❌ No |
| SAM (Alaluf et al.) | built on StyleGAN2 | ❌ No |
| Existing FRAN reimplementations | trained on FFHQ re-aged with SAM | ❌ No |
| CACD / AgeDB / IMDB-WIKI / UTKFace / FG-NET | research-only | ❌ No |
| MORPH | commercial licence, paid | 💰 Purchasable |

Everyone building in this space either doesn't notice, or notices and quits.
**That gap is the business.**

---

## Our lineage

### Training data (ships as: nothing — data stays local)

| Asset | Role | Licence | Status |
|---|---|---|---|
| Prompt matrix | identity specs | **Ours** (authored in this repo) | ✅ |
| Qwen2.5 via Ollama | prompt phrasing only, offline | Apache 2.0 (Qwen2.5 ≤72B) | ✅ |
| FLUX.1-schnell | generates every training pixel | **Apache 2.0** | ✅ |

**FLUX.1-schnell specifically.** Not FLUX.1 [dev], not FLUX.1 Kontext — both are
under the non-commercial FLUX.1 [dev] licence. `generate_pairs.py` hardcodes
schnell and there is a comment telling you not to swap it. Believe the comment.

**The HF repo is gated; the licence is not.** Downloading schnell requires a HF
account, accepting terms on the model page, and a read token. This is an access
gate (email collection), *not* a licence restriction — BFL still publishes
schnell under Apache 2.0 for personal, scientific, and commercial use. Clicking
"Agree and access" does not encumber your outputs. Keep a dated screenshot of
the model card's licence field with your records anyway; licences can change
going forward, and what matters is the licence on the day you downloaded.

**Possible upgrade path:** BFL's later FLUX.2 series includes *Klein*, also
Apache 2.0. If it holds identity better under seed-locking it could raise the
keep rate substantially. Same licence posture, so the provenance story is
unchanged. Worth benchmarking once schnell's baseline keep rate is known —
do not switch mid-dataset.

**No identity adapter.** PuLID / InstantID / IP-Adapter FaceID would be the
obvious way to hold identity across ages, but they depend on InsightFace
`antelopev2`, which is non-commercial. We use seed-and-latent locking plus
aggressive filtering instead. It's a worse algorithm with a better licence, and
the licence is what's scarce.

### QC models (ships as: nothing — offline filtering only)

| Asset | Role | Licence | Status |
|---|---|---|---|
| facenet-pytorch | identity filter | MIT (code) | ⚠️ see below |
| `nateraw/vit-age-classifier` | age filter | Apache 2.0 (fine-tuned on FairFace, CC BY 4.0) | ⚠️ verify |

⚠️ **The nuance.** These models never enter our weights — they only *select*
which generated images we keep. Using a model to filter data is a weaker
exposure than embedding or distributing it, but "weaker" isn't "none", and the
question of whether filtering creates a derivative work is genuinely unsettled.

facenet-pytorch's code is MIT, but its `vggface2` pretrained weights derive from
the VGGFace2 dataset, which was research-licensed and has been withdrawn. This
is the shakiest row in the ledger.

It's also a dependency-hell package: its `setup.py` declares `torch<=2.3.0`,
`torchvision<=0.18.0`, `numpy<2.0.0`, `Pillow<10.3.0`. Those pins are stale
rather than real -- the library runs fine on current torch -- but pip believes
them and will downgrade a working install. Always `pip install --no-deps
facenet-pytorch`. Two independent reasons (licence + stale pins) to replace it;
the embedder is behind a pluggable interface in `filter_pairs.py` for exactly
this. A permissively-licensed ONNX embedder run through onnxruntime would kill
both problems at once and drop a torch dependency from the QC stage.

Both are behind a pluggable interface in `filter_pairs.py` for exactly this
reason — swap in whatever counsel is comfortable with, and rerun `dvc repro`.

### Training losses (ships as: nothing)

| Asset | Role | Licence | Status |
|---|---|---|---|
| LPIPS (VGG backbone) | perceptual loss | BSD-2 code; backbone weights murky | ⚠️ **off by default** |

`configs/base.yaml` ships `perceptual: none`. L1 + PatchGAN converges slower and
gives slightly softer skin texture, but the lineage is unambiguous. Turn LPIPS
on only after review. The flag exists so this is your decision and it's
reversible.

### Shipped runtime

| Asset | Licence | Status |
|---|---|---|
| Our trained U-Net weights | **Ours** | ✅ |
| ONNX Runtime | MIT | ✅ |
| PySide6 | LGPL v3 — free for closed-source **if dynamically linked** | ✅ |
| OpenCV | Apache 2.0 | ✅ |
| MediaPipe | Apache 2.0 | ✅ |
| BiSeNet face parsing | ⚠️ check the specific checkpoint you use | ⚠️ |
| NumPy / Pillow | BSD / MIT-CMU | ✅ |

**PySide6, not PyQt.** PyQt is GPL-or-commercial: shipping a closed-source app
on PyQt means buying a Riverbank licence. PySide6 is LGPL, which is free for
commercial closed-source provided you link dynamically and don't statically
bundle it. PyInstaller's default onedir mode is fine here; verify before you
ship a onefile build.

**ffmpeg.** If you add video export, ship an **LGPL** build, not a GPL one.
`imageio-ffmpeg`'s bundled binaries are LGPL. Building your own with
`--enable-gpl` silently makes your whole app GPL.

---

## Safety & misuse posture

Not a legal footnote — an app-store and payment-processor survival issue.

1. **Adult-only training distribution.** `AGE_ANCHORS` starts at 20 and the
   prompt system prompt bans any child/teen phrasing (`BANNED` in
   `generate_prompts.py`). A model that has never seen a minor is dramatically
   harder to steer toward producing one. This constraint is why our de-aging
   floor is ~20 rather than ~10, and that is a deliberate trade.
2. **De-aging is the sensitive direction.** Age-*progression* is benign.
   Age-*regression* on an uploaded photo is the capability that gets products
   pulled. Enforce a floor in the UI and don't take the model below its
   training distribution.
3. **Consent gate at import.** Explicit affirmation that the subject is an
   adult and consented. Log it locally.
4. **Provenance metadata on output.** Embed C2PA / a visible marker. This is
   moving toward mandatory in several jurisdictions and it's cheap now,
   expensive to retrofit.
5. **All local, no upload.** Your privacy story *is* your marketing story
   against FaceApp. Don't quietly add telemetry that undermines it.

---

## Audit checklist before you charge money

- [ ] `dvc.lock` committed; `dvc repro` reproduces weights from scratch
- [ ] No `dev`/Kontext FLUX variant anywhere in the dep tree
- [ ] `pip-licenses --format=markdown` output reviewed and archived
- [ ] No InsightFace in the transitive dep tree (`pip show insightface` → not found)
- [ ] Every ⚠️ row above reviewed by counsel or removed
- [ ] ffmpeg build confirmed LGPL
- [ ] PySide6 dynamically linked
- [ ] Consent gate + de-aging floor live in the shipped UI
