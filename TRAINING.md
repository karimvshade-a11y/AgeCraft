# Training on a rented GPU, on a small balance

The dataset is done and lives on HuggingFace. What is left is training, and
training is the part you buy in hours.

**The idea that makes a small balance workable:** the checkpoint is uploaded to
HuggingFace every 30 minutes, and every run resumes from the last checkpoint.
So "the balance ran out" and "training is paused" are the same event. Top up,
run it again, it continues. You are never restarting from zero.

---

## 0. Use a fresh token

The training pod needs a **write** token. Make a new one each time rather than
reusing an old one -- a token typed into a pod terminal is visible in that
terminal's scrollback and in any screenshot of it.

huggingface.co/settings/tokens -> **Create new token** -> type **Write**.

Delete the previous one on that same page while you are there.

---

## 1. Deploy the pod

Do this only when you are at the keyboard. A deployed pod bills from the second
it boots, and it cannot start working until you paste the token in.

| Setting | Value |
|---|---|
| Template | RunPod PyTorch (any recent) |
| GPU | RTX 4090, Community Cloud (~$0.34/hr) |
| Container disk | 20 GB |
| Volume disk | 60 GB (dataset unpacks to ~10 GB) |
| Network volume | **none** |

No network volume on purpose. A network volume is what quietly billed $7/month
after the last run; everything worth keeping goes to HuggingFace anyway, so
when this pod dies there should be nothing left behind to charge you for.

---

## 2. Start it

Open the pod's **web terminal** and paste these three blocks.

```bash
git clone https://github.com/karimvshade-a11y/AgeCraft.git /workspace/agecraft
```

Then the token — paste your new one after the `=`:

```bash
export HF_TOKEN=
```

Then launch:

```bash
cd /workspace/agecraft && export MAX_HOURS=6 && setsid bash scripts/run_training.sh < /dev/null &
```

`setsid`, not `nohup`: RunPod's web terminal drops its session fairly often, and
a `nohup` job started from it still died with it. `setsid` puts the run in its
own session so nothing the terminal does can reach it. The script writes its own
log to `/workspace/train.log`, so there is no redirect to get wrong.

`MAX_HOURS` is your maximum loss. 6 hours of a 4090 is about $2.10.

Now you can close the browser. Losing power or connection costs nothing: the
job holds its own deadline and kills its own machine.

---

## 3. What should happen

Within the first two minutes the log proves the expensive parts will work
before any of them run:

```bash
tail -f /workspace/train.log
```

- `GPU: NVIDIA GeForce RTX 4090 25.4GB` — right machine
- `token: 37 chars, starts 'hf_'` then `authenticated as Abdelkarim40, role='write'`
- `HF write OK -> Abdelkarim40/agecraft-weights` — the token can actually save
  your work. **This is the check that failed silently in June.**

If any check fails the run does **not** kill the pod immediately. It writes the
reason to the log, tries to upload it, and holds the machine for `GRACE_MIN`
(default 10) minutes so you can actually read it — because the first version of
this script terminated instantly on failure and took the only copy of the reason
with it. To keep a failed pod alive longer while you dig: `touch /workspace/HOLD`.
- `17340 pairs across 867 identities` — the dataset came down and every path
  resolves
- then `e0 s0 G ... D ... l1 ...` and it is training

Expect roughly 7 minutes per epoch, so ~50 epochs in a 7-hour budget. Watch
`l1` fall and `idc` stay near zero (`idc` is the identity-cycle term: ageing a
face to its own age must be a no-op, and it is what keeps the sliders honest).

---

## 4. What you get

In `huggingface.co/Abdelkarim40/agecraft-weights`:

- `last.pt` — full training state. This is the resume point.
- `agecraft.onnx` + `agecraft.onnx.data` — the shippable model. **Both files.**
  The `.onnx` alone is topology with no weights in it.
- `train.log`

Pull the model down with:

```bash
python -c "from huggingface_hub import hf_hub_download as d; [d('Abdelkarim40/agecraft-weights', f) for f in ('agecraft.onnx','agecraft.onnx.data','last.pt')]"
```

---

## 5. Next time

Exactly the same three commands. It finds `last.pt` on HuggingFace, resumes
from that epoch, and keeps going. Nothing to configure and nothing to
remember.

If you would rather not rent anything: `configs/gtx1650.yaml` trains the same
data at home for free, slower and softer. `--resume` works there too.
