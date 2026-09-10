# yolo-class-recovery

Recover what each class index of an **undocumented object detector** responds to, using a
pretrained text-to-image diffusion model as a natural-image prior instead of raw pixel-space
activation maximisation (the "psychedelic" images).

You have `mystery.pt`, a YOLO checkpoint somebody fine-tuned, with class names missing, wrong or
just `0..N`. The package answers "what is class 7?" with a ranked list of candidate words and a
few clean images the detector fires on.

## The one call

```python
from classrecovery.represent import represent
results = represent("mystery.pt")          # {class_idx: images the detector fires on}
```

Reads the class count from the head architecture, then for each class index runs **classifier
guidance** (Dhariwal & Nichol, 2021) with the detector as the classifier: the diffusion model runs
*unconditionally* — no text, its only job is to know what natural images look like — and at every
denoising step the gradient of the detector's class-k score is pushed back through the diffusion
network into the noisy image. The detector's activations are the only thing that decides what gets
painted. Default prior is Stable Diffusion 2.1's unconditional branch (a broad natural-image prior);
pass `unconditional=True` with a text-free diffusers checkpoint (e.g. `google/ddpm-ema-church-256`)
for a model that has never seen a caption at all. `mode="embedding"` (textual inversion against the
detector) and `mode="latent"` (cheap x̂0-only nudging) are kept for comparison.

## How it works

| step | what | needs | cost |
|---|---|---|---|
| **weight diff** | compare against the public base checkpoint: which blocks moved, whether the head was re-initialised (class count changed), which class rows barely moved (probably still the original class), class-row similarity (related classes cluster), class-head bias (frequency prior — added classes are often rare) | weights of both | seconds |
| **prompt search** | sweep ~600 concrete nouns through the diffusion model, score every class on every image, report top words per class | forward passes only (works black-box) | ~4 min / 600 words on a T4 with `sd-turbo` |
| **guided sampling** | manifold-preserving guidance: at each denoising step decode the predicted clean image with TAESD, back-prop the target class score into it, nudge, re-noise. Box-area constraint, specificity margin, and augmentation averaging keep it from painting adversarial texture | gradients through the detector | ~1 min / class |
| **naming** | crop each image to the detector's best box, CLIP zero-shot against the vocabulary, aggregate across images, rank-fuse with prompt-search words | CLIP | seconds |
| **evaluate** | fine-tune a public YOLO on non-COCO classes yourself, strip the labels, measure top-k recovery | — | ~5 min training |

The guidance objective for class *k* is a soft-max over anchors of
`p_k − margin · max_{j≠k} p_j`, restricted to anchors whose box covers a sensible fraction of the
image, averaged over random crops/flips. Gradients never go through the UNet, so a Colab T4 is enough.

## Quick start (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dwtrott/yolo-class-recovery/blob/main/notebooks/colab_demo.ipynb)

Open `notebooks/colab_demo.ipynb`, pick a GPU runtime, run top to bottom. The last cell launches
the GUI with a public `*.gradio.live` link.

## Quick start (CLI)

```bash
pip install -e .
classrecovery testbed --out testbed --epochs 15              # makes testbed/mystery.pt + truth.json
classrecovery diff    --model testbed/mystery.pt --base yolov8n.pt
classrecovery search  --model testbed/mystery.pt --out runs/search
classrecovery recover --model testbed/mystery.pt --search-dir runs/search --out runs/recover
classrecovery eval    --results runs/recover --truth testbed/truth.json
classrecovery app     --share                                # Gradio GUI
```

## Python

```python
from classrecovery import Detector
from classrecovery.diffusion import DiffusionPrior, GuidanceConfig
from classrecovery.naming import Namer
from classrecovery.prompt_search import prompt_search
from classrecovery.pipeline import recover_class

det   = Detector("mystery.pt")
prior = DiffusionPrior("stabilityai/sd-turbo")
search = prompt_search(det, prior)                # gradient-free first pass
rec = recover_class(det, prior, Namer(), class_idx=3, search=search,
                    gcfg=GuidanceConfig(steps=8, strength=0.08, n_images=4))
print(rec.combined[:5])                           # [('zebra', ...), ('horse', ...), ...]
rec.guided_images[0].show()
```

## Knobs that matter

* **vocabulary** — the built-in list is broad and shallow. For a domain model, pass your own
  (`--vocab words.txt`, one term per line) and a matching template (`"an aerial photo of a {}"`).
* **guidance strength / range** — `strength` 0.05–0.15; guide from `t/T=1.0` down to `0.2`. Guiding
  through the last steps buys texture, not semantics, and is where adversarial patterns creep in.
* **specificity margin** — penalises images that light up other classes too; raise it when a class
  keeps resolving to a generic neighbour.
* **diffusion model** — `sd-turbo` for sweeps; `stable-diffusion-v1-5` / `sd-2-1-base` with 25–50
  steps and `cfg≈5` gives more steps to guide through when a class is stubborn.

## Layout

```
classrecovery/
  represent.py      the one-call interface
  detector.py       differentiable YOLO wrapper + ClassObjective
  diffusion.py      DiffusionPrior: generate(), guided_sample()
  prompt_search.py  vocabulary sweep
  naming.py         CLIP zero-shot naming (+ optional BLIP captions)
  weight_diff.py    fine-tune vs base analysis
  pipeline.py       recover_class / recover_all / evaluate
  testbed.py        build an "undocumented fine-tune" with known answers
  viz.py            per-class contact sheets
  app.py            Gradio GUI
  cli.py            command line
  vocab.py          built-in noun list
notebooks/colab_demo.ipynb
```

## Caveats

* Detectors are not robust classifiers; with too much guidance you get texture that scores high
  and means nothing. Watch the guidance trace and the crops, not just the score.
* A class the diffusion model cannot draw (a proprietary part, a rare variant) resolves to the
  nearest thing it *can* draw. Read results as "responds to things that look like X".
* If the class count changed in fine-tuning, the head was re-initialised and the base class order
  is gone — the weight diff tells you when that happened.
