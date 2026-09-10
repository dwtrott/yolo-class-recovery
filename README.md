# yolo-class-recovery

**Checkpoint in → clear, high-scoring images per class out.**

You are handed an object-detection checkpoint (Ultralytics YOLO) that somebody fine-tuned on
something, with no documentation and no class names. This package produces, for every class index,
clear representative images the class fires on — using nothing but the detector's own activations
and an unconditional diffusion prior. No text, no captions, no vocabulary, no labels anywhere.

```python
from classrecovery import represent
from classrecovery.seeds import download_pool

pool = download_pool("coco-val2017")            # any folder of unlabeled photos works
results = represent("mystery.pt", pool=pool)    # {class_idx: ClassResult}, contact sheet per class
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dwtrott/yolo-class-recovery/blob/main/notebooks/colab_demo.ipynb)

## How it works

The class count is read from the head architecture (`4 + nc` output channels). Then, per class:

1. **Seed.** Scan an unlabeled image pool with the detector. Keep the crops the class fires on
   *and* that pass the robustness checks. Real photos, so they are clear by construction. For a
   class the pool covers, this alone is usually the answer.
2. **Refine.** SDEdit: re-noise each seed to ~50 % of the diffusion schedule and denoise it under
   detector guidance. The prior only has to sharpen an object it is already sitting on into what
   the class most wants to see.
3. **Noise.** If the pool yielded nothing (a class it does not cover), sample from pure noise under
   the same guidance — classifier guidance in the original Dhariwal & Nichol sense, with the
   detector as the classifier.
4. **Rank.** Score every candidate with a robustness composite and show the best first, labelled by
   where it came from (`seed` / `refined` / `noise`).

**Guidance** pushes the gradient of the class objective back *through the diffusion network* into
the noisy image at each step, and averages that gradient over noisy copies of the decoded image
(SmoothGrad). Texture-biased detector gradients are the root cause of "psychedelic" feature
visualisations; smoothing is the cheapest counter-measure, a noise-robust twin of the detector is
the thorough one (planned).

**Robustness composite** (`robust.py`), all differentiable and label-free:

| term | meaning |
|---|---|
| class | soft-max class score averaged over crop / flip / rotate / colour augmentations |
| consistency | −Var of that score across augmentations |
| box | spread of the best box across augmentations, mapped back to the original frame |
| localization | the score must **collapse** when the best box is masked out — a wall-to-wall texture cannot pass this |
| cutout | the score must **survive** small erasures inside the box |
| prototype (optional) | cosine distance of multi-layer ROI activations to the dominant cluster of earlier candidates |

**Prior.** OpenAI's unconditional 256 px ImageNet diffusion model (`gd/`, vendored, MIT). It has
never seen a caption; a gradient is the only steering it accepts.

## Layout

```
classrecovery/
  represent.py   the pipeline (seed → refine → noise → rank)
  detector.py    differentiable YOLO wrapper + ClassObjective
  prior.py       UncondPrior: guided_sample (from noise), guided_refine (from a seed)
  seeds.py       unlabeled-pool mining, pool download
  robust.py      robustness composite, feature taps, prototype discovery
  viz.py         ClassResult + contact sheets
  weight_diff.py compare a fine-tune against its base checkpoint (what moved, class-row matching)
  testbed.py     make an "undocumented fine-tune" with known answers to evaluate on
  gd/            vendored guided-diffusion UNet
```

## Knobs

* `strength` (0.05–0.2): detector vs prior. Score flat and images generic → raise; images texture-y → lower.
* `smooth_k` (1–8): gradient smoothing; 4 is a good default, 8 if speckle persists.
* `refine_noise` (0.3–0.7): how much of the seed to keep; lower keeps more of the photo.
* `guide_to` (0.15–0.5): stop guiding earlier to let the prior finish cleanly.
* `use_prototype`: cluster candidates' internal activations and re-guide toward the dominant mode.

## Caveats

* A class the prior cannot draw (a proprietary part, a rare variant) resolves to the nearest clear
  thing that fires it. Read the images as "what this class responds to".
* The african-wildlife testbed is friendly (its classes exist in ImageNet). Evaluate on something
  the prior cannot draw before relying on the method — the Ultralytics `signature` or
  `medical-pills` sets are ready-made.
* History: the exploration phase (prompt search, CLIP naming, text-embedding inversion, GUI) is
  tagged `v0-exploration`.
