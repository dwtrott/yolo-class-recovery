"""Generates notebooks/colab_demo.ipynb (kept as a script so the notebook is easy to regenerate)."""
import json, os

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Detector class recovery — Colab demo

Recover what each class index of an **undocumented YOLO checkpoint** responds to, using a
diffusion model as a natural-image prior instead of psychedelic activation-maximisation images.

Pipeline: **weight-diff** against the base checkpoint → **prompt search** (gradient-free sweep of a
noun vocabulary through the diffusion model) → **guided sampling** (steer the sampler with the
detector's class-score gradient) → **CLIP naming** → **evaluate** on a testbed with known answers.

Runtime → *Change runtime type* → **T4 GPU** (or better) before running.""")

code("""#@title 1. Install  (≈2 min)
import os, sys
REPO = "yolo-class-recovery"
if not os.path.exists(REPO):
    !git clone -q https://github.com/dwtrott/yolo-class-recovery.git
%pip install -q -e ./yolo-class-recovery
import importlib, classrecovery; print("classrecovery", classrecovery.__version__)""")

md("""## The one call that does what the project is for
Give it a checkpoint. It reads the class count from the head, and for every class index runs an unconditional diffusion
model under classifier guidance from the detector's own gradient — no text, no class names, no labels.
`testbed/mystery.pt` below is made in the next cell (a fine-tune with its names stripped); any Ultralytics `.pt`
you upload works the same way.""")

code("""#@title 2. Build a mystery model to test on  (fine-tune YOLOv8n on 4 non-COCO classes, strip the names; ≈3 min)
from classrecovery.testbed import build_testbed
build_testbed("testbed", dataset="african-wildlife.yaml", base="yolov8n.pt", epochs=15, imgsz=640)
print("wrote testbed/mystery.pt (names stripped) and testbed/truth.json (answers, for checking yourself later)")""")

code("""#@title 3b. Robust mode: stable-concept search  (≈10 min/class on an A100)
#@markdown Generates many candidates, scores them for augmentation-consistency, box stability, localization
#@markdown (score must collapse when the box is masked) and cutout survival, clusters the detector's own
#@markdown internal activations, and regenerates toward the dominant mode. Sheet shows the final images first,
#@markdown then members of each discovered mode.
from classrecovery.represent import represent
results = represent("testbed/mystery.pt", mode="robust", n_candidates=24, steps=50, strength=0.1, n_images=4,
                    out_dir="runs/represent")""")

code("""#@title 3. Representative images for every class  (classifier guidance; a few min/class on an A100)
#@markdown No text, no labels. The diffusion model runs unconditionally (pure natural-image prior) and at every
#@markdown denoising step the gradient of the detector's class-k score is pushed back through the diffusion
#@markdown network into the noisy image. The detector's activations are the only thing deciding the content.
#@markdown Default prior: OpenAI's *unconditional* 256px ImageNet diffusion model (never saw a caption; ~2 GB download).
from classrecovery.represent import represent
results = represent("testbed/mystery.pt", prior_id="openai/imagenet-256-uncond", steps=50, strength=0.1,
                    n_images=4, out_dir="runs/represent")
# results[c].guided_images -> PIL images for class c, strongest detector response first""")

md("""Everything below is optional: naming the classes automatically, checking against the testbed's answers, and the GUI.""")

code("""#@title (optional) Reveal the testbed's true class names
import json; print(json.load(open("testbed/truth.json"))["names"])""")

code("""#@title (optional) Weight diff against the base checkpoint  (seconds)
from classrecovery import Detector
from classrecovery.weight_diff import weight_diff
mystery, base = Detector("testbed/mystery.pt"), Detector("yolov8n.pt")
print("names in mystery checkpoint:", mystery.names)
print(weight_diff(mystery, base).summary())""")

code("""#@title (optional) Prompt search over the built-in vocabulary  (≈4 min on T4 with sd-turbo)
from classrecovery.diffusion import DiffusionPrior
from classrecovery.prompt_search import prompt_search
prior = DiffusionPrior("stabilityai/sd-turbo")
search = prompt_search(mystery, prior, n_per_word=2, progress=lambda d,t,p: print(f"\\r{d}/{t} {p:<40}", end=""))
print(); print(search.summary(6))
search.save("runs/search")""")

code("""#@title (optional) Guided recovery + CLIP naming for every class  (≈1 min / class on T4)
from classrecovery.naming import Namer
from classrecovery.pipeline import recover_class, evaluate
from classrecovery.diffusion import GuidanceConfig
from IPython.display import display
namer = Namer()
gcfg = GuidanceConfig(steps=8, cfg=0.0, strength=0.08, repeats=2, n_images=4, seed=0)
results = {}
for c in range(mystery.nc):
    rec = recover_class(mystery, prior, namer, c, search=search, gcfg=gcfg)
    results[c] = rec
    print(f"class {c}: prompt={rec.prompt_used!r}")
    print("   search:", [w for w,_ in rec.search_words[:5]])
    print("   clip:  ", [w for w,_ in rec.clip_names[:5]])
    print("   final: ", [w for w,_ in rec.combined[:5]])
    for im in rec.guided_images[:2]: display(im.resize((256,256)))
truth = {int(k): v for k, v in json.load(open("testbed/truth.json"))["names"].items()}
ev = evaluate(results, truth); print({k: ev[k] for k in ("top1","top3","top5")}); print(ev["per_class"])""")

code("""#@title (optional) Contact sheet: what the diffusion model made for each class
from classrecovery.viz import contact_sheets, stack
sheet = stack(contact_sheets(results, search))      # add names=truth to print the answers in the titles
display(sheet); sheet.save("runs/contact_sheet.png")""")

code("""#@title GUI  (prints a public gradio.live link; also renders inline)
from classrecovery.app import launch
launch(share=True, debug=False)""")

md("""### Bring your own mystery model
Upload any Ultralytics `.pt` to Colab, then in the GUI's **Model** tab point at it (pick the base
checkpoint it was probably fine-tuned from, or `(none)`).  Give the **Prompt search** tab a
domain-specific vocabulary (one term per line) and a domain prefix such as
`an aerial photo of a {}` if the original classes suggest overhead imagery.""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
os.makedirs("notebooks", exist_ok=True)
json.dump(nb, open("notebooks/colab_demo.ipynb", "w"), indent=1)
print("wrote notebooks/colab_demo.ipynb")
