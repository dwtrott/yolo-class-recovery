"""Generates notebooks/colab_demo.ipynb."""
import json, os

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Detector class recovery

Checkpoint in → clear, high-scoring images per class out. Nothing about the model's classes is assumed:
the class count is read from the head, and every image comes from the detector's own activations steering
an **unconditional** diffusion prior (no text, no labels, no vocabulary).

Per class: **seed** (scan an unlabeled photo pool for crops the class fires on) → **refine** (re-noise each seed
halfway and denoise under smoothed detector guidance) → **noise** (generate from scratch if the pool gave nothing)
→ **rank** by a robustness composite (augmentation consistency, box stability, localization, cutout survival).

Runtime → *Change runtime type* → **A100 GPU**.""")

code("""#@title 1. Install  (≈2 min)
!git clone -q https://github.com/dwtrott/yolo-class-recovery.git
%pip install -q ./yolo-class-recovery
import classrecovery; print("classrecovery", classrecovery.__version__)""")

code("""#@title 2. A mystery model to test on  (fine-tune YOLOv8n, strip the names; ≈3–10 min)
#@markdown `african-wildlife` is the easy case (ImageNet can draw all four, COCO val contains two of them).
#@markdown Harder, fairer tests — classes the prior cannot draw and the pool does not contain:
#@markdown `signature.yaml` (handwritten signatures), `medical-pills.yaml`, `brain-tumor.yaml`,
#@markdown `dota8.yaml` / `VisDrone.yaml` (aerial: storage tank, roundabout, ship, small vehicle ...).
#@markdown Skip this cell if you have your own checkpoint — upload it and use its path below.
DATASET = "african-wildlife.yaml"  #@param ["african-wildlife.yaml", "signature.yaml", "medical-pills.yaml", "brain-tumor.yaml", "dota8.yaml", "VisDrone.yaml"]
EPOCHS = 15  #@param {type:"integer"}
from classrecovery.testbed import build_testbed
build_testbed("testbed", dataset=DATASET, base="yolov8n.pt", epochs=EPOCHS, imgsz=640)""")

code("""#@title 3. An unlabeled photo pool  (COCO val2017, 5 000 images, ≈1 GB; ≈1 min)
#@markdown Any folder of images works. No labels are read — the pool is just photographs.
from classrecovery.seeds import download_pool
POOL = download_pool("coco-val2017")
print(POOL)""")

code("""#@title 4. Recover one class first  (≈5 min; first run also downloads the 2 GB diffusion prior)
from classrecovery import represent
results = represent("testbed/mystery.pt", pool=POOL, classes=[0], n_seeds=6, n_noise=4, out_dir="runs/v1")
#@markdown Best images first, each labelled seed / refined / noise with: `det` (raw detector score),
#@markdown `degraded` (mean score under noise / blur / JPEG / half-res — adversarial patterns collapse here),
#@markdown `err` (the prior's reconstruction error — off-manifold images score high and are flagged).""")

code("""#@title 5. All classes
results = represent("testbed/mystery.pt", pool=POOL, n_seeds=6, n_noise=4, out_dir="runs/v1")""")

code("""#@title 6. Noise-robust twin: distil a copy of the detector on the pool (no labels; ≈10 min), then guide with it
#@markdown The original detector's input gradient is texture-shaped, which is why from-noise samples come out as
#@markdown stripes. The twin is trained to reproduce the original's predictions on clean pool photos while seeing
#@markdown noised/blurred versions, so its gradient points at object structure. It is used for guidance only;
#@markdown the original detector still does all scoring.
from classrecovery import Detector
from classrecovery.seeds import list_images
from classrecovery.twin import train_twin, gradient_alignment_check
from PIL import Image
det = Detector("testbed/mystery.pt", imgsz=320)
twin = train_twin(det, list_images(POOL), steps=1500, batch=16, save_path="runs/twin.pt")
probe = Image.open(list_images(POOL)[0]).convert("RGB")
print("high-frequency share of the guidance gradient (lower = more object-shaped):",
      gradient_alignment_check(det, twin, probe, 3))""")

code("""#@title 6a. From noise, guided by the twin — the real test of the generation path
results_noise = represent("testbed/mystery.pt", pool=None, classes=[3], n_noise=8, twin=twin, out_dir="runs/noise_twin")""")

code("""#@title 6b. Full pipeline with the twin
results = represent("testbed/mystery.pt", pool=POOL, twin=twin, n_seeds=6, n_noise=4, out_dir="runs/v1_twin")""")

code("""#@title 6c. (optional) Prototype pass — cluster the candidates' internal activations and re-guide toward the dominant mode
results = represent("testbed/mystery.pt", pool=POOL, classes=[3], use_prototype=True, out_dir="runs/v1_proto")""")

code("""#@title 7. (afterwards) The testbed's real names
import json; print(json.load(open("testbed/truth.json"))["names"])""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
os.makedirs("notebooks", exist_ok=True)
json.dump(nb, open("notebooks/colab_demo.ipynb", "w"), indent=1)
print("wrote notebooks/colab_demo.ipynb")
