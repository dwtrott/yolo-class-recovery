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

code("""#@title 2. A mystery model to test on  (fine-tune YOLOv8n on 4 non-COCO classes, strip the names; ≈3 min)
#@markdown Skip this if you have your own checkpoint — upload it and use its path below.
from classrecovery.testbed import build_testbed
build_testbed("testbed", dataset="african-wildlife.yaml", base="yolov8n.pt", epochs=15, imgsz=640)""")

code("""#@title 3. An unlabeled photo pool  (COCO val2017, 5 000 images, ≈1 GB; ≈1 min)
#@markdown Any folder of images works. No labels are read — the pool is just photographs.
from classrecovery.seeds import download_pool
POOL = download_pool("coco-val2017")
print(POOL)""")

code("""#@title 4. Recover one class first  (≈5 min; first run also downloads the 2 GB diffusion prior)
from classrecovery import represent
results = represent("testbed/mystery.pt", pool=POOL, classes=[3], n_seeds=6, n_noise=4,
                    steps=50, strength=0.1, smooth_k=4, out_dir="runs/v1")
#@markdown The sheet shows the best images first; each is labelled seed / refined / noise with the detector
#@markdown score and the robustness score.""")

code("""#@title 5. All classes
results = represent("testbed/mystery.pt", pool=POOL, n_seeds=6, n_noise=4, steps=50, strength=0.1, out_dir="runs/v1")""")

code("""#@title 6. (optional) Prototype pass — cluster the candidates' internal activations and re-guide toward the dominant mode
results = represent("testbed/mystery.pt", pool=POOL, classes=[3], use_prototype=True, out_dir="runs/v1_proto")""")

code("""#@title 7. (afterwards) The testbed's real names
import json; print(json.load(open("testbed/truth.json"))["names"])""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
os.makedirs("notebooks", exist_ok=True)
json.dump(nb, open("notebooks/colab_demo.ipynb", "w"), indent=1)
print("wrote notebooks/colab_demo.ipynb")
