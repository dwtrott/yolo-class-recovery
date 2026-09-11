
import json, os
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
md("""# Hard testbed + blind evaluation

Classes the prior cannot draw and the pool does not contain, three variants side by side
(retrieval-only baseline / anchored generation / noise-only fallback), and a blind-judging package.
Runtime → A100.""")
code("""#@title 1. Install
!git clone -q https://github.com/dwtrott/yolo-class-recovery.git
%pip install -q ./yolo-class-recovery
import classrecovery; print("classrecovery", classrecovery.__version__)""")
code("""#@title 2. Mystery model on a hard dataset
DATASET = "signature.yaml"  #@param ["signature.yaml", "medical-pills.yaml", "brain-tumor.yaml", "dota8.yaml", "VisDrone.yaml", "african-wildlife.yaml"]
EPOCHS = 20  #@param {type:"integer"}
from classrecovery.testbed import build_testbed
build_testbed("testbed", dataset=DATASET, base="yolov8n.pt", epochs=EPOCHS, imgsz=640)""")
code("""#@title 3. Pool + twin  (≈12 min)
from classrecovery import Detector
from classrecovery.seeds import download_pool, list_images
from classrecovery.twin import train_twin
POOL = download_pool("coco-val2017")
det = Detector("testbed/mystery.pt", imgsz=320)
twin = train_twin(det, list_images(POOL), steps=1500, batch=16, save_path="runs/twin.pt")""")
code("""#@title 4. Three variants, all classes  (≈10 min per class total)
from classrecovery.evaluate import run_variants, table
ev = run_variants("testbed/mystery.pt", POOL, twin=twin, out_dir="runs/eval")
print(table(ev["rows"]))
#@markdown `best_source` tells you what produced the top image; compare `det`/`degraded` across variants per class.""")
code("""#@title 5. Look at the anchored sheets
from IPython.display import display
from classrecovery.viz import class_sheet
for c, r in ev["results"]["anchored"].items():
    display(class_sheet(r, title=f"class {c}  ({r.note})"))""")
code("""#@title 6. Blind-judging package  (hand `runs/blind/sheets` + `judge_form.csv` to someone who has not seen truth.json)
from classrecovery.evaluate import blind_package
blind_package(ev["results"]["anchored"], out_dir="runs/blind", seed=0)
!ls runs/blind runs/blind/sheets
!zip -q -r blind_sheets.zip runs/blind/sheets runs/blind/judge_form.csv && echo "download blind_sheets.zip from the file pane"
""")
code("""#@title 7. Score a filled judge form  (upload the completed judge_form.csv first)
from classrecovery.evaluate import score_blind
res = score_blind("judge_form.csv", "runs/blind/key.json", "testbed/truth.json")
print("blind accuracy:", res["accuracy"]); [print(r) for r in res["rows"]]""")
code("""#@title 8. (afterwards) the true names
import json; print(json.load(open("testbed/truth.json"))["names"])""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
os.makedirs("notebooks", exist_ok=True)
json.dump(nb, open("notebooks/hard_testbed.ipynb", "w"), indent=1)
print("wrote notebooks/hard_testbed.ipynb")
