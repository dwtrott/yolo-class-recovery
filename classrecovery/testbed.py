"""Build an "undocumented fine-tune" to evaluate the recovery pipeline on.

We fine-tune a public COCO-pretrained YOLO on a small dataset whose classes
are *not* in COCO (Ultralytics' ``african-wildlife``: buffalo, elephant,
rhino, zebra — auto-downloaded), then produce a copy of the checkpoint with
the class names and training metadata stripped.  The pipeline is run against
the stripped copy; the original is only used for scoring.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, Optional

import torch


def strip_checkpoint(src: str, dst: str, keep_nc_names: bool = False) -> Dict[int, str]:
    """Copy ``src`` -> ``dst`` with class names replaced by ``class_<i>`` and training args removed.

    Returns the original names so they can be saved as ground truth.
    """
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    model = ckpt.get("ema") or ckpt.get("model")
    orig = dict(getattr(model, "names", {}))
    anon = {int(i): f"class_{int(i)}" for i in orig} if not keep_nc_names else orig
    for key in ("model", "ema"):
        m = ckpt.get(key)
        if m is not None:
            m.names = anon
            if hasattr(m, "yaml") and isinstance(m.yaml, dict):
                m.yaml.pop("names", None)
    for key in ("train_args", "train_results", "train_metrics", "date", "license", "docs"):
        ckpt.pop(key, None)
    ckpt["train_args"] = {"task": "detect"}      # ultralytics expects the key to exist
    torch.save(ckpt, dst)
    return {int(k): str(v) for k, v in orig.items()}


def build_testbed(out_dir: str = "testbed", dataset: str = "african-wildlife.yaml", base: str = "yolov8n.pt",
                  epochs: int = 15, imgsz: int = 640, batch: int = 16, device: Optional[str] = None,
                  freeze: Optional[int] = None, **train_kw) -> Dict[str, str]:
    """Fine-tune ``base`` on ``dataset`` and write ``truth.pt`` / ``mystery.pt`` / ``truth.json``.

    ``freeze=10`` keeps the backbone fixed (a common cheap fine-tune style);
    ``None`` trains everything.
    """
    from ultralytics import YOLO

    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(base)
    kw = dict(data=dataset, epochs=epochs, imgsz=imgsz, batch=batch, project=os.path.join(out_dir, "runs"),
              name="finetune", exist_ok=True, verbose=False, plots=False)
    if device is not None:
        kw["device"] = device
    if freeze is not None:
        kw["freeze"] = freeze
    kw.update(train_kw)
    model.train(**kw)
    save_dir = str(getattr(model.trainer, "save_dir", os.path.join(out_dir, "runs", "finetune")))
    best = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.exists(best):
        best = os.path.join(save_dir, "weights", "last.pt")
    truth = os.path.join(out_dir, "truth.pt")
    mystery = os.path.join(out_dir, "mystery.pt")
    shutil.copy(best, truth)
    names = strip_checkpoint(truth, mystery)
    with open(os.path.join(out_dir, "truth.json"), "w") as f:
        json.dump({"names": names, "base": base, "dataset": dataset}, f, indent=2)
    return {"truth": truth, "mystery": mystery, "truth_json": os.path.join(out_dir, "truth.json"), "base": base}
