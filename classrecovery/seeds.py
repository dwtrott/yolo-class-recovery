"""Seed mining: find real, unlabeled photos the mystery class already fires on.

No labels are consulted.  The detector is run over a pool of ordinary images
(any folder; ``download_pool`` fetches COCO val2017 — 5 000 diverse photos —
as a default), detections are re-scored with the robust objective, and the
best crops become seeds for ``UncondPrior.guided_refine``.  For anything the
pool covers this beats generating from noise; for anything it doesn't, the
pool simply yields nothing and generation from noise takes over.
"""

from __future__ import annotations

import glob
import os
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from .detector import Detector

POOLS = {
    "coco-val2017": "http://images.cocodataset.org/zips/val2017.zip",          # ~1 GB, 5000 images
    "coco128": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",  # 7 MB, 128 images
}


def download_pool(name: str = "coco-val2017", root: str = "pools") -> str:
    """Download and unzip an image pool; returns the directory of images."""
    import urllib.request
    os.makedirs(root, exist_ok=True)
    url = POOLS[name]
    zpath = os.path.join(root, os.path.basename(url))
    out = os.path.join(root, name)
    if not os.path.isdir(out):
        if not os.path.exists(zpath):
            print(f"[classrecovery] downloading {url}")
            urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    return out


def list_images(directory: str, limit: Optional[int] = None) -> List[str]:
    files = sorted(f for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
                   for f in glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return files[:limit] if limit else files


def _square_crop(im: Image.Image, box01: Sequence[float], pad: float = 0.35, size: int = 256) -> Image.Image:
    """Crop around a normalised xyxy box, padded and squared, resized to ``size``."""
    W, H = im.size
    x0, y0, x1, y1 = box01
    cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
    side = max((x1 - x0) * W, (y1 - y0) * H) * (1 + 2 * pad)
    side = max(side, 32)
    l, t = cx - side / 2, cy - side / 2
    return im.crop((int(l), int(t), int(l + side), int(t + side))).resize((size, size))


@torch.no_grad()
def mine_seeds(detector: Detector, images: Sequence[str], class_idx: int, top_k: int = 8, batch: int = 32,
               min_area: float = 0.01, max_area: float = 0.95, robust: bool = True, crop: bool = True,
               size: int = 256, progress=None) -> List[Dict[str, object]]:
    """Score every image in ``images`` for ``class_idx``; return the ``top_k`` seeds.

    Each seed: ``{"path", "image" (crop or full), "score", "robust", "box"}``.
    """
    from .detector import ClassObjective
    from .robust import RobustObjective
    hits: List[Tuple[float, str]] = []
    for b in range(0, len(images), batch):
        paths = images[b:b + batch]
        ims = []
        for p in paths:
            try:
                ims.append(Image.open(p).convert("RGB"))
            except Exception:
                ims.append(Image.new("RGB", (64, 64)))
        sc = detector.class_scores(ims, min_area=min_area, max_area=max_area)[:, class_idx]
        hits += [(float(s), p) for s, p in zip(sc, paths)]
        if progress and (b // batch) % 10 == 0:
            progress(f"   scanned {min(b + batch, len(images))}/{len(images)}  best so far {max(h[0] for h in hits):.2f}")
    hits.sort(key=lambda t: -t[0])
    cand = hits[: max(top_k * 4, top_k)]
    obj = ClassObjective(detector, class_idx, min_area=min_area, max_area=max_area, margin=0.0, n_aug=0)
    seeds = []
    for s, p in cand:
        im = Image.open(p).convert("RGB")
        rep = obj.report(_to01(im).to(detector.device))
        seed_im = _square_crop(im, rep["box"], size=size) if crop else im.resize((size, size))
        seeds.append({"path": p, "image": seed_im, "score": s, "box": rep["box"], "robust": s})
    if robust and seeds:
        ro = RobustObjective(detector, class_idx, seed=0)
        T = ro.evaluate([d["image"] for d in seeds])
        for d, r in zip(seeds, T["total"].tolist()):
            d["robust"] = float(r)
        seeds.sort(key=lambda d: -d["robust"])
    return seeds[:top_k]


def _to01(im: Image.Image) -> torch.Tensor:
    import numpy as np
    return torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1).float() / 255.0
