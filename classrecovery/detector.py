"""Differentiable wrapper around an Ultralytics YOLO detector.

The wrapper exposes the *pre-NMS* output of the network so that a class score
can be back-propagated to the input image (and from there into a diffusion
model's latent).  NMS is not differentiable, so everything gradient-related
lives above it; ordinary post-NMS detection is still available through
:meth:`Detector.detect` for evaluation and display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _to_tensor01(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """PIL / HWC uint8 array / CHW float tensor -> (3,H,W) float in [0,1]."""
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("RGB"))
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).float() / 255.0
        return t
    if torch.is_tensor(img):
        t = img.float()
        if t.max() > 1.5:
            t = t / 255.0
        return t
    raise TypeError(f"unsupported image type {type(img)}")


class Detector:
    """A YOLO checkpoint with a differentiable forward pass.

    Parameters
    ----------
    weights : path to a ``.pt`` Ultralytics checkpoint (v8/v9/v10/v11 style
        heads that emit ``(B, 4+nc, N)`` all work).
    imgsz : square input size used for the differentiable forward pass.
    device : torch device (defaults to CUDA if available).
    """

    def __init__(self, weights: str, imgsz: int = 640, device: Optional[str] = None):
        from ultralytics import YOLO

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.yolo = YOLO(weights)
        self.net = self.yolo.model.to(self.device).eval().float()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.imgsz = int(math.ceil(imgsz / 32) * 32)
        self.nc: int = int(getattr(self.net, "nc", None) or len(self.net.names))
        names = getattr(self.net, "names", None) or {}
        self.names: Dict[int, str] = {int(k): str(v) for k, v in dict(names).items()}
        self.weights_path = weights

    # ------------------------------------------------------------------ raw
    def raw(self, img01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable pre-NMS forward.

        img01 : (B,3,H,W) float in [0,1], any size (resized to ``imgsz``).
        Returns ``boxes`` (B,N,4) in xyxy *normalised to [0,1]* and ``scores``
        (B,N,nc) after sigmoid.
        """
        if img01.dim() == 3:
            img01 = img01[None]
        x = img01.to(self.device)
        if x.shape[-2:] != (self.imgsz, self.imgsz):
            x = F.interpolate(x, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False)
        out = self.net(x)
        y = out[0] if isinstance(out, (list, tuple)) else out
        if y.dim() == 3 and y.shape[1] == 4 + self.nc:
            y = y.transpose(1, 2)  # (B,N,4+nc)
        elif y.dim() == 3 and y.shape[2] == 4 + self.nc:
            pass
        else:  # v10-style end-to-end heads return (B,N,6); fall back to one2many if present
            raise RuntimeError(f"unexpected head output shape {tuple(y.shape)}; nc={self.nc}")
        xywh, scores = y[..., :4], y[..., 4:]
        cx, cy, w, h = xywh.unbind(-1)
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) / self.imgsz
        return boxes, scores

    # --------------------------------------------------------------- detect
    @torch.no_grad()
    def detect(self, images: Sequence[Union[Image.Image, np.ndarray]], conf: float = 0.25, iou: float = 0.6):
        """Ordinary post-NMS detection through Ultralytics' predictor.

        Returns a list (one per image) of dicts with ``boxes`` (n,4 xyxy px),
        ``cls`` (n,), ``conf`` (n,).
        """
        results = self.yolo.predict(list(images), conf=conf, iou=iou, imgsz=self.imgsz,
                                    device=self.device.type if self.device.type != "cuda" else 0,
                                    verbose=False)
        out = []
        for r in results:
            b = r.boxes
            out.append({
                "boxes": b.xyxy.cpu().numpy() if b is not None else np.zeros((0, 4)),
                "cls": b.cls.cpu().numpy().astype(int) if b is not None else np.zeros((0,), int),
                "conf": b.conf.cpu().numpy() if b is not None else np.zeros((0,)),
            })
        return out

    @torch.no_grad()
    def class_scores(self, images: Sequence[Union[Image.Image, np.ndarray, torch.Tensor]],
                     batch: int = 16, min_area: float = 0.0, max_area: float = 1.0) -> torch.Tensor:
        """Max pre-NMS score per class for each image -> (n_images, nc) on CPU."""
        rows = []
        for i in range(0, len(images), batch):
            x = torch.stack([_to_tensor01(im) for im in images[i:i + batch]])
            x = F.interpolate(x, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False)
            boxes, scores = self.raw(x)
            area = (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
            ok = (area >= min_area) & (area <= max_area)
            scores = scores.masked_fill(~ok[..., None], 0.0)
            rows.append(scores.amax(1).cpu())
        return torch.cat(rows) if rows else torch.zeros(0, self.nc)

    # ---------------------------------------------------------------- head
    def class_head_weights(self) -> List[torch.Tensor]:
        """Per-detection-scale class-head weight matrices, each (nc, C).

        For YOLOv8/9/11 these are the final 1x1 convs of ``Detect.cv3``.  For
        other heads we fall back to any conv whose output dim equals ``nc``.
        """
        head = self.net.model[-1]
        mats: List[torch.Tensor] = []
        cv3 = getattr(head, "cv3", None)
        if cv3 is not None:
            for seq in cv3:
                last = None
                for m in seq.modules():
                    if isinstance(m, torch.nn.Conv2d) and m.out_channels == self.nc:
                        last = m
                if last is not None:
                    mats.append(last.weight.detach().flatten(1).cpu())
        if not mats:
            for m in head.modules():
                if isinstance(m, torch.nn.Conv2d) and m.out_channels == self.nc:
                    mats.append(m.weight.detach().flatten(1).cpu())
        return mats

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.net.state_dict().items()}


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _random_crop_params(gen: torch.Generator, scale=(0.6, 1.0)) -> Tuple[float, float, float, float]:
    s = float(torch.empty(1).uniform_(*scale, generator=gen))
    x0 = float(torch.empty(1).uniform_(0, 1 - s, generator=gen))
    y0 = float(torch.empty(1).uniform_(0, 1 - s, generator=gen))
    return x0, y0, s, s


def augment(img01: torch.Tensor, n: int, size: int, gen: Optional[torch.Generator] = None,
            scale=(0.6, 1.0), flip: bool = True) -> torch.Tensor:
    """Random-resized-crop + flip, differentiable w.r.t. ``img01``.  (B,3,H,W) -> (B*n,3,size,size)."""
    gen = gen or torch.Generator().manual_seed(int(torch.randint(0, 2**31, (1,))))
    B, _, H, W = img01.shape
    outs = []
    for _ in range(n):
        x0, y0, sw, sh = _random_crop_params(gen, scale)
        # build an affine grid for the crop (keeps everything differentiable)
        theta = torch.tensor([[sw, 0.0, (2 * x0 + sw) - 1], [0.0, sh, (2 * y0 + sh) - 1]],
                             device=img01.device, dtype=img01.dtype)
        if flip and torch.rand(1, generator=gen).item() < 0.5:
            theta[0, 0] = -theta[0, 0]
        grid = F.affine_grid(theta[None].expand(B, -1, -1), (B, 3, size, size), align_corners=False)
        outs.append(F.grid_sample(img01, grid, mode="bilinear", padding_mode="reflection", align_corners=False))
    return torch.cat(outs, 0)


@dataclass
class ClassObjective:
    """Scalar objective "image contains an object of class ``class_idx``".

    score(img) = soft-max over anchors of  p_k(anchor)   [optionally minus the
    best competing class, to reward specificity], restricted to anchors whose
    predicted box has a reasonable area.  Larger is better; ``loss`` returns
    the negative for minimisation.

    Parameters
    ----------
    detector : Detector
    class_idx : target class index
    min_area, max_area : accepted box area as a fraction of the image; keeps the
        optimiser from painting a screen-filling texture or a 3-pixel speck.
    tau : temperature of the soft-max over anchors (0 -> hard max).
    margin : weight on the ``p_k - max_{j!=k} p_j`` specificity term.
    n_aug : number of random crops/flips averaged per image (the classic
        feature-visualisation trick for robustness to adversarial texture).
    """

    detector: Detector
    class_idx: int
    min_area: float = 0.02
    max_area: float = 0.85
    tau: float = 0.05
    margin: float = 0.5
    n_aug: int = 2
    aug_scale: Tuple[float, float] = (0.6, 1.0)
    seed: Optional[int] = None
    _gen: torch.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._gen = torch.Generator().manual_seed(self.seed if self.seed is not None else 0)

    def anchor_scores(self, img01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes, scores = self.detector.raw(img01)
        area = (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)
        ok = (area >= self.min_area) & (area <= self.max_area)
        p_k = scores[..., self.class_idx]
        if self.margin > 0 and scores.shape[-1] > 1:
            others = scores.clone()
            others[..., self.class_idx] = -1.0
            p_k = p_k - self.margin * others.amax(-1)
        p_k = p_k.masked_fill(~ok, -1.0)
        return p_k, boxes, scores

    def score(self, img01: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B,) differentiable score (higher = more class k)."""
        if self.n_aug > 0:
            x = augment(img01, self.n_aug, self.detector.imgsz, self._gen, self.aug_scale)
        else:
            x = img01
        p_k, _, _ = self.anchor_scores(x)
        if self.tau > 0:  # soft max over anchors; the log(N) offset is removed so 1.0 still means "certain"
            s = self.tau * (torch.logsumexp(p_k / self.tau, dim=-1) - math.log(p_k.shape[-1]))
        else:
            s = p_k.amax(-1)
        if self.n_aug > 0:
            s = s.view(self.n_aug, img01.shape[0]).mean(0)
        return s

    def loss(self, img01: torch.Tensor) -> torch.Tensor:
        return -self.score(img01).sum()

    @torch.no_grad()
    def report(self, img01: torch.Tensor) -> Dict[str, float]:
        """Non-augmented diagnostics for a single image."""
        p_k, boxes, scores = self.anchor_scores(img01[None] if img01.dim() == 3 else img01)
        i = int(p_k[0].argmax())
        return {
            "target_score": float(scores[0, i, self.class_idx]),
            "best_other_idx": int(scores[0, i].argsort(descending=True)[1]) if scores.shape[-1] > 1 else -1,
            "best_other_score": float(scores[0, i].sort(descending=True).values[1]) if scores.shape[-1] > 1 else 0.0,
            "box": [float(v) for v in boxes[0, i]],
        }
