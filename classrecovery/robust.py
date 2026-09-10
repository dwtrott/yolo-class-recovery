"""Robust objective + prototype discovery: "find a stable visual concept that
repeatedly explains class k", instead of "maximise class k".

Terms (all differentiable, all label-free):

* class          soft-max class score, averaged over augmentations
* consistency    -Var of the score across augmentations (crop/flip/rotate/colour)
* box            spread of the best box across augmentations, mapped back to the
                 original frame (a real object has one stable location)
* localization   the score must COLLAPSE when the best box is masked out — this is
                 the term a wall-to-wall texture cannot satisfy, because a texture
                 has evidence everywhere and no single box to lose
* cutout         the score must SURVIVE small random erasures inside the box (no
                 single patch may carry all the evidence)
* prototype      distance of multi-layer ROI features to a pseudo-prototype found
                 by clustering the detector's own activations on earlier candidates

``discover_prototype`` runs the candidate → robust-score → cluster loop and
returns the dominant mode; ``represent_robust`` chains everything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .detector import ClassObjective, Detector
from .prior import GuidanceConfig, UncondPrior


# --------------------------------------------------------------------- augmentations
def _affine_theta(gen: torch.Generator, scale=(0.6, 1.0), max_rot_deg: float = 10.0, flip: bool = True) -> torch.Tensor:
    s = float(torch.empty(1).uniform_(*scale, generator=gen))
    cx = float(torch.empty(1).uniform_(-(1 - s), (1 - s), generator=gen))
    cy = float(torch.empty(1).uniform_(-(1 - s), (1 - s), generator=gen))
    a = math.radians(float(torch.empty(1).uniform_(-max_rot_deg, max_rot_deg, generator=gen)))
    fx = -1.0 if (flip and torch.rand(1, generator=gen).item() < 0.5) else 1.0
    A = torch.tensor([[s * math.cos(a) * fx, -s * math.sin(a)], [s * math.sin(a) * fx, s * math.cos(a)]])
    return torch.cat([A, torch.tensor([[cx], [cy]])], 1)  # (2,3): output coords u -> input coords A u + t


def _photometric(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    b = float(torch.empty(1).uniform_(-0.15, 0.15, generator=gen))
    c = float(torch.empty(1).uniform_(0.8, 1.2, generator=gen))
    x = ((x - 0.5) * c + 0.5 + b)
    if torch.rand(1, generator=gen).item() < 0.2:
        x = x.mean(1, keepdim=True).expand_as(x)
    return x.clamp(0, 1)


def augment_with_thetas(img01: torch.Tensor, n: int, size: int, gen: torch.Generator, **kw) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    B = img01.shape[0]
    outs, thetas = [], []
    for _ in range(n):
        theta = _affine_theta(gen, **kw).to(img01.device, img01.dtype)
        grid = F.affine_grid(theta[None].expand(B, -1, -1), (B, 3, size, size), align_corners=False)
        x = F.grid_sample(img01, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        outs.append(_photometric(x, gen))
        thetas.append(theta)
    return torch.cat(outs, 0), thetas


def _box_to_original(box01: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """xyxy in [0,1] of the augmented frame -> xyxy in [0,1] of the original frame (bounding box of corners)."""
    x0, y0, x1, y1 = box01.unbind(-1)
    corners = torch.stack([torch.stack([x0, y0], -1), torch.stack([x1, y0], -1),
                           torch.stack([x0, y1], -1), torch.stack([x1, y1], -1)], -2) * 2 - 1     # (B,4,2)
    A, t = theta[:, :2], theta[:, 2]
    p = corners @ A.T + t                                                                          # (B,4,2)
    p = (p + 1) / 2
    return torch.cat([p.amin(-2), p.amax(-2)], -1).clamp(0, 1)


# --------------------------------------------------------------------- features
class FeatureTap:
    """Forward hooks on a few internal layers of the YOLO net (ultralytics ``model.model[i]``)."""

    def __init__(self, detector: Detector, layers: Sequence[int] = (4, 6, 9)):
        self.det = detector
        self.layers = [i for i in layers if i < len(detector.net.model)]
        self.out: Dict[int, torch.Tensor] = {}
        self._h = [detector.net.model[i].register_forward_hook(self._mk(i)) for i in self.layers]

    def _mk(self, i):
        def hook(_m, _inp, o):
            self.out[i] = o
        return hook

    def roi_features(self, boxes01: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Average-pool each tapped feature map inside ``boxes01`` (B,4 xyxy in [0,1]) -> {layer: (B,C)}, L2-normalised."""
        feats = {}
        for i in self.layers:
            f = self.out[i]
            B, C, H, W = f.shape
            ys = torch.arange(H, device=f.device).float().add(0.5).div(H)
            xs = torch.arange(W, device=f.device).float().add(0.5).div(W)
            inside = ((ys[None, :, None] >= boxes01[:, 1, None, None]) & (ys[None, :, None] <= boxes01[:, 3, None, None]) &
                      (xs[None, None, :] >= boxes01[:, 0, None, None]) & (xs[None, None, :] <= boxes01[:, 2, None, None])).float()
            inside = inside + 1e-6                                                # never empty
            pooled = (f * inside[:, None]).sum((2, 3)) / inside.sum((1, 2))[:, None]
            feats[i] = F.normalize(pooled, dim=-1)
        return feats

    def close(self):
        for h in self._h:
            h.remove()


# --------------------------------------------------------------------- objective
@dataclass
class RobustObjective:
    detector: Detector
    class_idx: int
    n_aug: int = 4
    w_class: float = 1.0
    w_cons: float = 0.5
    w_box: float = 0.5
    w_loc: float = 1.0
    w_cut: float = 0.5
    w_proto: float = 0.0
    prototype: Optional[Dict[int, torch.Tensor]] = None       # {layer: (C,)}
    layers: Sequence[int] = (4, 6, 9)
    n_cut: int = 2
    cut_frac: float = 0.25
    min_area: float = 0.03
    max_area: float = 0.8
    margin: float = 0.5
    tau: float = 0.05
    seed: int = 0
    _gen: torch.Generator = field(init=False, repr=False)
    _tap: Optional[FeatureTap] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self._gen = torch.Generator().manual_seed(self.seed)
        self._base = ClassObjective(self.detector, self.class_idx, min_area=self.min_area, max_area=self.max_area,
                                    tau=self.tau, margin=self.margin, n_aug=0)
        if self.w_proto > 0 and self.prototype is not None:
            self._tap = FeatureTap(self.detector, self.layers)

    # soft score + best box for a batch (no augmentation)
    def _score_box(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p_k, boxes, _ = self._base.anchor_scores(x)
        s = self.tau * (torch.logsumexp(p_k / self.tau, dim=-1) - math.log(p_k.shape[-1]))
        w = torch.softmax(p_k / self.tau, dim=-1)                                # (B,N) attention over anchors
        box = (w[..., None] * boxes).sum(1)                                      # soft best box (B,4), differentiable
        return s, box

    def terms(self, img01: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = img01.shape[0]
        size = self.detector.imgsz
        T: Dict[str, torch.Tensor] = {}

        # --- class + consistency + box stability over augmentations
        xa, thetas = augment_with_thetas(img01, self.n_aug, size, self._gen)
        s_a, box_a = self._score_box(xa)
        s_a, box_a = s_a.view(self.n_aug, B), box_a.view(self.n_aug, B, 4)
        T["class"] = s_a.mean(0)
        T["consistency"] = -s_a.var(0, unbiased=False) if self.n_aug > 1 else torch.zeros(B, device=img01.device)
        boxes_orig = torch.stack([_box_to_original(box_a[a], thetas[a]) for a in range(self.n_aug)])   # (n_aug,B,4)
        T["box"] = -(boxes_orig - boxes_orig.mean(0, keepdim=True)).abs().mean((0, 2))

        # --- reference pass on the plain image: best box, features
        s0, box0 = self._score_box(img01)
        T["plain"] = s0
        if self._tap is not None:
            feats = self._tap.roi_features(box0.detach())
            d = torch.zeros(B, device=img01.device)
            for l, mu in self.prototype.items():
                if l in feats:
                    d = d + (1 - (feats[l] * mu.to(feats[l].device)[None]).sum(-1))     # cosine distance
            T["prototype"] = -d / max(1, len(self.prototype))
        else:
            T["prototype"] = torch.zeros(B, device=img01.device)

        # --- localization: mask the best box out -> score should collapse
        H, W = img01.shape[-2:]
        ys = torch.arange(H, device=img01.device).float().add(0.5).div(H)
        xs = torch.arange(W, device=img01.device).float().add(0.5).div(W)
        bb = box0.detach()
        inside = ((ys[None, :, None] >= bb[:, 1, None, None]) & (ys[None, :, None] <= bb[:, 3, None, None]) &
                  (xs[None, None, :] >= bb[:, 0, None, None]) & (xs[None, None, :] <= bb[:, 2, None, None])).float()[:, None]
        fill = img01.mean((2, 3), keepdim=True)
        masked = img01 * (1 - inside) + fill * inside
        s_masked, _ = self._score_box(masked)
        T["localization"] = -F.relu(s_masked)                                     # evidence outside the box is penalised

        # --- cutout survival: small erasures inside the box must not kill the score
        cut_imgs = []
        for _ in range(self.n_cut):
            bw, bh = (bb[:, 2] - bb[:, 0]), (bb[:, 3] - bb[:, 1])
            cw, ch = bw * self.cut_frac, bh * self.cut_frac
            cx = bb[:, 0] + torch.rand(B, generator=self._gen).to(bb.device) * (bw - cw)
            cy = bb[:, 1] + torch.rand(B, generator=self._gen).to(bb.device) * (bh - ch)
            m = ((ys[None, :, None] >= cy[:, None, None]) & (ys[None, :, None] <= (cy + ch)[:, None, None]) &
                 (xs[None, None, :] >= cx[:, None, None]) & (xs[None, None, :] <= (cx + cw)[:, None, None])).float()[:, None]
            cut_imgs.append(img01 * (1 - m) + fill * m)
        s_cut, _ = self._score_box(torch.cat(cut_imgs, 0))
        T["cutout"] = -F.relu(s0.detach() - s_cut.view(self.n_cut, B)).mean(0)   # penalise the drop

        T["total"] = (self.w_class * T["class"] + self.w_cons * T["consistency"] + self.w_box * T["box"] +
                      self.w_loc * T["localization"] + self.w_cut * T["cutout"] + self.w_proto * T["prototype"])
        T["best_box"] = box0.detach()
        return T

    def score(self, img01: torch.Tensor) -> torch.Tensor:      # guided_sample-compatible
        return self.terms(img01)["total"]

    def loss(self, img01: torch.Tensor) -> torch.Tensor:
        return -self.score(img01).sum()

    @torch.no_grad()
    def evaluate(self, images: Sequence[Image.Image]) -> Dict[str, torch.Tensor]:
        from .detector import _to_tensor01
        sz = (self.detector.imgsz,) * 2
        x = torch.stack([F.interpolate(_to_tensor01(im)[None], size=sz, mode="bilinear", align_corners=False)[0]
                         for im in images]).to(self.detector.device)
        return {k: v.cpu() for k, v in self.terms(x).items()}

    def close(self):
        if self._tap is not None:
            self._tap.close()


# --------------------------------------------------------------------- prototype discovery
def _kmeans(X: torch.Tensor, k: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    k = min(k, X.shape[0])
    C = X[torch.randperm(X.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, C).argmin(1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(0)
    return torch.cdist(X, C).argmin(1)


@dataclass
class Mode:
    members: List[int]
    robust: float
    images: List[Image.Image]
    prototype: Dict[int, torch.Tensor]


def discover_prototype(detector: Detector, prior: UncondPrior, class_idx: int, n_candidates: int = 24,
                       batch: int = 4, gcfg: Optional[GuidanceConfig] = None, layers: Sequence[int] = (4, 6, 9),
                       k: int = 3, keep_frac: float = 0.5, seed: int = 0, candidates: Optional[List[Image.Image]] = None,
                       progress: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """Score candidates for ``class_idx`` robustly, cluster their internal activations, and return
    the dominant mode as a pseudo-prototype.  Candidates are generated from noise unless supplied
    (e.g. mined seeds + refinements).  Label-free."""
    gcfg = gcfg or GuidanceConfig(steps=25, strength=0.1, n_images=batch, seed=seed)
    robust = RobustObjective(detector, class_idx, seed=seed)
    imgs: List[Image.Image] = list(candidates) if candidates else []
    guide_obj = ClassObjective(detector, class_idx, n_aug=2, seed=seed)
    for b in range(len(imgs), n_candidates, batch):
        cfg = GuidanceConfig(**{**gcfg.__dict__, "n_images": min(batch, n_candidates - b), "seed": seed + 1000 * b})
        if progress:
            progress(f"   candidates {b + cfg.n_images}/{n_candidates}")
        imgs += prior.guided_sample(guide_obj, cfg)["images"]
    T = robust.evaluate(imgs)
    rs = T["total"]
    keep = rs.argsort(descending=True)[: max(2, int(len(imgs) * keep_frac))]
    tap = FeatureTap(detector, layers)
    with torch.no_grad():
        from .detector import _to_tensor01
        sz = (detector.imgsz,) * 2
        x = torch.stack([F.interpolate(_to_tensor01(imgs[i])[None], size=sz, mode="bilinear", align_corners=False)[0]
                         for i in keep.tolist()]).to(detector.device)
        detector.raw(x)
        feats = tap.roi_features(T["best_box"][keep].to(detector.device))
    tap.close()
    X = torch.cat([feats[l].cpu() for l in tap.layers], 1)
    assign = _kmeans(X, k, seed=seed)
    modes: List[Mode] = []
    for j in range(int(assign.max()) + 1):
        idx = [int(keep[i]) for i in range(len(keep)) if assign[i] == j]
        if not idx:
            continue
        proto = {l: feats[l].cpu()[assign == j].mean(0) for l in tap.layers}
        proto = {l: F.normalize(v, dim=0) for l, v in proto.items()}
        modes.append(Mode(idx, float(rs[idx].mean()), [imgs[i] for i in idx], proto))
    modes.sort(key=lambda m: -m.robust)
    return {"modes": modes, "candidates": imgs, "scores": T, "dominant": modes[0]}


# --------------------------------------------------------------------- degradation-robust score
def _jpeg(img01: torch.Tensor, quality: int = 40) -> torch.Tensor:
    import io
    from PIL import Image as _I
    out = []
    for x in img01:
        buf = io.BytesIO()
        _I.fromarray((x.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()).save(buf, "JPEG", quality=quality)
        buf.seek(0)
        import numpy as np
        y = torch.from_numpy(np.array(_I.open(buf).convert("RGB"), copy=True)).permute(2, 0, 1).float() / 255
        out.append(y.to(img01.device))
    return torch.stack(out)


def _blur(img01: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=img01.device).float() - (k - 1) / 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].expand(3, 1, k, k)
    return F.conv2d(F.pad(img01, (k // 2,) * 4, mode="reflect"), w, groups=3)


DEGRADATIONS = {
    "noise0.02": lambda x, g: (x + 0.02 * torch.randn(x.shape, generator=g, device=x.device)).clamp(0, 1),
    "noise0.04": lambda x, g: (x + 0.04 * torch.randn(x.shape, generator=g, device=x.device)).clamp(0, 1),
    "blur1.5": lambda x, g: _blur(x, 1.5),
    "jpeg50": lambda x, g: _jpeg(x, 50),
    "half-res": lambda x, g: F.interpolate(F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False),
                                            size=x.shape[-2:], mode="bilinear", align_corners=False),
}


@torch.no_grad()
def degradation_scores(detector: Detector, class_idx: int, images: Sequence[Image.Image], seed: int = 0,
                       min_area: float = 0.03, max_area: float = 0.8) -> Dict[str, torch.Tensor]:
    """Detector class score for each image under each degradation, plus ``min`` across them.

    Adversarial patterns collapse under mild noise / blur / JPEG; real objects do not.
    Ranking by the minimum is the simplest label-free defence against "det=1.00 speckle".
    """
    from .detector import _to_tensor01
    g = torch.Generator(device=detector.device).manual_seed(seed)
    sz = (detector.imgsz,) * 2
    x = torch.stack([F.interpolate(_to_tensor01(im)[None], size=sz, mode="bilinear", align_corners=False)[0]
                     for im in images]).to(detector.device)
    obj = ClassObjective(detector, class_idx, min_area=min_area, max_area=max_area, margin=0.0, n_aug=0)

    def score(xx):
        p_k, _, _ = obj.anchor_scores(xx)
        return p_k.amax(-1).clamp(min=0).cpu()

    out = {"clean": score(x)}
    for name, fn in DEGRADATIONS.items():
        out[name] = score(fn(x, g))
    out["min"] = torch.stack([out[k] for k in DEGRADATIONS]).amin(0)
    out["mean"] = torch.stack([out[k] for k in DEGRADATIONS]).mean(0)
    return out
