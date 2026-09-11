"""Noise-robust twin of the detector, self-distilled without labels.

Why: a standard detector's input gradient is texture-shaped, so using it to
steer a diffusion prior paints stripes instead of zebras.  Robust models have
perceptually aligned gradients (Santurkar et al. 2019); the 2021 classifier-
guidance result relied on a classifier trained on *noisy* images for the same
reason.  We get that property with no labels: copy the detector, and train the
copy to reproduce the ORIGINAL detector's predictions on clean pool images
while itself seeing noised / blurred / jittered versions.  The twin then
supplies the guidance gradient; the original detector still does all scoring.

    twin = train_twin(detector, pool_images, steps=1500)      # ~10 min on an A100
    represent(..., twin=twin)
"""

from __future__ import annotations

import copy
import os
import random
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from .detector import Detector


def _load_batch(paths: Sequence[str], size: int, device) -> torch.Tensor:
    import numpy as np
    xs = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            im = Image.new("RGB", (size, size), (114, 114, 114))
        # random-resized square crop for variety
        W, H = im.size
        s = random.uniform(0.6, 1.0) * min(W, H)
        l, t = random.uniform(0, W - s), random.uniform(0, H - s)
        im = im.crop((int(l), int(t), int(l + s), int(t + s))).resize((size, size))
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        xs.append(torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1).float() / 255)
    return torch.stack(xs).to(device)


def corrupt(x: torch.Tensor, max_sigma: float = 0.25) -> torch.Tensor:
    """Random Gaussian noise (the main one), sometimes blur, brightness/contrast, grayscale."""
    B = x.shape[0]
    sigma = torch.rand(B, 1, 1, 1, device=x.device) * max_sigma
    y = x + sigma * torch.randn_like(x)
    if random.random() < 0.3:
        k = 5
        w = torch.ones(3, 1, k, k, device=x.device) / (k * k)
        y = F.conv2d(F.pad(y, (k // 2,) * 4, mode="reflect"), w, groups=3)
    if random.random() < 0.5:
        c = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
        b = torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.15, 0.15)
        y = (y - 0.5) * c + 0.5 + b
    if random.random() < 0.15:
        y = y.mean(1, keepdim=True).expand_as(y)
    return y.clamp(0, 1)


def train_twin(detector: Detector, pool_images: Sequence[str], steps: int = 1500, batch: int = 16, lr: float = 3e-5,
               max_sigma: float = 0.25, size: Optional[int] = None, save_path: Optional[str] = None,
               pos_weight: float = 20.0, clean_frac: float = 0.25, adversarial: bool = False,
               eps: float = 4 / 255, pgd_steps: int = 3,
               progress: Optional[Callable[[str], None]] = print) -> Detector:
    """Self-distil a noise-robust copy of ``detector`` on unlabeled ``pool_images``.

    Loss: BCE between the twin's class probabilities on the corrupted image and the teacher's
    on the clean image, with anchors the teacher is confident about up-weighted (``pos_weight``)
    so the thousands of empty anchors do not drown the signal, plus L1 on box coordinates
    weighted by the teacher's confidence.  A ``clean_frac`` share of each batch is left
    uncorrupted so the twin keeps the teacher's clean-image behaviour.  No labels anywhere.

    ``adversarial=True``: instead of random noise, each batch is perturbed by ``pgd_steps`` of
    projected gradient ascent on the distillation loss (L-inf budget ``eps``) — the perturbation
    that most breaks agreement with the teacher — and the twin is trained to agree anyway.
    Adversarial training is what produces perceptually aligned gradients (Santurkar et al.
    2019); Gaussian noise alone barely changes them.  ~3x the cost.
    """
    size = size or detector.imgsz
    if adversarial and clean_frac < 0.5:
        clean_frac = 0.5                                  # adversarial training erodes clean behaviour; anchor it
    teacher = detector.net
    twin = copy.deepcopy(teacher).to(detector.device).float().eval()   # eval: BN uses running stats
    trainable = []
    for n, p in twin.named_parameters():
        frozen = "dfl" in n                                # the DFL box-decoding conv is a fixed arange: never train it
        p.requires_grad_(not frozen)
        if not frozen:
            trainable.append(p)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    paths = list(pool_images)
    nc = detector.nc

    def raw(net, x):
        out = net(x)
        y = out[0] if isinstance(out, (list, tuple)) else out
        if y.shape[1] == 4 + nc:
            y = y.transpose(1, 2)
        return y[..., :4], y[..., 4:]                      # boxes (B,N,4) xywh px, probs (B,N,nc)

    def distill_cls(sp, tp):
        wcls = 1.0 + pos_weight * tp                        # up-weight anchors/classes the teacher believes in
        return (wcls * F.binary_cross_entropy(sp.clamp(1e-6, 1 - 1e-6), tp, reduction="none")).sum() / wcls.sum()

    for it in range(steps):
        x = _load_batch(random.sample(paths, min(batch, len(paths))), size, detector.device)
        with torch.no_grad():
            tb, tp = raw(teacher, x)
        n_clean = int(round(clean_frac * x.shape[0]))
        if adversarial:
            # PGD on the distillation loss: the perturbation that most breaks agreement with the teacher
            delta = (torch.rand_like(x) * 2 - 1) * eps
            alpha = 2.5 * eps / pgd_steps
            for _ in range(pgd_steps):
                delta.requires_grad_(True)
                _, sp_adv = raw(twin, (x + delta).clamp(0, 1))
                g = torch.autograd.grad(distill_cls(sp_adv, tp), delta)[0]
                delta = (delta.detach() + alpha * g.sign()).clamp(-eps, eps)
            xin = (x + delta.detach()).clamp(0, 1)
            if random.random() < 0.5:                       # keep some plain-noise robustness too
                xin = corrupt(xin, max_sigma * 0.5)
        else:
            xin = corrupt(x, max_sigma)
        if n_clean:
            xin = torch.cat([x[:n_clean], xin[n_clean:]])
        sb, sp = raw(twin, xin)
        cls_loss = distill_cls(sp, tp)
        w = tp.amax(-1, keepdim=True)                       # only care about boxes where the teacher sees something
        box_loss = (w * (sb - tb).abs() / size).sum() / (w.sum() * 4 + 1e-6)
        loss = cls_loss + 0.5 * box_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(twin.parameters(), 5.0)
        opt.step()
        if progress and (it % 100 == 0 or it == steps - 1):
            progress(f"   twin step {it:>5}/{steps}  cls {cls_loss.item():.4f}  box {box_loss.item():.4f}")

    for p in twin.parameters():
        p.requires_grad_(False)
    twin_det = copy.copy(detector)                          # shares yolo/names/imgsz; swaps the network
    twin_det.net = twin
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(twin.state_dict(), save_path)
    return twin_det


def load_twin(detector: Detector, path: str) -> Detector:
    twin = copy.deepcopy(detector.net).to(detector.device).float().eval()
    twin.load_state_dict(torch.load(path, map_location=detector.device))
    for p in twin.parameters():
        p.requires_grad_(False)
    twin_det = copy.copy(detector)
    twin_det.net = twin
    return twin_det


@torch.no_grad()
def gradient_alignment_check(detector: Detector, twin: Detector, image: Image.Image, class_idx: int) -> dict:
    """How much high-frequency energy is in each model's input gradient (lower = more object-shaped)."""
    from .detector import ClassObjective, _to_tensor01
    out = {}
    for name, d in (("original", detector), ("twin", twin)):
        x = F.interpolate(_to_tensor01(image)[None], size=(d.imgsz,) * 2, mode="bilinear", align_corners=False).to(d.device)
        with torch.enable_grad():
            x = x.requires_grad_(True)
            ClassObjective(d, class_idx, n_aug=0, margin=0.0).loss(x).backward()
            g = x.grad[0].mean(0)
        gf = torch.fft.fftshift(torch.fft.fft2(g)).abs()
        H, W = gf.shape
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        r = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2).sqrt().to(gf.device)
        hi = gf[r > min(H, W) / 8].sum() / gf.sum()
        out[name] = float(hi)
    return out
