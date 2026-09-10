"""Unconditional diffusion prior + detector-guided sampling.

The prior is OpenAI's unconditional 256px ImageNet diffusion model (Dhariwal &
Nichol 2021; network vendored in ``classrecovery.gd``).  It has never seen a
caption.  The only steering it receives is the gradient of an objective built
from the detector's activations — classifier guidance in the original sense.

Two ways to sample:

* ``guided_sample``            from pure noise
* ``guided_refine``            from a real seed image, SDEdit-style: noise the
                               seed to a chosen level and denoise under guidance,
                               so the prior only has to *sharpen* an object it is
                               already sitting on rather than invent one

Both push the gradient back *through the diffusion network* into the noisy
state, and both can average that gradient over noisy copies of the decoded
image (SmoothGrad) — the single cheapest way to stop a texture-biased
detector from painting speckle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

_MODELS = {
    "imagenet-256-uncond": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt", 256,
        dict(num_channels=256, num_res_blocks=2, attention_resolutions="32,16,8", num_heads=4,
             num_head_channels=64, use_scale_shift_norm=True, resblock_updown=True, dropout=0.0, learn_sigma=True)),
}


def _build_unet(image_size, num_channels, num_res_blocks, attention_resolutions, num_heads, num_head_channels,
                use_scale_shift_norm, resblock_updown, dropout, learn_sigma, use_fp16, use_checkpoint):
    from .gd import UNetModel
    channel_mult = {512: (0.5, 1, 1, 2, 2, 4, 4), 256: (1, 1, 2, 2, 4, 4), 128: (1, 1, 2, 3, 4), 64: (1, 2, 3, 4)}[image_size]
    attn = tuple(image_size // int(r) for r in attention_resolutions.split(","))
    return UNetModel(image_size=image_size, in_channels=3, model_channels=num_channels,
                     out_channels=(6 if learn_sigma else 3), num_res_blocks=num_res_blocks,
                     attention_resolutions=attn, dropout=dropout, channel_mult=channel_mult, num_classes=None,
                     use_checkpoint=use_checkpoint, use_fp16=use_fp16, num_heads=num_heads,
                     num_head_channels=num_head_channels, num_heads_upsample=-1,
                     use_scale_shift_norm=use_scale_shift_norm, resblock_updown=resblock_updown,
                     use_new_attention_order=False)


def _download(url: str, cache_dir: Optional[str] = None) -> str:
    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "classrecovery")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(path):
        print(f"[classrecovery] downloading {url} -> {path}")
        import urllib.request
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def to_pil(x: torch.Tensor) -> List[Image.Image]:
    x = (x.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in x]


def to_tensor01(images: Sequence[Image.Image], size: int, device) -> torch.Tensor:
    import numpy as np
    x = torch.stack([torch.from_numpy(np.array(im.convert("RGB").resize((size, size)), copy=True)).permute(2, 0, 1)
                     for im in images]).float().div(255)
    return x.to(device)


@dataclass
class GuidanceConfig:
    steps: int = 50            # denoising steps for a from-noise run
    strength: float = 0.1      # guidance step as a fraction of ||z_t|| (per gradient step)
    repeats: int = 1           # gradient steps per denoising step
    guide_from: float = 1.0    # guide while guide_to <= t/T <= guide_from
    guide_to: float = 0.15
    smooth_k: int = 4          # SmoothGrad: average the objective over k noisy copies (1 = off)
    smooth_sigma: float = 0.08 # std of that noise in [0,1] image units
    n_images: int = 4
    seed: Optional[int] = None
    refine_noise: float = 0.5  # guided_refine: fraction of the noise schedule to re-noise the seed to


class UncondPrior:
    """Pixel-space unconditional diffusion model with classifier-guided DDIM sampling."""

    def __init__(self, name: str = "imagenet-256-uncond", device: Optional[str] = None, fp16: Optional[bool] = None,
                 weights: Optional[str] = None):
        from diffusers import DDIMScheduler
        url, size, kw = _MODELS[name]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        fp16 = (self.device.type == "cuda") if fp16 is None else fp16
        self.name, self.image_size = name, size
        model = _build_unet(image_size=size, use_checkpoint=True, use_fp16=fp16, **kw)
        model.load_state_dict(torch.load(weights or _download(url), map_location="cpu"))
        model.to(self.device).eval()
        if fp16:
            model.convert_to_fp16()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.ddim = DDIMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule="linear",
                                  clip_sample=False, set_alpha_to_one=False, prediction_type="epsilon")

    # ------------------------------------------------------------- core
    def eps(self, z: torch.Tensor, t) -> torch.Tensor:
        tt = torch.as_tensor(t, device=self.device).reshape(-1).expand(z.shape[0]).long()
        return self.model(z.float(), tt)[:, :3].float()        # fp32 in/out; body runs fp16

    def _x0_eps(self, z, t, a_t):
        e = self.eps(z, t)
        sa, sb = a_t.sqrt(), (1 - a_t).sqrt()
        return (z - sb * e) / sa, e

    @staticmethod
    def _img(x0: torch.Tensor) -> torch.Tensor:                # model space [-1,1] -> [0,1]
        return (x0 / 2 + 0.5).clamp(0, 1)

    def _objective_value(self, x0, objective, cfg: GuidanceConfig, gen) -> torch.Tensor:
        img = self._img(x0)
        if cfg.smooth_k <= 1:
            return objective.score(img)
        B = img.shape[0]
        noisy = img[None].expand(cfg.smooth_k, -1, -1, -1, -1).reshape(cfg.smooth_k * B, *img.shape[1:])
        noisy = (noisy + cfg.smooth_sigma * torch.randn(noisy.shape, generator=gen, device=img.device)).clamp(0, 1)
        return objective.score(noisy).view(cfg.smooth_k, B).mean(0)

    def _sample(self, z: torch.Tensor, ts, objective, cfg: GuidanceConfig, gen,
                on_step: Optional[Callable[[int, int, torch.Tensor], None]] = None) -> Dict[str, object]:
        acp = self.ddim.alphas_cumprod.to(self.device)
        T = float(self.ddim.config.num_train_timesteps)
        n = z.shape[0]
        trace: List[List[float]] = [[] for _ in range(n)]
        for i, t in enumerate(ts):
            a_t = acp[t]
            a_prev = acp[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=self.device)
            frac = float(t) / T
            guide = objective is not None and cfg.strength > 0 and cfg.guide_to <= frac <= cfg.guide_from
            if guide:
                zg = z.detach()
                for _ in range(max(1, cfg.repeats)):
                    zg = zg.requires_grad_(True)
                    x0, _ = self._x0_eps(zg, t, a_t)
                    s = self._objective_value(x0, objective, cfg, gen)
                    (-s.sum()).backward()
                    g = zg.grad
                    with torch.no_grad():
                        step = cfg.strength * (zg.flatten(1).norm(dim=1) / g.flatten(1).norm(dim=1).clamp_min(1e-8))
                        zg = (zg - step.view(-1, 1, 1, 1) * g).detach()
                for k in range(n):
                    trace[k].append(float(s[k].detach()))
                z = zg
            with torch.no_grad():
                x0, e = self._x0_eps(z, t, a_t)
                if not guide and objective is not None:
                    s = objective.score(self._img(x0))
                    for k in range(n):
                        trace[k].append(float(s[k]))
                z = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * e
            if on_step is not None:
                on_step(i, len(ts), x0)
        with torch.no_grad():
            img = self._img(x0)
            scores = [float(v) for v in objective.score(img)] if objective is not None else [0.0] * n
        return {"images": to_pil(img), "tensor": img, "scores": scores, "trace": trace}

    # ------------------------------------------------------------- public
    def guided_sample(self, objective, cfg: GuidanceConfig = GuidanceConfig(), on_step=None) -> Dict[str, object]:
        """Sample ``cfg.n_images`` images from pure noise under detector guidance."""
        gen = torch.Generator(self.device).manual_seed(cfg.seed) if cfg.seed is not None else None
        self.ddim.set_timesteps(cfg.steps, device=self.device)
        ts = self.ddim.timesteps
        z = torch.randn((cfg.n_images, 3, self.image_size, self.image_size), generator=gen, device=self.device)
        return self._sample(z, ts, objective, cfg, gen, on_step)

    def guided_refine(self, seeds: Sequence[Image.Image], objective, cfg: GuidanceConfig = GuidanceConfig(),
                      on_step=None) -> Dict[str, object]:
        """SDEdit: re-noise real ``seeds`` to ``cfg.refine_noise`` of the schedule and denoise under guidance."""
        gen = torch.Generator(self.device).manual_seed(cfg.seed) if cfg.seed is not None else None
        self.ddim.set_timesteps(cfg.steps, device=self.device)
        ts_all = self.ddim.timesteps
        start = int(len(ts_all) * (1 - cfg.refine_noise))
        ts = ts_all[start:]
        acp = self.ddim.alphas_cumprod.to(self.device)
        a_t = acp[ts[0]]
        x = to_tensor01(seeds, self.image_size, self.device) * 2 - 1
        noise = torch.randn(x.shape, generator=gen, device=self.device)
        z = a_t.sqrt() * x + (1 - a_t).sqrt() * noise
        return self._sample(z, ts, objective, cfg, gen, on_step)
