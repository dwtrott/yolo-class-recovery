"""Diffusion prior: plain generation for prompt search, and detector-guided sampling.

Guided sampling follows the "manifold-preserving" family of plug-and-play
guidance (Universal Guidance / FreeDoM / MPGD): at every denoising step we form
the model's estimate of the clean image x̂0, decode it with a small VAE,
push it through the detector, and move x̂0 a little in the direction that
raises the target class score before re-noising.  The gradient never goes
through the UNet, so it is cheap enough for a Colab T4 at 512px.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch
from PIL import Image

from .detector import ClassObjective


def _pil(x: torch.Tensor) -> List[Image.Image]:
    x = (x.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in x]


@dataclass
class GuidanceConfig:
    steps: int = 8                 # denoising steps (sd-turbo: 4-8; SD1.5/2.1: 25-50)
    cfg: float = 0.0               # classifier-free guidance scale (0 for *-turbo models)
    strength: float = 0.08         # guidance step as a fraction of ||x̂0|| per denoising step
    guide_from: float = 1.0        # guide only while t/T is within [guide_to, guide_from]
    guide_to: float = 0.2          #   (early = coarse structure, late = fine texture)
    repeats: int = 2               # gradient steps per denoising step
    height: int = 512
    width: int = 512
    n_images: int = 4
    seed: Optional[int] = None


class DiffusionPrior:
    """Thin wrapper around a Stable-Diffusion-family pipeline.

    Parameters
    ----------
    model_id : any SD 1.x / 2.x / *-turbo checkpoint on the Hub or on disk.
        ``stabilityai/sd-turbo`` (default) is fast enough for large prompt
        sweeps; ``stable-diffusion-v1-5/stable-diffusion-v1-5`` or
        ``stabilityai/stable-diffusion-2-1-base`` give more denoising steps to
        guide through.
    tiny_vae : use TAESD (``madebyollin/taesd``) as the differentiable decoder
        during guidance; the full VAE is still used for the final image.
    """

    def __init__(self, model_id: str = "stabilityai/sd-turbo", device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None, tiny_vae: bool = True, tiny_vae_id: str = "madebyollin/taesd"):
        from diffusers import AutoencoderTiny, DDIMScheduler, StableDiffusionPipeline

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.model_id = model_id
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=self.dtype,
                                                            safety_checker=None, requires_safety_checker=False)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(self.device)
        self.unet, self.vae = self.pipe.unet, self.pipe.vae
        self.tokenizer, self.text_encoder = self.pipe.tokenizer, self.pipe.text_encoder
        # a DDIM copy of the scheduler for the manual guided sampler
        self.ddim = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pred_type = getattr(self.ddim.config, "prediction_type", "epsilon")
        self.guide_vae = None
        if tiny_vae:
            try:
                self.guide_vae = AutoencoderTiny.from_pretrained(tiny_vae_id, torch_dtype=torch.float32).to(self.device)
            except Exception as e:  # pragma: no cover - network / compat
                print(f"[classrecovery] TAESD unavailable ({e}); falling back to the full VAE for guidance")
        self.is_turbo = "turbo" in model_id.lower() or "lcm" in model_id.lower()

    # ------------------------------------------------------------ helpers
    @torch.no_grad()
    def encode_prompt(self, prompts: Sequence[str], negative: Optional[str] = None):
        tok = self.tokenizer(list(prompts), padding="max_length", truncation=True,
                             max_length=self.tokenizer.model_max_length, return_tensors="pt")
        cond = self.text_encoder(tok.input_ids.to(self.device))[0]
        uncond = None
        if negative is not None:
            tok_u = self.tokenizer([negative] * len(prompts), padding="max_length", truncation=True,
                                   max_length=self.tokenizer.model_max_length, return_tensors="pt")
            uncond = self.text_encoder(tok_u.input_ids.to(self.device))[0]
        return cond, uncond

    def decode(self, latents: torch.Tensor, tiny: bool = False) -> torch.Tensor:
        """latents (UNet space) -> image in [0,1].  Differentiable."""
        vae = self.guide_vae if (tiny and self.guide_vae is not None) else self.vae
        z = latents / vae.config.scaling_factor
        img = vae.decode(z.to(vae.dtype)).sample
        return (img.float() / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------- plain sampling
    @torch.no_grad()
    def generate(self, prompts: Sequence[str], steps: Optional[int] = None, cfg: Optional[float] = None,
                 seed: Optional[int] = None, height: int = 512, width: int = 512,
                 negative: Optional[str] = None) -> List[Image.Image]:
        """Ordinary text-to-image through the diffusers pipeline (batched)."""
        steps = steps or (2 if self.is_turbo else 25)
        cfg = (0.0 if self.is_turbo else 7.0) if cfg is None else cfg
        gen = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        out = self.pipe(list(prompts), num_inference_steps=steps, guidance_scale=cfg, generator=gen,
                        height=height, width=width, negative_prompt=[negative] * len(prompts) if negative else None)
        return out.images

    # ------------------------------------------------------ guided sampling
    def guided_sample(self, prompt: str, objective: ClassObjective, cfg: GuidanceConfig = GuidanceConfig(),
                      negative: Optional[str] = None, on_step: Optional[Callable[[int, int, torch.Tensor], None]] = None
                      ) -> Dict[str, object]:
        """Sample ``cfg.n_images`` images from ``prompt`` while steering towards ``objective``.

        Returns ``{"images": [PIL], "scores": [float], "trace": [[score per step] ...]}``.
        """
        n = cfg.n_images
        gen = torch.Generator(self.device).manual_seed(cfg.seed) if cfg.seed is not None else None
        use_cfg = cfg.cfg > 1.0 and not self.is_turbo
        cond, uncond = self.encode_prompt([prompt] * n, negative if use_cfg else None)
        if use_cfg and uncond is None:
            uncond, _ = self.encode_prompt([""] * n)

        self.ddim.set_timesteps(cfg.steps, device=self.device)
        ts = self.ddim.timesteps
        acp = self.ddim.alphas_cumprod.to(self.device)
        T = float(self.ddim.config.num_train_timesteps)
        ch = self.unet.config.in_channels
        z = torch.randn((n, ch, cfg.height // 8, cfg.width // 8), generator=gen, device=self.device, dtype=self.dtype)
        z = z * self.ddim.init_noise_sigma
        trace: List[List[float]] = [[] for _ in range(n)]

        for i, t in enumerate(ts):
            a_t = acp[t]
            a_prev = acp[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=self.device)
            with torch.no_grad():
                zin = torch.cat([z, z]) if use_cfg else z
                cin = torch.cat([uncond, cond]) if use_cfg else cond
                pred = self.unet(zin, t, encoder_hidden_states=cin).sample
                if use_cfg:
                    pu, pc = pred.chunk(2)
                    pred = pu + cfg.cfg * (pc - pu)
                pred = pred.float()
                zf = z.float()
                sa, sb = a_t.sqrt(), (1 - a_t).sqrt()
                if self.pred_type == "v_prediction":
                    x0 = sa * zf - sb * pred
                    eps = sa * pred + sb * zf
                else:
                    eps = pred
                    x0 = (zf - sb * eps) / sa

            frac = float(t) / T
            guide = cfg.strength > 0 and cfg.guide_to <= frac <= cfg.guide_from
            if guide:
                x0 = self._guide_x0(x0, objective, cfg, trace)
                eps = (zf - sa * x0) / sb          # keep (x0, eps) consistent with z_t
            else:
                with torch.no_grad():
                    s = objective.score(self.decode(x0, tiny=True))
                for k in range(n):
                    trace[k].append(float(s[k]))

            with torch.no_grad():
                z = (a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps).to(self.dtype)
            if on_step is not None:
                on_step(i, len(ts), x0.detach())

        with torch.no_grad():
            imgs = self.decode(z.float() if a_prev.item() == 1.0 else x0)
            final_scores = [float(v) for v in objective.score(imgs)]
        return {"images": _pil(imgs), "scores": final_scores, "trace": trace}

    def _guide_x0(self, x0: torch.Tensor, objective: ClassObjective, cfg: GuidanceConfig,
                  trace: List[List[float]]) -> torch.Tensor:
        x0 = x0.detach()
        for _ in range(max(1, cfg.repeats)):
            x0 = x0.requires_grad_(True)
            img = self.decode(x0, tiny=True)
            s = objective.score(img)
            (-s.sum()).backward()
            g = x0.grad
            with torch.no_grad():
                gn = g.flatten(1).norm(dim=1).clamp_min(1e-8)
                xn = x0.flatten(1).norm(dim=1)
                step = cfg.strength * (xn / gn).view(-1, 1, 1, 1)
                x0 = (x0 - step * g).detach()
        for k in range(x0.shape[0]):
            trace[k].append(float(s[k].detach()))
        return x0
