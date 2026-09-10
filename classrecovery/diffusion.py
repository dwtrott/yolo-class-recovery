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
    through_unet: bool = True      # true classifier guidance: gradient flows back through the diffusion network
                                   # into the noisy state z_t (changes content); False = cheap x̂0-only nudge (texture)


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
                 dtype: Optional[torch.dtype] = None, tiny_vae: bool = True, tiny_vae_id: str = "madebyollin/taesd",
                 unconditional: bool = False):
        from diffusers import AutoencoderTiny, DDIMScheduler, StableDiffusionPipeline

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.model_id = model_id
        self.has_text = not unconditional
        if unconditional:
            # a pixel-space unconditional model (diffusers UNet2DModel, e.g. google/ddpm-ema-*): no text, no VAE
            from diffusers import DDPMPipeline
            self.pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
            self.unet, self.vae, self.tokenizer, self.text_encoder = self.pipe.unet, None, None, None
            self.ddim = DDIMScheduler.from_config(self.pipe.scheduler.config)
            self.pred_type = getattr(self.ddim.config, "prediction_type", "epsilon")
            self.guide_vae, self.is_turbo, self.pixel_space = None, False, True
            self.latent_scale = 1
            return
        self.pixel_space, self.latent_scale = False, 8
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

    def _unet(self, z: torch.Tensor, t, emb: Optional[torch.Tensor]) -> torch.Tensor:
        if self.has_text:
            return self.unet(z.to(self.dtype), t, encoder_hidden_states=emb.to(self.dtype)).sample.float()
        return self.unet(z.to(self.dtype), t).sample.float()

    def decode(self, latents: torch.Tensor, tiny: bool = False) -> torch.Tensor:
        """latents (UNet space) -> image in [0,1].  Differentiable."""
        if getattr(self, "pixel_space", False):
            return (latents.float() / 2 + 0.5).clamp(0, 1)
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
        use_cfg = self.has_text and cfg.cfg > 1.0 and not self.is_turbo and prompt != ""
        cond = uncond = None
        if self.has_text:
            # prompt "" = the model's unconditional branch: pure image prior, no text steering at all
            cond, uncond = self.encode_prompt([prompt] * n, negative if use_cfg else None)
            if use_cfg and uncond is None:
                uncond, _ = self.encode_prompt([""] * n)

        self.ddim.set_timesteps(cfg.steps, device=self.device)
        ts = self.ddim.timesteps
        acp = self.ddim.alphas_cumprod.to(self.device)
        T = float(self.ddim.config.num_train_timesteps)
        ch = self.unet.config.in_channels
        f = self.latent_scale
        z = torch.randn((n, ch, cfg.height // f, cfg.width // f), generator=gen, device=self.device, dtype=self.dtype)
        z = z * self.ddim.init_noise_sigma
        trace: List[List[float]] = [[] for _ in range(n)]
        if cfg.through_unet and hasattr(self.unet, "enable_gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()

        def predict(zz):
            zin = torch.cat([zz, zz]) if use_cfg else zz
            cin = torch.cat([uncond, cond]) if use_cfg else cond
            pred = self._unet(zin, t, cin)
            if use_cfg:
                pu, pc = pred.chunk(2)
                pred = pu + cfg.cfg * (pc - pu)
            zf = zz.float()
            sa, sb = a_t.sqrt(), (1 - a_t).sqrt()
            if self.pred_type == "v_prediction":
                return sa * zf - sb * pred, sa * pred + sb * zf
            return (zf - sb * pred) / sa, pred

        for i, t in enumerate(ts):
            a_t = acp[t]
            a_prev = acp[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=self.device)
            sa, sb = a_t.sqrt(), (1 - a_t).sqrt()
            frac = float(t) / T
            guide = cfg.strength > 0 and cfg.guide_to <= frac <= cfg.guide_from

            if guide and cfg.through_unet:
                # classifier guidance proper: d score / d z_t through the diffusion network
                zg = z.detach().float()
                for _ in range(max(1, cfg.repeats)):
                    zg = zg.requires_grad_(True)
                    x0, _ = predict(zg)
                    s = objective.score(self.decode(x0, tiny=True))
                    (-s.sum()).backward()
                    g = zg.grad
                    with torch.no_grad():
                        gn = g.flatten(1).norm(dim=1).clamp_min(1e-8)
                        zn = zg.flatten(1).norm(dim=1)
                        zg = (zg - cfg.strength * (zn / gn).view(-1, 1, 1, 1) * g).detach()
                for k in range(n):
                    trace[k].append(float(s[k].detach()))
                z = zg.to(self.dtype)
                with torch.no_grad():
                    x0, eps = predict(z)
                zf = z.float()
            elif guide:
                with torch.no_grad():
                    x0, eps = predict(z)
                zf = z.float()
                x0 = self._guide_x0(x0, objective, cfg, trace)
                eps = (zf - sa * x0) / sb          # keep (x0, eps) consistent with z_t
            else:
                with torch.no_grad():
                    x0, eps = predict(z)
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

    # ------------------------------------------------ embedding-space inversion
    def generate_from_embedding(self, emb: torch.Tensor, noise: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Differentiable few-step DDIM generation conditioned on a text embedding.

        emb   : (B,77,D) text-encoder output (may require grad)
        noise : (B,C,h,w) starting Gaussian noise
        Returns the *guide-VAE decoded* image in [0,1], (B,3,H,W), with grad.
        Intended for distilled 1–4 step models (sd-turbo / LCM).
        """
        self.ddim.set_timesteps(steps, device=self.device)
        ts = self.ddim.timesteps
        acp = self.ddim.alphas_cumprod.to(self.device)
        z = noise * self.ddim.init_noise_sigma
        for i, t in enumerate(ts):
            a_t = acp[t]
            a_prev = acp[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=self.device)
            pred = self.unet(z.to(self.dtype), t, encoder_hidden_states=emb.to(self.dtype)).sample.float()
            zf = z.float()
            sa, sb = a_t.sqrt(), (1 - a_t).sqrt()
            if self.pred_type == "v_prediction":
                x0, eps = sa * zf - sb * pred, sa * pred + sb * zf
            else:
                eps, x0 = pred, (zf - sb * pred) / sa
            z = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return self.decode(x0, tiny=True)

    def invert_embedding(self, objective: ClassObjective, init_prompt: str = "a photo", iters: int = 150,
                         lr: float = 0.02, batch: int = 4, steps: int = 1, height: int = 512, width: int = 512,
                         reg: float = 0.05, seed: int = 0, grad_ckpt: bool = True,
                         on_iter: Optional[Callable[[int, float, torch.Tensor], None]] = None) -> Dict[str, object]:
        """Textual inversion against the detector: find a text embedding whose images make class k fire.

        The generator (UNet, ``steps`` denoising steps) is differentiated end to
        end, so the gradient can change *what* is painted, not just its texture.
        Fresh noise every iteration keeps the embedding from memorising one
        layout; ``reg`` keeps it near the neutral ``init_prompt`` embedding so
        it stays on the text manifold.
        """
        self.unet.requires_grad_(False)
        if grad_ckpt and hasattr(self.unet, "enable_gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()
        e0, _ = self.encode_prompt([init_prompt])
        e0 = e0.float().detach()
        emb = e0.clone().requires_grad_(True)
        opt = torch.optim.Adam([emb], lr=lr)
        gen = torch.Generator(self.device).manual_seed(seed)
        ch = self.unet.config.in_channels
        trace: List[float] = []
        best = (-1e9, None)
        for it in range(iters):
            noise = torch.randn((batch, ch, height // 8, width // 8), generator=gen, device=self.device)
            img = self.generate_from_embedding(emb.expand(batch, -1, -1), noise, steps)
            s = objective.score(img)
            loss = -s.mean() + reg * ((emb - e0) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sm = float(s.detach().mean())
            trace.append(sm)
            if sm > best[0]:
                best = (sm, emb.detach().clone())
            if on_iter is not None:
                on_iter(it, sm, img.detach())
        emb_best = best[1] if best[1] is not None else emb.detach()
        return {"embedding": emb_best, "init_embedding": e0, "trace": trace, "best_score": best[0]}

    @torch.no_grad()
    def sample_from_embedding(self, emb: torch.Tensor, n: int = 6, steps: int = 2, height: int = 512,
                              width: int = 512, seed: int = 1) -> List[Image.Image]:
        """Render fresh images from an (optimised) embedding with the full VAE."""
        gen = torch.Generator(self.device).manual_seed(seed)
        ch = self.unet.config.in_channels
        noise = torch.randn((n, ch, height // 8, width // 8), generator=gen, device=self.device)
        self.ddim.set_timesteps(steps, device=self.device)
        ts = self.ddim.timesteps
        acp = self.ddim.alphas_cumprod.to(self.device)
        z = noise * self.ddim.init_noise_sigma
        e = emb.expand(n, -1, -1)
        for i, t in enumerate(ts):
            a_t = acp[t]
            a_prev = acp[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=self.device)
            pred = self.unet(z.to(self.dtype), t, encoder_hidden_states=e.to(self.dtype)).sample.float()
            zf = z.float()
            sa, sb = a_t.sqrt(), (1 - a_t).sqrt()
            if self.pred_type == "v_prediction":
                x0, eps = sa * zf - sb * pred, sa * pred + sb * zf
            else:
                eps, x0 = pred, (zf - sb * pred) / sa
            z = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return _pil(self.decode(x0))

    @torch.no_grad()
    def nearest_words(self, emb: torch.Tensor, words: Sequence[str], template: str = "a photo of a {}",
                      k: int = 10) -> List[tuple]:
        """Which vocabulary prompts' embeddings are closest to an optimised embedding (mean-pooled cosine)."""
        q = emb.float().mean(1)
        q = q / q.norm(dim=-1, keepdim=True)
        out = []
        for i in range(0, len(words), 64):
            ws = words[i:i + 64]
            e, _ = self.encode_prompt([template.format(w) for w in ws])
            e = e.float().mean(1)
            e = e / e.norm(dim=-1, keepdim=True)
            out.extend(zip(ws, (e @ q.T).squeeze(-1).tolist()))
        return sorted(out, key=lambda t: -t[1])[:k]
