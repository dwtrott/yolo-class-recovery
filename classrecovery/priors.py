"""Text-free image priors for classifier guidance.

``OpenAIUncondPrior`` wraps the unconditional ImageNet diffusion models from
Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis* (2021) —
the models classifier guidance was designed and demonstrated on.  They have
never seen a caption; the only steering signal they accept is a gradient.

Usage::

    from classrecovery.priors import OpenAIUncondPrior
    prior = OpenAIUncondPrior()                   # 256x256 unconditional ImageNet
    represent("mystery.pt", prior=prior, res=256)

or simply ``represent("mystery.pt", prior_id="openai/imagenet-256-uncond")``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

import torch

from .diffusion import DiffusionPrior

_MODELS = {
    # name -> (weights url, image_size, model kwargs)
    "openai/imagenet-256-uncond": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt", 256,
        dict(num_channels=256, num_res_blocks=2, channel_mult="", attention_resolutions="32,16,8",
             num_heads=4, num_head_channels=64, num_heads_upsample=-1, use_scale_shift_norm=True,
             resblock_updown=True, dropout=0.0, learn_sigma=True, class_cond=False, use_new_attention_order=False)),
    "openai/lsun-bedroom-256": (
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt", 256,
        dict(num_channels=256, num_res_blocks=2, channel_mult="", attention_resolutions="32,16,8",
             num_heads=4, num_head_channels=64, num_heads_upsample=-1, use_scale_shift_norm=True,
             resblock_updown=True, dropout=0.1, learn_sigma=True, class_cond=False, use_new_attention_order=False)),
}


def _ensure_guided_diffusion():
    try:
        import guided_diffusion  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/openai/guided-diffusion"])
    # guided-diffusion's own checkpoint() differentiates w.r.t. the *parameters* too, which fails when
    # they are frozen; swap in torch's checkpoint so only the input gradient is computed.
    import guided_diffusion.nn as gdnn
    import guided_diffusion.unet as gdunet
    from torch.utils.checkpoint import checkpoint as _torch_ckpt

    def _ckpt(func, inputs, params, flag):
        if flag:
            return _torch_ckpt(func, *inputs, use_reentrant=False)
        return func(*inputs)
    gdnn.checkpoint = _ckpt
    gdunet.checkpoint = _ckpt


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


class OpenAIUncondPrior(DiffusionPrior):
    """Unconditional pixel-space diffusion prior (guided-diffusion UNet) usable by ``guided_sample``."""

    def __init__(self, name: str = "openai/imagenet-256-uncond", device: Optional[str] = None,
                 fp16: Optional[bool] = None, use_checkpoint: bool = True, weights: Optional[str] = None):
        from diffusers import DDIMScheduler
        _ensure_guided_diffusion()
        from guided_diffusion.script_util import create_model

        url, size, kw = _MODELS[name]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        fp16 = (self.device.type == "cuda") if fp16 is None else fp16
        self.dtype = torch.float16 if fp16 else torch.float32
        self.model_id = name
        self.image_size = size
        model = create_model(image_size=size, use_checkpoint=use_checkpoint, use_fp16=fp16, **kw)
        path = weights or _download(url)
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd)
        model.to(self.device).eval()
        if fp16:
            model.convert_to_fp16()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        # linear beta schedule 1e-4..0.02 over 1000 steps, epsilon prediction — as in the original code
        self.ddim = DDIMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule="linear",
                                  clip_sample=False, set_alpha_to_one=False, prediction_type="epsilon")
        self.pred_type = "epsilon"
        self.has_text, self.pixel_space, self.latent_scale = False, True, 1
        self.is_turbo, self.guide_vae, self.vae = False, None, None
        self.unet = _Shim(self._model)          # gives .config.in_channels / .enable_gradient_checkpointing()

    def _unet(self, z: torch.Tensor, t, emb: Optional[torch.Tensor]) -> torch.Tensor:
        tt = torch.as_tensor(t, device=self.device).reshape(-1).expand(z.shape[0])
        out = self._model(z.to(self.dtype), tt.long())
        return out[:, :3].float()               # first 3 channels = eps; the rest is the learned variance


class _Shim:
    """Minimal duck-typing of a diffusers UNet for the bits ``guided_sample`` touches."""

    class _Cfg:
        in_channels = 3

    def __init__(self, model):
        self.model = model
        self.config = self._Cfg()

    def enable_gradient_checkpointing(self):
        pass                                     # handled by ``use_checkpoint`` at construction

    def requires_grad_(self, flag: bool):
        for p in self.model.parameters():
            p.requires_grad_(flag)
        return self
