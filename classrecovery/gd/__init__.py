"""Vendored from https://github.com/openai/guided-diffusion (MIT License, see LICENSE).

Only the UNet definition and its helpers are included, so the unconditional
ImageNet diffusion models can be loaded without installing the upstream
package (whose setup.py does not install an importable package).
"""
from .unet import UNetModel  # noqa: F401
