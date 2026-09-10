"""classrecovery — recover the unknown classes of an object detector.

Given a YOLO (Ultralytics) checkpoint whose class names are missing or
untrustworthy, this package tries to work out what each class index responds
to, using a pretrained text-to-image diffusion model as a natural-image prior
instead of raw pixel-space activation maximisation.

Main pieces
-----------
- ``Detector``            differentiable wrapper around a YOLO checkpoint
- ``weight_diff``         compare a fine-tuned checkpoint against its base
- ``prompt_search``       gradient-free: generate images for a vocabulary of
                          nouns, see which class fires
- ``guidance``            gradient-based: steer the diffusion sampler with the
                          detector's class score (Universal-Guidance style)
- ``naming``              turn recovered images into ranked candidate names
                          with CLIP zero-shot
- ``testbed``             build an "undocumented fine-tune" to evaluate on
- ``app``                 Gradio GUI (works in Colab with ``share=True``)
"""

from .detector import Detector, ClassObjective

__all__ = ["Detector", "ClassObjective"]
__version__ = "0.1.0"
