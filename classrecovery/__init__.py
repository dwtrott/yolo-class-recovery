"""classrecovery — checkpoint in, clear high-scoring images per class out.

Given an object-detection checkpoint whose classes are unknown, produce clear,
representative images that each class index fires on, using nothing but the
detector's own activations and an unconditional diffusion prior.

    from classrecovery import represent
    results = represent("mystery.pt", pool="pools/coco-val2017")
"""

from .detector import ClassObjective, Detector
from .prior import GuidanceConfig, UncondPrior
from .represent import represent
from .robust import RobustObjective
from .viz import ClassResult, class_sheet, contact_sheets, stack

__all__ = ["represent", "Detector", "ClassObjective", "UncondPrior", "GuidanceConfig", "RobustObjective",
           "ClassResult", "class_sheet", "contact_sheets", "stack"]
__version__ = "1.0.0"
