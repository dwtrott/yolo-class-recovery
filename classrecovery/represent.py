"""The one-call interface: detector checkpoint in, representative images per class out.

    from classrecovery.represent import represent
    sheets = represent("mystery.pt")            # list of PIL images, one per class

Nothing about the classes is assumed.  The class count is read from the
architecture, and for every class index the diffusion model is steered by the
detector's own gradient from a neutral prompt until the detector fires on what
it paints.  No vocabulary, no CLIP, no ground truth — just weights + prior.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence

from PIL import Image

from .detector import ClassObjective, Detector
from .diffusion import DiffusionPrior, GuidanceConfig
from .pipeline import ClassRecovery
from .viz import class_sheet


def represent(weights: str, classes: Optional[Sequence[int]] = None, prior: Optional[DiffusionPrior] = None,
              prior_id: str = "openai/imagenet-256-uncond", unconditional: bool = False,
              mode: str = "classifier", prompt: str = "",
              n_images: int = 4, iters: int = 150, lr: float = 0.02, batch: int = 4, gen_steps: int = 1,
              reg: float = 0.05, steps: int = 50, cfg: float = 0.0, strength: float = 0.1, repeats: int = 1,
              guide_to: float = 0.15, res: int = 512, seed: int = 0, margin: float = 0.5, n_aug: int = 2,
              hint_words: bool = False, out_dir: Optional[str] = None, show: bool = True,
              progress: Optional[Callable[[str], None]] = print) -> Dict[int, ClassRecovery]:
    """Produce representative images for every class of ``weights``.

    mode="classifier" (default): classifier guidance in the original sense.
        The diffusion model runs *unconditionally* and at every denoising
        step the gradient of the detector's class-k score is pushed back
        through the diffusion network into the noisy image.  Default prior
        is OpenAI's unconditional 256px ImageNet model (``openai/...``, never
        saw a caption); any Stable-Diffusion id uses SD's no-text branch and
        ``unconditional=True`` loads a text-free diffusers checkpoint.
        At every denoising step the gradient of the
        detector's class-k score is pushed back through the diffusion network
        into the noisy state.  The only thing that decides the content is the
        detector's activations.  ~50 steps, gradient through the UNet each
        step: a few minutes per class on an A100.
    mode="embedding": textual inversion against the detector — the
        text embedding fed to the diffusion model is optimised (``iters`` Adam
        steps, ``batch`` fresh noises each) until the detector fires on what it
        paints, then ``n_images`` fresh samples are rendered from it.  This
        changes *what* is painted, which is what you want.  Needs backprop
        through the generator: fine on an A100 at 512px with sd-turbo.
    mode="latent": the cheaper latent-guidance sampler (steers texture/layout
        only; kept for comparison).

    Returns ``{class_idx: ClassRecovery}``; each has ``guided_images`` (PIL),
    ``guided_scores`` (detector score on each) and ``guided_trace``.  With
    ``show=True`` a contact sheet per class is displayed inline (notebooks);
    with ``out_dir`` the images and sheets are also written to disk.
    ``hint_words`` adds, as a label-free hint, which vocabulary prompts the
    optimised embedding ended up closest to.
    """
    det = Detector(weights)
    if progress:
        progress(f"loaded {weights}: {det.nc} classes read from the head architecture")
    if prior is None:
        if prior_id.startswith("openai/"):
            from .priors import OpenAIUncondPrior
            prior = OpenAIUncondPrior(prior_id)
            res = prior.image_size                      # these models are fixed-size
        else:
            prior = DiffusionPrior(prior_id, unconditional=unconditional)
    classes = list(classes) if classes is not None else list(range(det.nc))
    results: Dict[int, ClassRecovery] = {}
    for c in classes:
        if progress:
            progress(f"class {c}: searching for what makes class {c} fire ({mode} mode) ...")
        obj = ClassObjective(det, c, margin=margin, n_aug=n_aug, seed=seed)
        if mode == "embedding":
            inv = prior.invert_embedding(obj, init_prompt=prompt, iters=iters, lr=lr, batch=batch, steps=gen_steps,
                                         height=res, width=res, reg=reg, seed=seed,
                                         on_iter=(lambda it, sc, _: progress(f"   iter {it:>4}  score {sc:.3f}")
                                                  if it % 10 == 0 else None) if progress else None)
            imgs = prior.sample_from_embedding(inv["embedding"], n=n_images, steps=max(2, gen_steps),
                                               height=res, width=res, seed=seed + 1)
            scores = [float(v) for v in det.class_scores(imgs)[:, c]]
            rec = ClassRecovery(c, guided_images=imgs, guided_scores=scores, guided_trace=[inv["trace"]],
                                prompt_used=f"inverted from '{prompt}'")
            if hint_words:
                from .vocab import load_vocab
                rec.clip_names = [(w, float(v)) for w, v in prior.nearest_words(inv["embedding"], load_vocab())]
                rec.combine()
        else:
            gcfg = GuidanceConfig(steps=steps, cfg=cfg, strength=strength, guide_to=guide_to, repeats=repeats,
                                  height=res, width=res, n_images=n_images, seed=seed,
                                  through_unet=(mode == "classifier"))
            out = prior.guided_sample(prompt, obj, gcfg)
            rec = ClassRecovery(c, guided_images=out["images"], guided_scores=out["scores"],
                                guided_trace=out["trace"], prompt_used=prompt or "(unconditional)")
            if progress:
                progress(f"   score along the trajectory: " + " ".join(f"{v:.2f}" for v in out["trace"][0][::max(1, steps // 8)]))
        # order images by how strongly the detector fires on them
        order = sorted(range(len(rec.guided_images)), key=lambda i: -rec.guided_scores[i])
        rec.guided_images = [rec.guided_images[i] for i in order]
        rec.guided_scores = [rec.guided_scores[i] for i in order]
        results[c] = rec
        sheet = class_sheet(rec, title=f"class {c}  (detector fires: {max(rec.guided_scores):.2f})")
        if out_dir:
            d = os.path.join(out_dir, f"class_{c:03d}")
            os.makedirs(d, exist_ok=True)
            for i, im in enumerate(rec.guided_images):
                im.save(os.path.join(d, f"rep_{i}_score{rec.guided_scores[i]:.2f}.png"))
            sheet.save(os.path.join(out_dir, f"class_{c:03d}.png"))
        if show:
            try:
                from IPython.display import display
                display(sheet)
            except Exception:
                pass
    return results


def represent_images(weights: str, **kw) -> Dict[int, List[Image.Image]]:
    """Same as :func:`represent` but returns just ``{class_idx: [images...]}``."""
    return {c: r.guided_images for c, r in represent(weights, show=False, **kw).items()}
