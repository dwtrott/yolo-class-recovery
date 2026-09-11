"""Checkpoint in, clear high-scoring images per class out.

    from classrecovery import represent
    results = represent("mystery.pt", pool="pools/coco-val2017")

For every class index (read from the head architecture — nothing else is
assumed about the model):

1. **seed**    scan an unlabeled image pool with the detector; keep the crops the
               class fires on and that pass the robust checks (real photos, so
               they are clear by construction)
2. **refine**  SDEdit: re-noise each seed to ~50 % and denoise under detector
               guidance (smoothed gradient, through the diffusion network) so the
               prior sharpens the object into what the class most wants to see
3. **noise**   if the pool gave nothing (a class the pool does not cover), sample
               from pure noise under the same guidance
4. **rank**    score everything with the robust composite (augmentation
               consistency, box stability, localization, cutout survival); the
               contact sheet shows the best first, labelled by where it came from

No class names, captions, vocabularies, or labels are used anywhere.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence

from PIL import Image

from .detector import ClassObjective, Detector
from .prior import GuidanceConfig, UncondPrior
from .robust import BNGuidedObjective, RobustObjective, agreement, degradation_scores, discover_prototype
from .seeds import list_images, mine_seeds
from .viz import ClassResult, class_sheet


def represent(weights: str, pool: Optional[str] = None, classes: Optional[Sequence[int]] = None,
              prior: Optional[UncondPrior] = None, n_seeds: int = 6, n_noise: int = 4, pool_limit: Optional[int] = None,
              steps: int = 50, strength: float = 0.03, refine_noise: float = 0.5, smooth_k: int = 8,
              repeats: int = 1, guide_frac: float = 0.5, seed: int = 0, use_prototype: bool = False,
              min_seed_score: float = 0.15, det_imgsz: int = 320, bn_weight: float = 0.0,
              twin: Optional[object] = None, twin_steps: int = 0, realness_max: Optional[float] = None,
              refine: bool = True,
              out_dir: Optional[str] = None, show: bool = True,
              progress: Optional[Callable[[str], None]] = print) -> Dict[int, ClassResult]:
    """See module docstring.  Returns ``{class_idx: ClassResult}``.

    pool            directory of unlabeled images (see ``seeds.download_pool``); None = noise only
    n_seeds         seeds to mine and refine per class
    n_noise         from-noise samples per class (always generated when the pool yields < 2 seeds,
                    otherwise only if ``n_noise`` > 0 and you want the comparison)
    use_prototype   also cluster the candidates' internal activations and re-guide toward the
                    dominant mode (slower; helps when the candidates are mixed)
    bn_weight       BatchNorm-statistics term (DeepInversion).  OFF by default: at 0.3-0.6 it pulls
                    images toward flat posterised colour and collapses from-noise samples; if you
                    experiment, start at 0.02.
    twin            a noise-robust twin Detector (see ``twin.train_twin``) used for the GUIDANCE
                    gradient only; the original detector still does all scoring.  ``twin_steps>0``
                    trains one on the pool first (no labels; ~10 min on an A100 for 1500 steps).
    realness_max    absolute cap on the prior's reconstruction error; above it a candidate is
                    flagged off-manifold.  Default: 3x the median error of real pool images
                    (or 0.03 without a pool).
    """
    det = Detector(weights, imgsz=det_imgsz)      # 320 suits the 256px prior; raise for a high-res detector
    if progress:
        progress(f"loaded {weights}: {det.nc} classes read from the head architecture")
    prior = prior or UncondPrior()
    pool_images = list_images(pool, pool_limit) if pool else []
    if progress and pool:
        progress(f"pool: {len(pool_images)} images")
    classes = list(classes) if classes is not None else list(range(det.nc))
    results: Dict[int, ClassResult] = {}

    if twin is None and twin_steps > 0:
        if not pool_images:
            raise ValueError("twin_steps>0 needs a pool of unlabeled images to distil on")
        from .twin import train_twin
        if progress:
            progress(f"training a noise-robust twin of the detector on the pool ({twin_steps} steps, no labels)")
        twin = train_twin(det, pool_images, steps=twin_steps, progress=progress,
                          save_path=os.path.join(out_dir, "twin.pt") if out_dir else None)
    guide_det = twin if twin is not None else det

    # realness calibration: what does the prior's reconstruction error look like on real photos?
    if realness_max is None:
        if pool_images:
            ref_ims = [Image.open(p_).convert("RGB") for p_ in pool_images[:24]]
            realness_max = 3.0 * float(prior.realness(ref_ims, seed=seed).median())
        else:
            realness_max = 0.03
        if progress:
            progress(f"realness cap: {realness_max:.3f} (reconstruction error above this = off-manifold)")

    for c in classes:
        rec = ClassResult(c)
        guide = ClassObjective(guide_det, c, n_aug=2, seed=seed)
        if bn_weight > 0:
            guide = BNGuidedObjective(guide, w_bn=bn_weight)
        cfg = GuidanceConfig(steps=steps, strength=strength, repeats=repeats, guide_frac=guide_frac,
                             smooth_k=smooth_k, seed=seed, refine_noise=refine_noise)
        cand: List[Image.Image] = []
        src: List[str] = []

        # 1. seeds
        seeds = []
        if pool_images:
            if progress:
                progress(f"class {c}: mining seeds from the pool")
            seeds = [s for s in mine_seeds(det, pool_images, c, top_k=n_seeds, progress=progress)
                     if s["score"] >= min_seed_score]
            if progress:
                progress(f"   {len(seeds)} seeds above {min_seed_score:.2f}" +
                         (f"; best det={seeds[0]['score']:.2f}" if seeds else ""))
            cand += [s["image"] for s in seeds]
            src += ["seed"] * len(seeds)

        # 2. refine seeds
        if seeds and refine:
            if progress:
                progress(f"class {c}: refining {len(seeds)} seeds under guidance")
            rcfg = GuidanceConfig(**{**cfg.__dict__, "n_images": len(seeds)})
            out = prior.guided_refine([s["image"] for s in seeds], guide, rcfg)
            cand += out["images"]
            src += ["refined"] * len(out["images"])
            rec.trace = out["trace"]

        # 3. from noise
        if len(seeds) < 2 or n_noise > 0:
            k = max(n_noise, 4) if len(seeds) < 2 else n_noise
            if progress:
                progress(f"class {c}: sampling {k} images from noise under guidance")
            ncfg = GuidanceConfig(**{**cfg.__dict__, "n_images": k})
            out = prior.guided_sample(guide, ncfg)
            cand += out["images"]
            src += ["noise"] * len(out["images"])
            rec.trace = rec.trace or out["trace"]

        # 3b. optional prototype pass
        if use_prototype and len(cand) >= 4:
            if progress:
                progress(f"class {c}: clustering candidate activations, re-guiding toward the dominant mode")
            disc = discover_prototype(det, prior, c, n_candidates=len(cand), candidates=cand, seed=seed)
            rec.modes = [(len(m.members), m.robust) for m in disc["modes"]]
            pobj = RobustObjective(det, c, w_proto=1.0, prototype=disc["dominant"].prototype, seed=seed)
            base = [cand[i] for i in disc["dominant"].members[:max(2, n_seeds // 2)]]
            pcfg = GuidanceConfig(**{**cfg.__dict__, "n_images": len(base)})
            out = prior.guided_refine(base, pobj, pcfg)
            pobj.close()
            cand += out["images"]
            src += ["prototype"] * len(out["images"])

        # 4. rank: adversarial patterns die under mild degradation and sit off the prior's manifold;
        #    real objects survive both.  Rank by the mean score under degradation, drop candidates
        #    the prior itself cannot reconstruct, and use the raw score only as a tie-breaker.
        D = degradation_scores(det, c, cand, seed=seed)
        real = prior.realness(cand, seed=seed)
        on_manifold = [float(real[i]) <= realness_max for i in range(len(cand))]
        det_scores = D["clean"].tolist()
        key = [float(D["mean"][i]) + 0.05 * det_scores[i] - (0.0 if on_manifold[i] else 10.0) for i in range(len(cand))]
        order = sorted(range(len(cand)), key=lambda i: -key[i])
        rec.images = [cand[i] for i in order]
        rec.scores = [det_scores[i] for i in order]
        rec.robust = [float(D["mean"][i]) for i in order]             # mean score under noise/blur/jpeg/half-res
        rec.realness = [float(real[i]) for i in order]
        rec.source = [src[i] + ("" if on_manifold[i] else "*off-manifold") for i in order]
        if hasattr(guide, "close"):
            guide.close()
        ref = [Image.open(p_).convert("RGB") for p_ in pool_images[:48]] if pool_images else None
        agree = agreement(det, rec.images, c, reference=ref)
        conf = "confident" if agree >= 0.6 else ("mixed" if agree >= 0.4 else "prior-limited")
        rec.agreement = agree
        rec.note = (f"{len(seeds)} seeds, {src.count('refined')} refined, {src.count('noise')} from noise   |   "
                    f"agreement of top images {agree:.2f} -> {conf}")
        results[c] = rec
        if progress:
            progress(f"   class {c}: best={rec.source[0]} det={rec.scores[0]:.2f} degraded={rec.robust[0]:.2f}  "
                     f"agreement={agree:.2f} ({conf})")

        sheet = class_sheet(rec, title=f"class {c}   {conf.upper()}   best: {rec.source[0]}  det={rec.scores[0]:.2f}  "
                                       f"degraded={rec.robust[0]:.2f}  err={rec.realness[0]:.3f}")
        if out_dir:
            d = os.path.join(out_dir, f"class_{c:03d}")
            os.makedirs(d, exist_ok=True)
            for i, (im, s_) in enumerate(zip(rec.images, rec.source)):
                im.save(os.path.join(d, f"{i:02d}_{s_}_det{rec.scores[i]:.2f}.png"))
            sheet.save(os.path.join(out_dir, f"class_{c:03d}.png"))
        if show:
            try:
                from IPython.display import display
                display(sheet)
            except Exception:
                pass
    return results
