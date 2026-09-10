"""End-to-end recovery of one class (or all classes) and a simple evaluator."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .detector import ClassObjective, Detector
from .diffusion import DiffusionPrior, GuidanceConfig
from .naming import Namer, name_class
from .prompt_search import PromptSearchResult, prompt_search
from .vocab import load_vocab, make_prompt


@dataclass
class ClassRecovery:
    class_idx: int
    search_words: List[Tuple[str, float]] = field(default_factory=list)     # from prompt search (detector score)
    guided_images: List[Image.Image] = field(default_factory=list)
    guided_scores: List[float] = field(default_factory=list)
    guided_trace: List[List[float]] = field(default_factory=list)
    clip_names: List[Tuple[str, float]] = field(default_factory=list)       # from CLIP on guided/search images
    captions: List[str] = field(default_factory=list)
    crops: List[Image.Image] = field(default_factory=list)
    crop_scores: List[float] = field(default_factory=list)
    prompt_used: str = ""
    combined: List[Tuple[str, float]] = field(default_factory=list)

    def combine(self, k: int = 10, w_search: float = 0.5) -> List[Tuple[str, float]]:
        """Rank-fuse prompt-search words and CLIP names into one candidate list."""
        score: Dict[str, float] = {}
        for r, (w, _) in enumerate(self.search_words[:k]):
            score[w] = score.get(w, 0.0) + w_search / (r + 1)
        for r, (w, _) in enumerate(self.clip_names[:k]):
            score[w] = score.get(w, 0.0) + (1 - w_search) / (r + 1)
        self.combined = sorted(score.items(), key=lambda kv: -kv[1])[:k]
        return self.combined

    def to_json(self) -> Dict[str, object]:
        return {"class_idx": self.class_idx, "search_words": self.search_words, "clip_names": self.clip_names,
                "combined": self.combined, "guided_scores": self.guided_scores, "captions": self.captions,
                "prompt_used": self.prompt_used}


def recover_class(detector: Detector, prior: DiffusionPrior, namer: Optional[Namer], class_idx: int,
                  search: Optional[PromptSearchResult] = None, guided: bool = True,
                  gcfg: GuidanceConfig = GuidanceConfig(), prompt: Optional[str] = None,
                  domain_prefix: str = "a photo of", words: Optional[Sequence[str]] = None,
                  vocab_path: Optional[str] = None, k: int = 10, seed_prompt_from_search: bool = True,
                  objective_kwargs: Optional[dict] = None,
                  on_step: Optional[Callable[[int, int, object], None]] = None) -> ClassRecovery:
    """Recover one class.

    If a prompt-search result is supplied, its top word seeds the prompt for
    guided sampling (``"<domain_prefix> a <word>"``); otherwise guided sampling
    starts from the generic ``"<domain_prefix> an object"`` and the detector's
    gradient has to do all the work.
    """
    rec = ClassRecovery(class_idx)
    words = list(words) if words is not None else load_vocab(vocab_path)
    if search is not None:
        rec.search_words = search.top_words(class_idx, k)
        imgs = list(search.best_images.get(class_idx, {}).values())
    else:
        imgs = []

    if guided:
        if prompt is None:
            if seed_prompt_from_search and rec.search_words:
                prompt = make_prompt(rec.search_words[0][0], domain_prefix + " a {}")
            else:
                prompt = f"{domain_prefix} an object"
        rec.prompt_used = prompt
        obj = ClassObjective(detector, class_idx, seed=gcfg.seed, **(objective_kwargs or {}))
        out = prior.guided_sample(prompt, obj, gcfg, on_step=on_step)
        rec.guided_images, rec.guided_scores, rec.guided_trace = out["images"], out["scores"], out["trace"]
        imgs = rec.guided_images + imgs

    if namer is not None and imgs:
        res = name_class(detector, namer, imgs, class_idx, words=words, k=k)
        rec.clip_names, rec.crops, rec.captions = res["ranked"], res["crops"], res["captions"]
        rec.crop_scores = [float(s) for s in res["scores"]]
    rec.combine(k)
    return rec


def recover_all(detector: Detector, prior: DiffusionPrior, namer: Optional[Namer], classes: Optional[Sequence[int]] = None,
                do_search: bool = True, guided: bool = True, search_kwargs: Optional[dict] = None,
                gcfg: GuidanceConfig = GuidanceConfig(), out_dir: Optional[str] = None, k: int = 10,
                progress: Optional[Callable[[str], None]] = None, **kw) -> Dict[int, ClassRecovery]:
    classes = list(classes) if classes is not None else list(range(detector.nc))
    search = None
    if do_search:
        if progress:
            progress("prompt search over vocabulary ...")
        search = prompt_search(detector, prior, **(search_kwargs or {}))
        if out_dir:
            search.save(os.path.join(out_dir, "search"))
    results: Dict[int, ClassRecovery] = {}
    for c in classes:
        if progress:
            progress(f"class {c}: " + ("guided sampling" if guided else "naming"))
        rec = recover_class(detector, prior, namer, c, search=search, guided=guided, gcfg=gcfg, k=k, **kw)
        results[c] = rec
        if out_dir:
            d = os.path.join(out_dir, f"class_{c:03d}")
            os.makedirs(d, exist_ok=True)
            for i, im in enumerate(rec.guided_images):
                im.save(os.path.join(d, f"guided_{i}.png"))
            for i, im in enumerate(rec.crops):
                im.save(os.path.join(d, f"crop_{i}.png"))
            json.dump(rec.to_json(), open(os.path.join(d, "result.json"), "w"), indent=2)
    return results


# ------------------------------------------------------------------ evaluation

_SYNONYMS = {
    "rhino": {"rhinoceros"}, "rhinoceros": {"rhino"}, "buffalo": {"bison", "water buffalo", "cape buffalo"},
    "tv": {"television", "monitor"}, "cell phone": {"smartphone", "phone", "mobile phone"},
    "motorcycle": {"motorbike"}, "airplane": {"aeroplane", "plane", "aircraft"}, "couch": {"sofa"},
    "dining table": {"table"}, "hair drier": {"hair dryer"}, "sports ball": {"ball"}, "person": {"man", "woman", "people"},
    "car": {"sedan", "automobile"}, "truck": {"pickup truck", "semi truck"}, "cup": {"mug"},
}


def _match(truth: str, cand: str) -> bool:
    t, c = truth.lower().strip(), cand.lower().strip()
    if t == c or t in c.split() or c in t.split():
        return True
    return c in _SYNONYMS.get(t, set()) or t in _SYNONYMS.get(c, set())


def evaluate(results: Dict[int, ClassRecovery], truth: Dict[int, str], ks: Sequence[int] = (1, 3, 5, 10),
             field: str = "combined") -> Dict[str, object]:
    """Top-k recovery rate against ground-truth names (exact / substring / small synonym table)."""
    ranks: Dict[int, Optional[int]] = {}
    for c, rec in results.items():
        cands = getattr(rec, field)
        r = next((i for i, (w, _) in enumerate(cands) if _match(truth.get(c, ""), w)), None)
        ranks[c] = r
    n = max(1, len(ranks))
    summary = {f"top{k}": sum(1 for r in ranks.values() if r is not None and r < k) / n for k in ks}
    summary["per_class"] = {int(c): {"truth": truth.get(c), "rank": (None if r is None else r + 1),
                                     "top": [w for w, _ in getattr(results[c], field)[:3]]} for c, r in ranks.items()}
    return summary
