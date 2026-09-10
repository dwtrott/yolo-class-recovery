"""Gradient-free class recovery: sweep a vocabulary through the diffusion model.

For every word we generate a few images, run the detector, and record the
maximum pre-NMS score of every class.  A class whose top-scoring words are
"zebra, horse, donkey" is not hard to name.  Works with any detector (only
needs forward passes) and gives *words* directly rather than pictures.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .detector import Detector
from .diffusion import DiffusionPrior
from .vocab import load_vocab, make_prompt


@dataclass
class PromptSearchResult:
    words: List[str]
    scores: np.ndarray                     # (n_words, nc): max over samples of max-over-anchors class score
    mean_scores: np.ndarray                # (n_words, nc): mean over samples
    best_images: Dict[int, Dict[str, Image.Image]] = field(default_factory=dict)  # class -> word -> best image
    template: str = "a photo of a {}"

    def top_words(self, class_idx: int, k: int = 10, use_mean: bool = False) -> List[tuple]:
        col = (self.mean_scores if use_mean else self.scores)[:, class_idx]
        order = np.argsort(-col)[:k]
        return [(self.words[i], float(col[i])) for i in order]

    def top_classes(self, word: str, k: int = 5) -> List[tuple]:
        row = self.scores[self.words.index(word)]
        order = np.argsort(-row)[:k]
        return [(int(c), float(row[c])) for c in order]

    def specificity(self, class_idx: int) -> float:
        """How peaked class ``class_idx`` is over the vocabulary (max / mean)."""
        col = self.scores[:, class_idx]
        return float(col.max() / (col.mean() + 1e-8))

    def summary(self, k: int = 5, names: Optional[Dict[int, str]] = None) -> str:
        lines = []
        for c in range(self.scores.shape[1]):
            top = ", ".join(f"{w} ({s:.2f})" for w, s in self.top_words(c, k))
            label = f"[{c}]" + (f" {names[c]}" if names and c in names else "")
            lines.append(f"{label:>24}: {top}")
        return "\n".join(lines)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "scores.npy"), self.scores)
        np.save(os.path.join(path, "mean_scores.npy"), self.mean_scores)
        with open(os.path.join(path, "words.json"), "w") as f:
            json.dump({"words": self.words, "template": self.template}, f)
        for c, d in self.best_images.items():
            for w, im in d.items():
                im.save(os.path.join(path, f"class{c:03d}_{w.replace(' ', '_')}.png"))

    @classmethod
    def load(cls, path: str) -> "PromptSearchResult":
        meta = json.load(open(os.path.join(path, "words.json")))
        return cls(meta["words"], np.load(os.path.join(path, "scores.npy")),
                   np.load(os.path.join(path, "mean_scores.npy")), template=meta.get("template", "a photo of a {}"))


def prompt_search(detector: Detector, prior: DiffusionPrior, words: Optional[Sequence[str]] = None,
                  vocab_path: Optional[str] = None, template: str = "a photo of a {}",
                  n_per_word: int = 2, batch: int = 8, steps: Optional[int] = None, seed: int = 0,
                  height: int = 512, width: int = 512, keep_images_per_class: int = 3,
                  min_area: float = 0.01, max_area: float = 0.95,
                  progress: Optional[Callable[[int, int, str], None]] = None) -> PromptSearchResult:
    """Sweep ``words`` (default: built-in vocabulary) and score every class."""
    words = list(words) if words is not None else load_vocab(vocab_path)
    prompts = [make_prompt(w, template) for w in words]
    nc = detector.nc
    max_scores = np.zeros((len(words), nc), dtype=np.float32)
    mean_scores = np.zeros((len(words), nc), dtype=np.float32)
    # running best images per class (small heap by score)
    best: Dict[int, List[tuple]] = {c: [] for c in range(nc)}

    flat = [(wi, r) for wi in range(len(words)) for r in range(n_per_word)]
    for b0 in range(0, len(flat), batch):
        chunk = flat[b0:b0 + batch]
        ps = [prompts[wi] for wi, _ in chunk]
        imgs = prior.generate(ps, steps=steps, seed=seed + b0, height=height, width=width)
        sc = detector.class_scores(imgs, min_area=min_area, max_area=max_area).numpy()  # (b, nc)
        for (wi, _), row, im in zip(chunk, sc, imgs):
            max_scores[wi] = np.maximum(max_scores[wi], row)
            mean_scores[wi] += row / n_per_word
            if keep_images_per_class > 0:
                for c in np.argsort(-row)[:3]:
                    lst = best[int(c)]
                    lst.append((float(row[c]), words[wi], im))
                    lst.sort(key=lambda t: -t[0])
                    del lst[keep_images_per_class:]
        if progress is not None:
            progress(min(b0 + batch, len(flat)), len(flat), ps[0])

    best_images = {c: {w: im for _, w, im in lst} for c, lst in best.items() if lst}
    return PromptSearchResult(words, max_scores, mean_scores, best_images, template)
