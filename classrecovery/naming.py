"""Turn recovered images into ranked candidate *names*.

The picture is an intermediate artefact; what the analyst wants is a word.
We crop each image to the detector's best box for the target class (so CLIP
looks at the object rather than the background), score the crop against a
vocabulary with CLIP zero-shot, and average across images.  Optionally a
captioner (BLIP) adds free-text descriptions as a sanity check.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .detector import ClassObjective, Detector
from .vocab import load_vocab


class Namer:
    def __init__(self, clip_model: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k",
                 device: Optional[str] = None, caption_model: Optional[str] = None):
        import open_clip

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained,
                                                                                device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(clip_model)
        self._text_cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        self.captioner = None
        if caption_model:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            self.cap_proc = BlipProcessor.from_pretrained(caption_model)
            self.captioner = BlipForConditionalGeneration.from_pretrained(caption_model).to(self.device).eval()

    @torch.no_grad()
    def text_features(self, words: Sequence[str], template: str = "a photo of a {}") -> torch.Tensor:
        key = (template,) + tuple(words)
        if key not in self._text_cache:
            toks = self.tokenizer([template.format(w) for w in words]).to(self.device)
            feats = []
            for i in range(0, len(toks), 256):
                f = self.model.encode_text(toks[i:i + 256])
                feats.append(f / f.norm(dim=-1, keepdim=True))
            self._text_cache[key] = torch.cat(feats)
        return self._text_cache[key]

    @torch.no_grad()
    def image_features(self, images: Sequence[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        f = self.model.encode_image(x)
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def rank(self, images: Sequence[Image.Image], words: Optional[Sequence[str]] = None,
             vocab_path: Optional[str] = None, k: int = 10, temperature: float = 100.0,
             weights: Optional[Sequence[float]] = None) -> List[Tuple[str, float]]:
        """Average CLIP zero-shot probabilities over ``images`` -> top-k (word, prob)."""
        words = list(words) if words is not None else load_vocab(vocab_path)
        tf = self.text_features(words)
        imf = self.image_features(images)
        probs = (temperature * imf @ tf.T).softmax(-1)          # (n_img, n_words)
        w = torch.ones(len(images), device=self.device) if weights is None else torch.tensor(weights, device=self.device).float()
        w = (w.clamp_min(0) + 1e-6)
        agg = (probs * w[:, None]).sum(0) / w.sum()
        order = agg.argsort(descending=True)[:k]
        return [(words[int(i)], float(agg[i])) for i in order]

    @torch.no_grad()
    def caption(self, images: Sequence[Image.Image], max_new_tokens: int = 20) -> List[str]:
        if self.captioner is None:
            return []
        out = []
        for im in images:
            inputs = self.cap_proc(im.convert("RGB"), return_tensors="pt").to(self.device)
            ids = self.captioner.generate(**inputs, max_new_tokens=max_new_tokens)
            out.append(self.cap_proc.decode(ids[0], skip_special_tokens=True))
        return out


def crop_to_class(detector: Detector, image: Image.Image, class_idx: int, pad: float = 0.15,
                  min_area: float = 0.01, max_area: float = 0.95) -> Tuple[Image.Image, float]:
    """Crop ``image`` to the highest-scoring box for ``class_idx`` (with padding).  Returns (crop, score)."""
    obj = ClassObjective(detector, class_idx, min_area=min_area, max_area=max_area, margin=0.0, n_aug=0)
    rep = obj.report(_to01(image))
    x0, y0, x1, y1 = rep["box"]
    W, H = image.size
    bw, bh = (x1 - x0), (y1 - y0)
    x0, y0 = max(0.0, x0 - pad * bw), max(0.0, y0 - pad * bh)
    x1, y1 = min(1.0, x1 + pad * bw), min(1.0, y1 + pad * bh)
    if x1 - x0 < 0.05 or y1 - y0 < 0.05:
        return image, rep["target_score"]
    return image.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))), rep["target_score"]


def _to01(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1).float() / 255.0


def name_class(detector: Detector, namer: Namer, images: Sequence[Image.Image], class_idx: int,
               words: Optional[Sequence[str]] = None, vocab_path: Optional[str] = None, k: int = 10,
               crop: bool = True, weight_by_score: bool = True) -> Dict[str, object]:
    """Full naming step for one class: crop -> CLIP rank (+captions if available)."""
    crops, scores = [], []
    for im in images:
        if crop:
            c, s = crop_to_class(detector, im, class_idx)
        else:
            c, s = im, 1.0
        crops.append(c)
        scores.append(s)
    ranked = namer.rank(crops, words, vocab_path, k=k, weights=scores if weight_by_score else None)
    return {"ranked": ranked, "crops": crops, "scores": scores, "captions": namer.caption(crops)}
