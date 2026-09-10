"""Contact sheets: one image per class showing what the diffusion model produced for it."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from .pipeline import ClassRecovery
from .prompt_search import PromptSearchResult


def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def class_sheet(rec: ClassRecovery, search: Optional[PromptSearchResult] = None, thumb: int = 256,
                max_images: int = 8, title: Optional[str] = None) -> Image.Image:
    """Compose a header + a row of images for one class.

    Left group: guided samples (what the detector's gradient steered the diffusion
    model to) with their detector score.  Right group: the best prompt-search
    images (what the model painted for the words that fired this class).
    """
    tiles: List[tuple] = []
    for im, s in zip(rec.guided_images, rec.guided_scores):
        tiles.append((im, f"guided  det={s:.2f}"))
    if search is not None:
        for w, im in list(search.best_images.get(rec.class_idx, {}).items())[:4]:
            tiles.append((im, f"search '{w}'"))
    tiles = tiles[:max_images] or [(Image.new("RGB", (thumb, thumb), (40, 40, 40)), "no images yet")]

    head_h, cap_h, pad = 64, 22, 6
    W = len(tiles) * (thumb + pad) + pad
    H = head_h + thumb + cap_h + pad * 2
    sheet = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    f_big, f_small = _font(20), _font(13)
    title = title or f"class {rec.class_idx}"
    d.text((pad, 6), title, fill=(255, 255, 255), font=f_big)
    names = ", ".join(w for w, _ in rec.combined[:5]) or "(not named)"
    clip = ", ".join(w for w, _ in rec.clip_names[:3])
    srch = ", ".join(w for w, _ in rec.search_words[:3])
    d.text((pad, 32), f"candidates: {names}", fill=(200, 230, 200), font=f_small)
    d.text((pad, 47), f"CLIP: {clip or '-'}   |   prompt search: {srch or '-'}", fill=(170, 170, 190), font=f_small)
    x = pad
    for im, cap in tiles:
        t = im.convert("RGB").resize((thumb, thumb))
        sheet.paste(t, (x, head_h))
        d.text((x + 2, head_h + thumb + 3), cap, fill=(220, 220, 220), font=f_small)
        x += thumb + pad
    return sheet


def contact_sheets(results: Dict[int, ClassRecovery], search: Optional[PromptSearchResult] = None,
                   names: Optional[Dict[int, str]] = None, **kw) -> List[Image.Image]:
    out = []
    for c in sorted(results):
        title = f"class {c}" + (f"  ({names[c]})" if names and c in names else "")
        out.append(class_sheet(results[c], search, title=title, **kw))
    return out


def stack(sheets: Sequence[Image.Image], gap: int = 8) -> Image.Image:
    """Stack per-class sheets vertically into one tall image (for notebooks / saving)."""
    if not sheets:
        return Image.new("RGB", (10, 10))
    W = max(s.width for s in sheets)
    H = sum(s.height for s in sheets) + gap * (len(sheets) - 1)
    out = Image.new("RGB", (W, H), (0, 0, 0))
    y = 0
    for s in sheets:
        out.paste(s, (0, y))
        y += s.height + gap
    return out
