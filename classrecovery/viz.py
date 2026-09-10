"""Contact sheets: one image per class showing what the diffusion model produced for it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont



@dataclass
class ClassResult:
    """Everything recovered for one class index."""
    class_idx: int
    images: List[Image.Image] = field(default_factory=list)      # best first
    scores: List[float] = field(default_factory=list)            # detector class score per image
    robust: List[float] = field(default_factory=list)            # min class score under degradations
    realness: List[float] = field(default_factory=list)          # prior reconstruction error (lower = more natural)
    source: List[str] = field(default_factory=list)              # "refined", "seed", "noise"
    modes: List[Tuple[int, float]] = field(default_factory=list) # (n members, robust) per discovered mode
    trace: List[List[float]] = field(default_factory=list)
    agreement: float = float("nan")                              # feature agreement of top images (confidence)
    note: str = ""


def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def class_sheet(rec: ClassResult, thumb: int = 256, max_images: int = 8, title: Optional[str] = None) -> Image.Image:
    """Header + one row of images for one class (best first), each captioned with source and scores."""
    rl = rec.realness or [float("nan")] * len(rec.images)
    tiles = [(im, f"{src}  det={sc:.2f}  degraded={rb:.2f}  err={r:.3f}") for im, sc, rb, r, src in
             zip(rec.images, rec.scores, rec.robust, rl, rec.source)][:max_images]
    tiles = tiles or [(Image.new("RGB", (thumb, thumb), (40, 40, 40)), "no images")]
    head_h, cap_h, pad = 64, 22, 6
    W = len(tiles) * (thumb + pad) + pad
    H = head_h + thumb + cap_h + pad * 2
    sheet = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    f_big, f_small = _font(20), _font(13)
    d.text((pad, 6), title or f"class {rec.class_idx}", fill=(255, 255, 255), font=f_big)
    modes = "  ".join(f"mode{i}: {n} imgs, robust {r:+.2f}" for i, (n, r) in enumerate(rec.modes)) or rec.note
    d.text((pad, 34), modes, fill=(200, 230, 200), font=f_small)
    x = pad
    for im, cap in tiles:
        sheet.paste(im.convert("RGB").resize((thumb, thumb)), (x, head_h))
        d.text((x + 2, head_h + thumb + 3), cap, fill=(220, 220, 220), font=f_small)
        x += thumb + pad
    return sheet


def contact_sheets(results: Dict[int, ClassResult], names: Optional[Dict[int, str]] = None, **kw) -> List[Image.Image]:
    out = []
    for c in sorted(results):
        title = f"class {c}" + (f"  ({names[c]})" if names and c in names else "")
        out.append(class_sheet(results[c], title=title, **kw))
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
