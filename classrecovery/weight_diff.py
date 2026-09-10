"""Compare an undocumented fine-tune against the public checkpoint it descends from.

Cheap, gradient-free, and often the first thing to run: it tells you which
parts of the network moved, whether the class head was re-initialised (class
count changed), which classes look like they kept their original meaning, and
how the class-head rows cluster (similar rows = semantically related classes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .detector import Detector


@dataclass
class WeightDiffReport:
    per_layer: List[Tuple[str, float]]               # (param name, relative change) for shape-matched params
    reshaped: List[Tuple[str, tuple, tuple]]         # params whose shape differs (base shape, new shape)
    per_block: Dict[str, float]                      # "model.<i>" -> mean relative change
    nc_base: int
    nc_new: int
    class_row_drift: Optional[np.ndarray] = None     # (nc,) relative change of each class-head row (same nc only)
    class_similarity: Optional[np.ndarray] = None    # (nc, nc) cosine similarity between class-head rows
    class_bias: Optional[np.ndarray] = None          # (nc,) mean class-head bias (frequency prior proxy)
    base_names: Dict[int, str] = field(default_factory=dict)
    row_match: Optional[List[Tuple[int, float]]] = None   # per new class: (most similar base class, cosine)

    def likely_unchanged(self, thresh: float = 0.05) -> List[int]:
        if self.class_row_drift is None:
            return []
        return [int(i) for i in np.where(self.class_row_drift < thresh)[0]]

    def neighbours(self, class_idx: int, k: int = 5) -> List[Tuple[int, float]]:
        if self.class_similarity is None:
            return []
        row = self.class_similarity[class_idx].copy()
        row[class_idx] = -np.inf
        order = np.argsort(-row)[:k]
        return [(int(j), float(row[j])) for j in order]

    def summary(self, top_blocks: int = 8) -> str:
        L = []
        L.append(f"classes: base nc={self.nc_base}, new nc={self.nc_new}"
                 + ("  -> class count changed; class rows were re-initialised (or remapped by name) — see row matching below"
                    if self.nc_base != self.nc_new else "  -> same class count; head rows are comparable"))
        if self.reshaped:
            L.append(f"{len(self.reshaped)} parameters changed shape (head re-init): " +
                     ", ".join(n for n, _, _ in self.reshaped[:6]) + (" ..." if len(self.reshaped) > 6 else ""))
        blocks = sorted(self.per_block.items(), key=lambda kv: -kv[1])
        L.append("most-changed blocks (relative L2 change): " +
                 ", ".join(f"{b}={v:.3f}" for b, v in blocks[:top_blocks]))
        least = sorted(self.per_block.items(), key=lambda kv: kv[1])[:4]
        L.append("least-changed blocks: " + ", ".join(f"{b}={v:.4f}" for b, v in least))
        if self.class_row_drift is not None:
            unchanged = self.likely_unchanged()
            changed = [i for i in range(self.nc_new) if i not in unchanged]
            L.append(f"class rows with <5% drift (probably still the base class): {unchanged}")
            L.append("class rows that moved a lot (candidates for changed/added classes): " +
                     ", ".join(f"{i}({self.class_row_drift[i]:.2f})" for i in
                               sorted(changed, key=lambda i: -self.class_row_drift[i])[:15]))
        if self.row_match is not None:
            L.append("class-head rows matched to base classes by cosine similarity "
                     "(>0.9 = the row was copied from the base, i.e. the class probably kept its meaning):")
            for i, (j, cs) in enumerate(self.row_match):
                tag = "KEPT" if cs > 0.9 else ("similar" if cs > 0.6 else "new")
                L.append(f"    class {i:>3} -> base {j:>3} {self.base_names.get(j, '?'):<16} cos={cs:.2f}  [{tag}]")
        if self.class_bias is not None:
            order = np.argsort(self.class_bias)
            L.append("lowest class-head bias (rarest classes in the fine-tune data): " +
                     ", ".join(f"{int(i)}({self.class_bias[i]:.1f})" for i in order[:8]))
            L.append("highest class-head bias (most frequent classes): " +
                     ", ".join(f"{int(i)}({self.class_bias[i]:.1f})" for i in order[::-1][:8]))
        return "\n".join(L)


def _class_bias(det: Detector) -> Optional[np.ndarray]:
    head = det.net.model[-1]
    cv3 = getattr(head, "cv3", None)
    if cv3 is None:
        return None
    rows = []
    for seq in cv3:
        last = None
        for m in seq.modules():
            if isinstance(m, torch.nn.Conv2d) and m.out_channels == det.nc and m.bias is not None:
                last = m
        if last is not None:
            rows.append(last.bias.detach().cpu().numpy())
    return np.mean(rows, 0) if rows else None


def class_similarity(det: Detector) -> Optional[np.ndarray]:
    mats = det.class_head_weights()
    if not mats:
        return None
    normed = [m / (m.norm(dim=1, keepdim=True) + 1e-8) for m in mats]
    W = torch.cat(normed, 1)                 # (nc, sum C)
    W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    return (W @ W.T).numpy()


def weight_diff(new: Detector, base: Detector) -> WeightDiffReport:
    sd_new, sd_base = new.state_dict(), base.state_dict()
    per_layer, reshaped = [], []
    block_acc: Dict[str, List[float]] = {}
    for k, v in sd_new.items():
        if k not in sd_base or not torch.is_floating_point(v):
            continue
        b = sd_base[k]
        if b.shape != v.shape:
            reshaped.append((k, tuple(b.shape), tuple(v.shape)))
            continue
        rel = float((v - b).norm() / (b.norm() + 1e-8))
        per_layer.append((k, rel))
        blk = ".".join(k.split(".")[:2])
        block_acc.setdefault(blk, []).append(rel)
    per_block = {b: float(np.mean(v)) for b, v in block_acc.items()}

    drift, row_match = None, None
    mn, mb = new.class_head_weights(), base.class_head_weights()
    if mn and mb and len(mn) == len(mb) and any(a.shape[1] != b.shape[1] for a, b in zip(mn, mb)):
        print(f"[classrecovery] class-head input width differs ({mn[0].shape[1]} vs {mb[0].shape[1]}): "
              "the class branch was rebuilt, so rows cannot be matched to base classes (biases may still carry over)")
    if mn and mb and len(mn) == len(mb) and all(a.shape[1] == b.shape[1] for a, b in zip(mn, mb)):
        sims = []
        for a, b in zip(mn, mb):
            a_n = a / (a.norm(dim=1, keepdim=True) + 1e-8)
            b_n = b / (b.norm(dim=1, keepdim=True) + 1e-8)
            sims.append(a_n @ b_n.T)                      # (nc_new, nc_base)
        S = torch.stack(sims).mean(0)
        row_match = [(int(j), float(S[i, j])) for i, j in enumerate(S.argmax(1))]
    if new.nc == base.nc:
        mn, mb = new.class_head_weights(), base.class_head_weights()
        if mn and mb and len(mn) == len(mb):
            d = [((a - b).norm(dim=1) / (b.norm(dim=1) + 1e-8)).numpy() for a, b in zip(mn, mb)]
            drift = np.mean(d, 0)
    return WeightDiffReport(per_layer, reshaped, per_block, base.nc, new.nc, drift,
                            class_similarity(new), _class_bias(new), dict(base.names), row_match)
