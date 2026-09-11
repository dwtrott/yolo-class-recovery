"""Honest evaluation: variants side by side, blind judging, scoring.

    from classrecovery.evaluate import run_variants, blind_package, score_blind

* ``run_variants``  runs retrieval-only / anchored / noise-only on the same model
                    and returns one row per (class, variant) with best source, scores
                    and the confidence flag — the table that says what the method is
* ``blind_package`` writes the sheets under shuffled anonymous ids with a separate
                    answer key, plus a CSV form for a judge who has not seen truth.json
* ``score_blind``   scores a filled form against the key and the true names
"""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Dict, List, Optional, Sequence

from .represent import represent
from .viz import ClassResult, class_sheet

VARIANTS = {
    "retrieval": dict(refine=False, n_noise=0),      # seeds only: the baseline the method must beat
    "anchored": dict(refine=True, n_noise=0),        # the method
    "noise": dict(pool=None, n_noise=8),             # the fallback, on its own
}


def run_variants(weights: str, pool: str, classes: Optional[Sequence[int]] = None, twin=None,
                 variants: Sequence[str] = ("retrieval", "anchored", "noise"), out_dir: str = "runs/eval",
                 **kw) -> Dict[str, object]:
    """Run the chosen variants; return ``{"rows": [...], "results": {variant: {class: ClassResult}}}``."""
    rows: List[dict] = []
    results: Dict[str, Dict[int, ClassResult]] = {}
    for v in variants:
        opts = dict(pool=pool, twin=twin, out_dir=os.path.join(out_dir, v), show=False)
        opts.update(VARIANTS[v]); opts.update(kw)
        if "pool" in VARIANTS[v]:
            opts["pool"] = VARIANTS[v]["pool"]
        res = represent(weights, classes=classes, **opts)
        results[v] = res
        for c, r in res.items():
            rows.append({"class": c, "variant": v, "best_source": r.source[0] if r.source else "-",
                         "det": round(r.scores[0], 3) if r.scores else None,
                         "degraded": round(r.robust[0], 3) if r.robust else None,
                         "recon_err": round(r.realness[0], 4) if r.realness else None,
                         "agreement": round(r.agreement, 3), "n_off_manifold": sum("off-manifold" in s for s in r.source)})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "variants.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return {"rows": rows, "results": results}


def table(rows: Sequence[dict]) -> str:
    cols = ["class", "variant", "best_source", "det", "degraded", "recon_err", "agreement", "n_off_manifold"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    out = [line, "-" * len(line)]
    for r in rows:
        out.append("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    return "\n".join(out)


def blind_package(results: Dict[int, ClassResult], out_dir: str = "runs/blind", seed: int = 0,
                  hide_scores: bool = True) -> str:
    """Write sheets as sheet_A.png, sheet_B.png ... in shuffled order, a judge form, and a separate key.

    Give the judge the ``sheets/`` folder and ``judge_form.csv`` only.  Keep ``key.json`` yourself.
    """
    os.makedirs(os.path.join(out_dir, "sheets"), exist_ok=True)
    classes = sorted(results)
    ids = [chr(ord("A") + i) for i in range(len(classes))]
    rnd = random.Random(seed); rnd.shuffle(ids)
    key = {}
    for c, sid in zip(classes, ids):
        r = results[c]
        if hide_scores:
            r2 = ClassResult(c, images=r.images, scores=[0.0] * len(r.images), robust=[0.0] * len(r.images),
                             realness=[0.0] * len(r.images), source=["?"] * len(r.images))
            sheet = class_sheet(r2, title=f"sheet {sid}")
        else:
            sheet = class_sheet(r, title=f"sheet {sid}")
        sheet.save(os.path.join(out_dir, "sheets", f"sheet_{sid}.png"))
        key[sid] = c
    with open(os.path.join(out_dir, "judge_form.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sheet", "what_is_it (one or two words)", "confidence 1-5"])
        for sid in sorted(key):
            w.writerow([sid, "", ""])
    with open(os.path.join(out_dir, "key.json"), "w") as f:
        json.dump(key, f, indent=2)
    return out_dir


_SYN = {"rhino": {"rhinoceros"}, "buffalo": {"bison", "water buffalo", "cape buffalo", "cow", "cattle", "bull", "ox"},
        "signature": {"handwriting", "scribble", "autograph", "writing"}, "elephant": set(), "zebra": set()}


def _match(truth: str, ans: str) -> bool:
    t, a = truth.lower().strip(), ans.lower().strip()
    if not a:
        return False
    if t in a or a in t:
        return True
    return any(s in a for s in _SYN.get(t, set()))


def score_blind(form_csv: str, key_json: str, truth_json: str) -> Dict[str, object]:
    """Per-sheet correctness of a judge's answers; synonyms allowed, substring match."""
    key = json.load(open(key_json))
    truth = {int(k): v for k, v in json.load(open(truth_json))["names"].items()}
    rows, hits = [], 0
    with open(form_csv) as f:
        for row in csv.DictReader(f):
            sid = row["sheet"].strip()
            c = key[sid]
            ans = row.get("what_is_it (one or two words)", "")
            ok = _match(truth[c], ans)
            hits += ok
            rows.append({"sheet": sid, "class": c, "truth": truth[c], "answer": ans, "correct": ok,
                         "confidence": row.get("confidence 1-5", "")})
    return {"accuracy": hits / max(1, len(rows)), "rows": rows}
