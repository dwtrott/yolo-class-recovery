"""Command line entry points.

    classrecovery testbed  --out testbed --epochs 15
    classrecovery diff     --model testbed/mystery.pt --base yolov8n.pt
    classrecovery search   --model testbed/mystery.pt --out runs/search
    classrecovery recover  --model testbed/mystery.pt --classes 0 1 2 3 --out runs/recover
    classrecovery eval     --results runs/recover --truth testbed/truth.json
    classrecovery app      --share
"""

from __future__ import annotations

import argparse
import json
import os

from .detector import Detector
from .diffusion import DiffusionPrior, GuidanceConfig


def _p_model(p):
    p.add_argument("--model", required=True, help="mystery checkpoint (.pt)")
    p.add_argument("--imgsz", type=int, default=640)


def _p_prior(p):
    p.add_argument("--prior", default="stabilityai/sd-turbo")
    p.add_argument("--res", type=int, default=512)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="classrecovery")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("testbed", help="fine-tune a YOLO on non-COCO classes and strip its labels")
    p.add_argument("--out", default="testbed")
    p.add_argument("--dataset", default="african-wildlife.yaml")
    p.add_argument("--base", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--freeze", type=int, default=None)

    p = sub.add_parser("diff", help="weight-diff a fine-tune against its base");  _p_model(p)
    p.add_argument("--base", default="yolov8n.pt")

    p = sub.add_parser("search", help="prompt search over a vocabulary");  _p_model(p); _p_prior(p)
    p.add_argument("--vocab", default=None)
    p.add_argument("--template", default="a photo of a {}")
    p.add_argument("--n-per-word", type=int, default=2)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", default="runs/search")

    p = sub.add_parser("recover", help="guided recovery (+ naming) for classes");  _p_model(p); _p_prior(p)
    p.add_argument("--classes", type=int, nargs="*", default=None)
    p.add_argument("--no-search", action="store_true")
    p.add_argument("--search-dir", default=None, help="reuse a saved prompt-search result")
    p.add_argument("--vocab", default=None)
    p.add_argument("--domain-prefix", default="a photo of")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--cfg", type=float, default=0.0)
    p.add_argument("--strength", type=float, default=0.08)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--n-images", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-clip", action="store_true")
    p.add_argument("--out", default="runs/recover")

    p = sub.add_parser("eval", help="score saved results against truth.json")
    p.add_argument("--results", required=True)
    p.add_argument("--truth", required=True)

    p = sub.add_parser("app", help="launch the Gradio GUI")
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)

    a = ap.parse_args(argv)

    if a.cmd == "testbed":
        from .testbed import build_testbed
        print(json.dumps(build_testbed(a.out, a.dataset, a.base, a.epochs, a.imgsz, freeze=a.freeze), indent=2))

    elif a.cmd == "diff":
        from .weight_diff import weight_diff
        rep = weight_diff(Detector(a.model, a.imgsz), Detector(a.base, a.imgsz))
        print(rep.summary())

    elif a.cmd == "search":
        from .prompt_search import prompt_search
        det, prior = Detector(a.model, a.imgsz), DiffusionPrior(a.prior)
        res = prompt_search(det, prior, vocab_path=a.vocab, template=a.template, n_per_word=a.n_per_word,
                            steps=a.steps, height=a.res, width=a.res,
                            progress=lambda d, t, p: print(f"\r{d}/{t} {p:<40}", end=""))
        print()
        print(res.summary(5, det.names))
        res.save(a.out)

    elif a.cmd == "recover":
        from .naming import Namer
        from .pipeline import recover_all
        from .prompt_search import PromptSearchResult, prompt_search
        det, prior = Detector(a.model, a.imgsz), DiffusionPrior(a.prior)
        namer = None if a.no_clip else Namer()
        gcfg = GuidanceConfig(steps=a.steps, cfg=a.cfg, strength=a.strength, repeats=a.repeats,
                              n_images=a.n_images, seed=a.seed, height=a.res, width=a.res)
        search = PromptSearchResult.load(a.search_dir) if a.search_dir else None
        results = recover_all(det, prior, namer, a.classes, do_search=(search is None and not a.no_search),
                              guided=True, gcfg=gcfg, out_dir=a.out, progress=print, vocab_path=a.vocab,
                              domain_prefix=a.domain_prefix)
        if search is not None:
            for c, rec in results.items():
                rec.search_words = search.top_words(c, 10); rec.combine()
        for c, rec in results.items():
            print(f"class {c}: {[w for w, _ in rec.combined[:5]]}")

    elif a.cmd == "eval":
        from .pipeline import ClassRecovery, evaluate
        truth = {int(k): v for k, v in json.load(open(a.truth))["names"].items()}
        results = {}
        for d in sorted(os.listdir(a.results)):
            f = os.path.join(a.results, d, "result.json")
            if d.startswith("class_") and os.path.exists(f):
                j = json.load(open(f))
                results[j["class_idx"]] = ClassRecovery(j["class_idx"], search_words=[tuple(x) for x in j["search_words"]],
                                                        clip_names=[tuple(x) for x in j["clip_names"]],
                                                        combined=[tuple(x) for x in j["combined"]])
        print(json.dumps(evaluate(results, truth), indent=2))

    elif a.cmd == "app":
        from .app import launch
        launch(share=a.share, server_port=a.port)


if __name__ == "__main__":
    main()
