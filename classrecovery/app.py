"""Gradio GUI.  ``classrecovery app --share`` or, from a notebook::

    from classrecovery.app import launch
    launch(share=True)     # prints a public *.gradio.live link in Colab
"""

from __future__ import annotations

import io
import json
import os
import traceback
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .detector import Detector
from .diffusion import DiffusionPrior, GuidanceConfig
from .naming import Namer
from .pipeline import ClassRecovery, evaluate, recover_class
from .prompt_search import PromptSearchResult, prompt_search
from .vocab import load_vocab
from .weight_diff import WeightDiffReport, weight_diff

BASE_CHOICES = ["(none)", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
PRIOR_CHOICES = ["stabilityai/sd-turbo", "stable-diffusion-v1-5/stable-diffusion-v1-5",
                 "stabilityai/stable-diffusion-2-1-base"]


class Session:
    """Everything the GUI holds between clicks (single-user)."""

    def __init__(self):
        self.detector: Optional[Detector] = None
        self.base: Optional[Detector] = None
        self.report: Optional[WeightDiffReport] = None
        self.prior: Optional[DiffusionPrior] = None
        self.namer: Optional[Namer] = None
        self.search: Optional[PromptSearchResult] = None
        self.results: Dict[int, ClassRecovery] = {}
        self.vocab: List[str] = load_vocab()

    def get_prior(self, model_id: str) -> DiffusionPrior:
        if self.prior is None or self.prior.model_id != model_id:
            self.prior = DiffusionPrior(model_id)
        return self.prior

    def get_namer(self) -> Namer:
        if self.namer is None:
            self.namer = Namer()
        return self.namer


S = Session()


# ----------------------------------------------------------------- plotting helpers
def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _similarity_plot(rep: WeightDiffReport) -> Optional[Image.Image]:
    if rep.class_similarity is None:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = rep.class_similarity.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, min(12, n * 0.25)),) * 2)
    im = ax.imshow(rep.class_similarity, cmap="viridis", vmin=-0.2, vmax=1)
    ax.set_title("class-head row cosine similarity")
    if n <= 40:
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _fig_to_pil(fig)


def _trace_plot(rec: ClassRecovery) -> Optional[Image.Image]:
    if not rec.guided_trace:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, tr in enumerate(rec.guided_trace):
        ax.plot(tr, label=f"img {i}")
    ax.set_xlabel("denoising step"); ax.set_ylabel("class score"); ax.set_title(f"class {rec.class_idx} guidance trace")
    ax.legend(fontsize=7)
    return _fig_to_pil(fig)


# ----------------------------------------------------------------- callbacks
def load_models(mystery_file, mystery_path, base_choice, imgsz):
    try:
        path = mystery_file if isinstance(mystery_file, str) else (getattr(mystery_file, "name", None) or mystery_path)
        path = path or mystery_path
        if not path or not os.path.exists(path):
            return "Give a path to (or upload) a .pt checkpoint.", "", None, []
        S.detector = Detector(path, imgsz=int(imgsz))
        S.results, S.search = {}, None
        info = [f"loaded {path}: nc={S.detector.nc}, imgsz={S.detector.imgsz}, device={S.detector.device}",
                "names in checkpoint: " + ", ".join(f"{i}:{n}" for i, n in list(S.detector.names.items())[:20])
                + (" ..." if S.detector.nc > 20 else "")]
        rep_txt, heat = "", None
        if base_choice and base_choice != "(none)":
            S.base = Detector(base_choice, imgsz=int(imgsz))
            S.report = weight_diff(S.detector, S.base)
            rep_txt = S.report.summary()
            heat = _similarity_plot(S.report)
        rows = _class_rows()
        return "\n".join(info), rep_txt, heat, rows
    except Exception:
        return "error:\n" + traceback.format_exc(), "", None, []


def _class_rows():
    if S.detector is None:
        return []
    rows = []
    for c in range(S.detector.nc):
        drift = ("%.3f" % S.report.class_row_drift[c]) if (S.report is not None and S.report.class_row_drift is not None) else ""
        bias = ("%.1f" % S.report.class_bias[c]) if (S.report is not None and S.report.class_bias is not None) else ""
        nb = ", ".join(str(j) for j, _ in S.report.neighbours(c, 3)) if S.report is not None else ""
        match = ""
        if S.report is not None and S.report.row_match is not None:
            j, cs = S.report.row_match[c]
            match = f"{S.report.base_names.get(j, j)} ({cs:.2f})"
        srch = ", ".join(w for w, _ in S.search.top_words(c, 3)) if S.search is not None else ""
        rec = S.results.get(c)
        best = ", ".join(w for w, _ in rec.combined[:3]) if rec else ""
        rows.append([c, S.detector.names.get(c, ""), match, drift, bias, nb, srch, best])
    return rows


def run_search(prior_id, template, vocab_file, n_per_word, steps, res, progress=None):
    import gradio as gr
    progress = progress or gr.Progress()
    if S.detector is None:
        return "Load a detector first.", [], []
    try:
        prior = S.get_prior(prior_id)
        words = load_vocab(vocab_file if isinstance(vocab_file, str) else getattr(vocab_file, "name", None)) if vocab_file else S.vocab
        S.vocab = words

        def cb(done, total, p):
            progress(done / total, desc=f"{done}/{total}  {p}")

        S.search = prompt_search(S.detector, prior, words=words, template=template, n_per_word=int(n_per_word),
                                 steps=int(steps) if steps else None, height=int(res), width=int(res), progress=cb)
        gallery = []
        for c in range(S.detector.nc):
            for w, im in S.search.best_images.get(c, {}).items():
                gallery.append((im, f"class {c}: {w} ({S.search.scores[words.index(w), c]:.2f})"))
        return S.search.summary(5, S.detector.names), gallery, _class_rows()
    except Exception:
        return "error:\n" + traceback.format_exc(), [], []


def run_guided(prior_id, class_idx, prompt, domain_prefix, steps, cfg, strength, guide_from, guide_to, repeats,
               n_images, res, seed, min_area, max_area, margin, n_aug, use_clip, progress=None):
    import gradio as gr
    progress = progress or gr.Progress()
    if S.detector is None:
        return "Load a detector first.", [], None, [], []
    try:
        prior = S.get_prior(prior_id)
        namer = S.get_namer() if use_clip else None
        gcfg = GuidanceConfig(steps=int(steps), cfg=float(cfg), strength=float(strength), guide_from=float(guide_from),
                              guide_to=float(guide_to), repeats=int(repeats), height=int(res), width=int(res),
                              n_images=int(n_images), seed=int(seed) if seed not in (None, "", -1) else None)

        def on_step(i, n, _):
            progress((i + 1) / n, desc=f"denoising step {i + 1}/{n}")

        rec = recover_class(S.detector, prior, namer, int(class_idx), search=S.search, guided=True, gcfg=gcfg,
                            prompt=prompt.strip() or None, domain_prefix=domain_prefix, words=S.vocab,
                            objective_kwargs=dict(min_area=float(min_area), max_area=float(max_area),
                                                  margin=float(margin), n_aug=int(n_aug)), on_step=on_step)
        S.results[int(class_idx)] = rec
        gallery = [(im, f"score {s:.2f}") for im, s in zip(rec.guided_images, rec.guided_scores)]
        caps = rec.captions or [""] * len(rec.crops)
        crops = [(im, f"det score {s:.2f}" + (f" — {cap}" if cap else ""))
                 for im, s, cap in zip(rec.crops, rec.crop_scores, caps)]
        txt = [f"prompt used: {rec.prompt_used}"]
        if rec.search_words:
            txt.append("prompt-search words: " + ", ".join(f"{w} ({s:.2f})" for w, s in rec.search_words[:8]))
        if rec.clip_names:
            txt.append("CLIP names:          " + ", ".join(f"{w} ({p:.2f})" for w, p in rec.clip_names[:8]))
        txt.append("combined ranking:    " + ", ".join(f"{w}" for w, _ in rec.combined[:8]))
        return "\n".join(txt), gallery, _trace_plot(rec), crops, _class_rows()
    except Exception:
        return "error:\n" + traceback.format_exc(), [], None, [], []


def run_eval(truth_file, truth_path):
    try:
        path = truth_file if isinstance(truth_file, str) else (getattr(truth_file, "name", None) or truth_path)
        path = path or truth_path
        meta = json.load(open(path))
        truth = {int(k): v for k, v in meta.get("names", meta).items()}
        if not S.results:
            # fall back to prompt-search only
            if S.search is None:
                return "Nothing to evaluate yet."
            res = {c: ClassRecovery(c, search_words=S.search.top_words(c, 10)) for c in range(S.detector.nc)}
            for r in res.values():
                r.combine()
        else:
            res = S.results
        ev = evaluate(res, truth)
        lines = [f"top-{k}: {ev[f'top{k}']:.2f}" for k in (1, 3, 5, 10)]
        for c, d in ev["per_class"].items():
            lines.append(f"class {c}: truth={d['truth']!r} rank={d['rank']} top={d['top']}")
        return "\n".join(lines)
    except Exception:
        return "error:\n" + traceback.format_exc()


# ----------------------------------------------------------------- UI
def build():
    import gradio as gr

    with gr.Blocks(title="Detector class recovery") as demo:
        gr.Markdown("# Detector class recovery\nRecover what each class index of an undocumented YOLO checkpoint "
                    "responds to, using a diffusion model as the image prior.")
        with gr.Tab("1 · Model"):
            with gr.Row():
                mystery_file = gr.File(label="mystery checkpoint (.pt)", file_types=[".pt"])
                mystery_path = gr.Textbox(label="...or path on disk", value="testbed/mystery.pt")
            with gr.Row():
                base_choice = gr.Dropdown(BASE_CHOICES, value="yolov8n.pt", label="base checkpoint to diff against")
                imgsz = gr.Number(value=640, label="detector input size")
                load_btn = gr.Button("Load", variant="primary")
            info = gr.Textbox(label="model", lines=3)
            rep = gr.Textbox(label="weight-diff report", lines=8)
            heat = gr.Image(label="class-head similarity", type="pil")
        with gr.Tab("2 · Prompt search (gradient-free)"):
            with gr.Row():
                prior_id = gr.Dropdown(PRIOR_CHOICES, value=PRIOR_CHOICES[0], label="diffusion model", allow_custom_value=True)
                template = gr.Textbox(value="a photo of a {}", label="prompt template")
                vocab_file = gr.File(label="vocabulary (one term per line; blank = built-in ~600 nouns)")
            with gr.Row():
                n_per_word = gr.Slider(1, 4, value=2, step=1, label="images per word")
                s_steps = gr.Slider(1, 30, value=2, step=1, label="denoising steps (2 for sd-turbo)")
                s_res = gr.Dropdown([384, 512], value=512, label="resolution")
                search_btn = gr.Button("Run prompt search", variant="primary")
            search_txt = gr.Textbox(label="top words per class", lines=12)
            search_gallery = gr.Gallery(label="best image per class/word", columns=6, height=420)
        with gr.Tab("3 · Guided recovery (gradient)"):
            with gr.Row():
                g_prior = gr.Dropdown(PRIOR_CHOICES, value=PRIOR_CHOICES[0], label="diffusion model", allow_custom_value=True)
                class_idx = gr.Number(value=0, label="class index", precision=0)
                prompt = gr.Textbox(value="", label="prompt (blank = seed from prompt search / generic)")
                domain_prefix = gr.Textbox(value="a photo of", label="domain prefix")
            with gr.Row():
                g_steps = gr.Slider(2, 50, value=8, step=1, label="denoising steps")
                g_cfg = gr.Slider(0, 12, value=0, step=0.5, label="CFG scale (0 for turbo)")
                strength = gr.Slider(0, 0.3, value=0.08, step=0.01, label="guidance strength")
                repeats = gr.Slider(1, 5, value=2, step=1, label="grad steps / denoise step")
            with gr.Row():
                guide_from = gr.Slider(0, 1, value=1.0, step=0.05, label="guide from t/T")
                guide_to = gr.Slider(0, 1, value=0.2, step=0.05, label="guide to t/T")
                n_images = gr.Slider(1, 8, value=4, step=1, label="images")
                g_res = gr.Dropdown([384, 512], value=512, label="resolution")
                seed = gr.Number(value=0, label="seed (-1 random)", precision=0)
            with gr.Accordion("objective", open=False):
                with gr.Row():
                    min_area = gr.Slider(0, 0.5, value=0.02, step=0.01, label="min box area")
                    max_area = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label="max box area")
                    margin = gr.Slider(0, 1, value=0.5, step=0.1, label="specificity margin")
                    n_aug = gr.Slider(0, 4, value=2, step=1, label="augmentations")
                    use_clip = gr.Checkbox(value=True, label="name with CLIP")
            guided_btn = gr.Button("Recover class", variant="primary")
            g_txt = gr.Textbox(label="result", lines=5)
            with gr.Row():
                g_gallery = gr.Gallery(label="guided samples", columns=4, height=300)
                trace = gr.Image(label="guidance trace", type="pil")
            g_crops = gr.Gallery(label="crops used for naming", columns=6, height=200)
        with gr.Tab("4 · Overview / evaluate"):
            table = gr.Dataframe(headers=["class", "name in ckpt", "base row match", "row drift", "head bias", "nearest classes",
                                          "prompt-search top", "recovered top"], label="per-class overview",
                                 interactive=False)
            with gr.Row():
                truth_file = gr.File(label="truth.json (for the testbed)")
                truth_path = gr.Textbox(value="testbed/truth.json", label="...or path")
                eval_btn = gr.Button("Evaluate")
            eval_txt = gr.Textbox(label="evaluation", lines=10)

        load_btn.click(load_models, [mystery_file, mystery_path, base_choice, imgsz], [info, rep, heat, table])
        search_btn.click(run_search, [prior_id, template, vocab_file, n_per_word, s_steps, s_res],
                         [search_txt, search_gallery, table])
        guided_btn.click(run_guided, [g_prior, class_idx, prompt, domain_prefix, g_steps, g_cfg, strength, guide_from,
                                      guide_to, repeats, n_images, g_res, seed, min_area, max_area, margin, n_aug, use_clip],
                         [g_txt, g_gallery, trace, g_crops, table])
        eval_btn.click(run_eval, [truth_file, truth_path], [eval_txt])
    return demo


def launch(share: bool = False, **kw):
    demo = build()
    return demo.launch(share=share, **kw)


if __name__ == "__main__":
    launch(share=True)
