"""SHAP explanations: which words push an article towards FAKE."""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from .train import predict

log = logging.getLogger(__name__)


def explainer(model, tok, max_len):
    import shap

    def f(texts):
        p = predict(model, tok, [str(t) for t in texts], max_len)
        return np.stack([1 - p, p], 1)
    return shap.Explainer(f, shap.maskers.Text(tok), output_names=["AUTHENTIC", "FAKE"])


def shorten(t: str, n_words: int) -> str:
    return " ".join(str(t).split()[:n_words])


def explain(model, tok, texts: list[str], cfg):
    ex = explainer(model, tok, cfg.max_len)
    return ex([shorten(t, cfg.shap_words) for t in texts], max_evals=cfg.shap_max_evals, batch_size=32)


def html_report(sv, meta: pd.DataFrame) -> str:
    import shap
    blocks = []
    for i in range(len(meta)):
        r = meta.iloc[i]
        blocks.append(f"<h3>#{i + 1} — true: {'FAKE' if r.label else 'AUTHENTIC'} · predicted P(fake) = {r.p:.2f}</h3>"
                      + shap.plots.text(sv[i, :, 1], display=False))
    return ("<html><head><meta charset='utf-8'><title>SHAP — Bangla fake news</title></head><body>"
            "<p>Red words push the prediction towards FAKE, blue towards AUTHENTIC.</p>" + "<hr>".join(blocks) + "</body></html>")


def global_tokens(sv, min_count=3, top=25) -> pd.DataFrame:
    agg = defaultdict(list)
    for i in range(len(sv)):
        for tok, v in zip(sv[i].data, sv[i].values[:, 1]):
            t = str(tok).strip()
            if len(t) >= 2 and not t.isdigit():
                agg[t].append(float(v))
    rows = [(t, np.mean(v), len(v)) for t, v in agg.items() if len(v) >= min_count]
    df = pd.DataFrame(rows, columns=["token", "mean_shap_to_fake", "count"])
    return pd.concat([df.nlargest(top, "mean_shap_to_fake"), df.nsmallest(top, "mean_shap_to_fake")])


FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosansbengali/NotoSansBengali%5Bwdth,wght%5D.ttf"


def _bangla_font():
    """Best effort: register Noto Sans Bengali so token labels are not empty boxes."""
    from pathlib import Path

    from matplotlib import font_manager
    p = Path("/tmp/NotoSansBengali.ttf")
    try:
        if not p.exists():
            import requests
            r = requests.get(FONT_URL, timeout=30)
            r.raise_for_status()
            p.write_bytes(r.content)
        font_manager.fontManager.addfont(str(p))
        return font_manager.FontProperties(fname=str(p)).get_name()
    except Exception:
        return None


def plot_tokens(g: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fam = _bangla_font()
    g = g.sort_values("mean_shap_to_fake")
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(g) + 1))
    ax.barh(g.token, g.mean_shap_to_fake, color=["#c0392b" if v > 0 else "#2e86c1" for v in g.mean_shap_to_fake])
    ax.set_xlabel("mean SHAP towards FAKE")
    ax.set_title("Words that push towards FAKE (red) / AUTHENTIC (blue)")
    if fam:
        for lbl in ax.get_yticklabels():
            lbl.set_fontfamily(fam)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
