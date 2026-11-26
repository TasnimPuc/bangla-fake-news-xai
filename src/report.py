from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _t(df):
    f = lambda v: "–" if isinstance(v, float) and np.isnan(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def build(out: Path, res, by_origin, by_cat, tokens, s, by_sub=None) -> str:
    rep = out / "report"
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    for f in (out / "figures").glob("*.png"):
        shutil.copy(f, rep / "figures" / f.name)
    counts = pd.DataFrame(s["counts"]).rename(columns={"count": "articles", "sum": "fake"})
    cols = ["model", "eval_set", "n", "n_fake", "fake_f1", "macro_f1", "fake_precision", "fake_recall", "auroc", "auprc"]
    md = f"""# Results — Bangla Fake News Detection + Explainability

Data: **BanFakeNews** (Hossain et al., LREC 2020) **+ BanFakeNews-2.0** (Shibu et al., 2025), merged and
**deduplicated across both** ({s['pool']:,} unique articles, {s['fake_rate']:.1%} fake). Articles that also appear in the
BanFakeNews-2.0 independent test set ({s['independent_test']:,}) were removed from training. Split stratified by label × source.
Decision threshold tuned on validation for fake-F1: **{s['threshold']:.2f}**.

## Data
{_t(counts)}

## Results
{_t(res[[c for c in cols if c in res.columns]])}

![ROC / PR](figures/roc_pr.png)

## By source (pooled test) — does the model only learn one dataset's style?
{_t(by_origin) if len(by_origin) else '–'}

## Which kinds of fake news are caught? (pooled test)
BanFakeNews-2.0 marks fake articles with sub-types 0/1/2 (3 = authentic); by inspection type0 looks like satire,
type1 like fabricated news and type2 like clickbait/misleading. `authentic` row = false-alarm rate.

{_t(by_sub) if by_sub is not None and len(by_sub) else '–'}

## By news category (pooled test, ≥50 articles)
{_t(by_cat) if len(by_cat) else '–'}

## Why does it say "fake"? (SHAP)
Global view over the explained test articles (mean SHAP value per word):

![SHAP tokens](figures/shap_tokens.png)

Per-article highlights (red = towards FAKE): **[shap_examples.html](shap_examples.html)**

{_t(tokens.head(15).assign(mean_shap_to_fake=tokens.head(15).mean_shap_to_fake.astype(float)))}

---
Runtime {s['runtime_min']} min · imbalance handled with `{s['config']['loss']}` loss.
"""
    (rep / "REPORT.md").write_text(md, encoding="utf-8")
    return md
