from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)


def best_threshold(y, p) -> float:
    ts = np.linspace(0.05, 0.95, 91)
    return float(ts[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in ts]))])


def metrics(y, p, thr) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thr
    out = dict(n=len(y), n_fake=int(y.sum()), fake_f1=f1_score(y, pred, zero_division=0),
               macro_f1=f1_score(y, pred, average="macro", zero_division=0),
               fake_precision=precision_score(y, pred, zero_division=0), fake_recall=recall_score(y, pred, zero_division=0))
    if 0 < y.sum() < len(y):
        out.update(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p))
    return out


def table(name, sets: dict, thr) -> pd.DataFrame:
    return pd.DataFrame([dict(model=name, eval_set=k, **metrics(d.label, d.p, thr)) for k, d in sets.items()
                         if len(d) and d.label.nunique() > 1])


def by_group(test: pd.DataFrame, col, thr, min_n=50) -> pd.DataFrame:
    rows = []
    for g, d in test.groupby(col):
        if len(d) >= min_n and d.label.nunique() > 1:
            rows.append(dict(**{col: g}, **metrics(d.label, d.p, thr)))
    return pd.DataFrame(rows)


def plot_pr_roc(curves: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, p):.3f})")
        ax[1].plot(rc, pr, label=f"{name} ({average_precision_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=.7); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("ROC (AUROC)")
    ax[1].set_xlabel("Recall (fake)"); ax[1].set_ylabel("Precision (fake)"); ax[1].set_title("PR (AUPRC)")
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def detection_by_subtype(test: pd.DataFrame, thr, min_n=10) -> pd.DataFrame:
    """Share of each fake sub-type flagged as FAKE (recall) and of authentic articles wrongly flagged."""
    if "subtype" not in test:
        return pd.DataFrame()
    rows = []
    for g, d in test.groupby(test.subtype.fillna("unknown")):
        if len(d) >= min_n:
            rate = float((d.p >= thr).mean())
            rows.append(dict(subtype=g, n=len(d), flagged_as_fake=rate))
    return pd.DataFrame(rows).sort_values("subtype")
