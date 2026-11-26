"""data (both datasets) -> TF-IDF baseline -> BanglaBERT (focal loss) -> eval (pooled test + independent test)
-> SHAP explanations -> .pth / HF export -> report."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import torch

from . import data, evaluate, explain
from .config import Config
from .train import baseline, history, predict, train

log = logging.getLogger("bfn")


def _hf_token(cfg):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(cfg.hf_token_secret)
    except Exception:
        return os.environ.get(cfg.hf_token_secret)


def run(cfg: Config, roots=None) -> dict:
    t0 = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out = Path(cfg.work_dir)
    for d in ("figures", "model", "report", "tables"):
        (out / d).mkdir(parents=True, exist_ok=True)
    store = None
    if cfg.project_id or cfg.local_store:
        from .store import make_store
        store = make_store(cfg)

    # %% Load + merge both datasets | BanFakeNews + BanFakeNews-2.0 → dedupe → stratified split. The independent test set stays separate.
    # show: pool.groupby(["split", "origin"]).label.agg(["count", "mean"]).rename(columns={"mean": "fake_rate"})
    pool, indep = data.build(cfg, roots)
    tr, va, te = (pool[pool.split == s] for s in ("train", "val", "test"))
    sets = {"pooled test": te}
    if indep is not None and len(indep):
        sets["independent test (BanFakeNews-2.0)"] = indep
    results, curves = [], {}

    # %% Baseline: TF-IDF + Logistic Regression | Class-balanced linear model as a reference point.
    # show: pd.concat(results, ignore_index=True) if results else "baseline skipped"
    if cfg.run_baseline:
        pv, *pe = baseline(tr, [va.text] + [d.text for d in sets.values()])
        thr_b = evaluate.best_threshold(va.label, pv)
        bsets = {k: d.assign(p=p) for (k, d), p in zip(sets.items(), pe)}
        results.append(evaluate.table("TF-IDF + LogReg (balanced)", bsets, thr_b))
        curves["TF-IDF + LogReg"] = (te.label, bsets["pooled test"].p)

    # %% Train BanglaBERT (focal loss, resumable) | Each epoch is mirrored to the bucket; a stopped session continues from the last checkpoint. The table shows every epoch, also the ones loaded from the checkpoint (★ = best).
    # show: history(out / "ckpt")
    model, tok = train(cfg, tr, va, out / "ckpt", store)
    # %% Evaluate | Pooled test + independent BanFakeNews-2.0 test. The threshold is chosen on validation.
    # show: res
    # show: by_sub
    thr = evaluate.best_threshold(va.label, predict(model, tok, va.text.tolist(), cfg.max_len))
    msets = {k: d.assign(p=predict(model, tok, d.text.tolist(), cfg.max_len)) for k, d in sets.items()}
    name = f"BanglaBERT ({cfg.loss} loss)"
    results.append(evaluate.table(name, msets, thr))
    curves[name] = (te.label, msets["pooled test"].p)
    res = pd.concat(results, ignore_index=True)
    res.to_csv(out / "tables" / "results.csv", index=False)
    test = msets["pooled test"]
    by_origin = evaluate.by_group(test, "origin", thr)
    by_cat = evaluate.by_group(test, "category", thr)
    by_origin.to_csv(out / "tables" / "by_origin.csv", index=False)
    by_cat.to_csv(out / "tables" / "by_category.csv", index=False)
    by_sub = evaluate.detection_by_subtype(test, thr)
    by_sub.to_csv(out / "tables" / "by_subtype.csv", index=False)
    evaluate.plot_pr_roc(curves, out / "figures" / "roc_pr.png")

    # %% Explain with SHAP | A mix of caught, missed and false-alarm articles: which words push towards "fake".
    # show: g.head(20)
    n = cfg.shap_n
    t = test.assign(pred=(test.p >= thr).astype(int))
    pick = pd.concat([t[(t.label == 1) & (t.pred == 1)].head(n // 2), t[(t.label == 1) & (t.pred == 0)].head(n // 4),
                      t[(t.label == 0) & (t.pred == 1)].head(n // 4)]).head(n)
    sv = explain.explain(model, tok, pick.text.tolist(), cfg)
    (out / "report" / "shap_examples.html").write_text(explain.html_report(sv, pick), encoding="utf-8")
    g = explain.global_tokens(sv)
    g.to_csv(out / "tables" / "shap_tokens.csv", index=False)
    explain.plot_tokens(g, out / "figures" / "shap_tokens.png")

    # %% Export + report | `.pth`, Hugging Face folder, report, optional Hub upload, copy to the bucket.
    md = out / "model"
    model.save_pretrained(md / "hf"); tok.save_pretrained(md / "hf")
    torch.save({"state_dict": model.state_dict(), "base_model": cfg.model_name, "max_len": cfg.max_len,
                "threshold": thr, "id2label": {0: "AUTHENTIC", 1: "FAKE"}}, md / "banglabert_fake_news.pth")
    (md / "hf" / "threshold.json").write_text(json.dumps({"threshold": thr}))
    summary = dict(pool=len(pool), fake_rate=float(pool.label.mean()),
                   counts=pool.groupby(["split", "origin"]).label.agg(["count", "sum"]).reset_index().to_dict("records"),
                   independent_test=len(indep) if indep is not None else 0, threshold=thr,
                   results=res.to_dict("records"), runtime_min=round((time.time() - t0) / 60, 1), config=cfg.to_dict())
    (out / "metrics.json").write_text(json.dumps(summary, indent=1, default=float, ensure_ascii=False))
    from .report import build as build_report
    build_report(out, res, by_origin, by_cat, g, summary, by_sub)

    tok_hf = _hf_token(cfg)
    if tok_hf and (cfg.hub_model_repo or cfg.hub_space_repo):
        from .hub import push_all
        summary["hub"] = push_all(cfg, out, tok_hf)
    if store is not None:
        store.put_dir(out / "report", f"runs/{cfg.run_name}/report")
        store.put(md / "banglabert_fake_news.pth", f"runs/{cfg.run_name}/model/banglabert_fake_news.pth")
    r = res[(res.model == name) & (res.eval_set == "pooled test")].iloc[0]
    log.info("DONE in %.1f min | %s: fake-F1 %.3f macro-F1 %.3f AUROC %.3f", summary["runtime_min"], name,
             r.fake_f1, r.macro_f1, r.auroc)
    return summary
