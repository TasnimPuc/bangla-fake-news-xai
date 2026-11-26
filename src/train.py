"""BanglaBERT fine-tuning with focal loss (imbalance) and checkpoints mirrored to the GCS bucket (resume)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def focal_loss(logits, y, gamma, alpha_pos):
    ce = F.cross_entropy(logits.float(), y, reduction="none")
    pt = torch.exp(-ce)
    alpha = torch.where(y == 1, torch.full_like(ce, alpha_pos), torch.full_like(ce, 1 - alpha_pos))
    return (alpha * (1 - pt) ** gamma * ce).mean()


def _metrics(ev):
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    logits, y = ev
    p = torch.softmax(torch.tensor(logits).float(), -1)[:, 1].numpy()
    return {"fake_f1": f1_score(y, p >= 0.5), "macro_f1": f1_score(y, p >= 0.5, average="macro"),
            "auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p)}


def _load(name, **kw):
    from transformers import AutoModelForSequenceClassification
    try:
        m = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float32, **kw)
    except TypeError:
        m = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float32, **kw)
    return m.float()        # fp32 master weights for mixed precision


def train(cfg, tr: pd.DataFrame, va: pd.DataFrame, out_dir: Path, store=None):
    from datasets import Dataset
    from transformers import (AutoTokenizer, DataCollatorWithPadding, Trainer, TrainerCallback,
                              TrainingArguments)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = _load(cfg.model_name, num_labels=2, id2label={0: "AUTHENTIC", 1: "FAKE"},
                  label2id={"AUTHENTIC": 0, "FAKE": 1}, ignore_mismatched_sizes=True)
    pos = float(tr.label.mean())
    ce_w = torch.tensor([pos, 1 - pos]) / max(pos, 1 - pos)

    class ImbTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            y = inputs.pop("labels")
            out = model(**inputs)
            if cfg.loss == "focal":
                loss = focal_loss(out.logits, y, cfg.focal_gamma, cfg.focal_alpha_fake)
            elif cfg.loss == "weighted_ce":
                loss = F.cross_entropy(out.logits.float(), y, weight=ce_w.to(y.device))
            else:
                loss = F.cross_entropy(out.logits.float(), y)
            return (loss, out) if return_outputs else loss

    from .data import fingerprint
    ck = f"ckpt/{cfg.run_name}-{fingerprint(pd.concat([tr, va]))}"     # new data -> fresh checkpoints

    class Mirror(TrainerCallback):
        """Copy each saved checkpoint to the bucket so a dead Kaggle session can resume."""

        def on_save(self, args, state, control, **kw):
            if store is None:
                return
            last = max(Path(args.output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
            store.put_dir(last, f"{ck}/{last.name}")
            (Path(args.output_dir) / "LATEST").write_text(last.name)
            store.put(Path(args.output_dir) / "LATEST", f"{ck}/LATEST")
            log.info("checkpoint %s mirrored to bucket", last.name)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if store is not None and not list(out_dir.glob("checkpoint-*")) and store.get(f"{ck}/LATEST", out_dir / "LATEST"):
        name = (out_dir / "LATEST").read_text().strip()
        for k in store.list(f"{ck}/{name}"):
            store.get(k, out_dir / name / Path(k).relative_to(f"{ck}/{name}"))
        log.info("RESUME from bucket checkpoint %s", name)

    def enc(df):
        ds = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}).reset_index(drop=True))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg.max_len), batched=True,
                      remove_columns=["text"])
    cuda = torch.cuda.is_available()
    steps = cfg.epochs * -(-len(tr) // (cfg.batch_size * cfg.grad_accum))
    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=cfg.epochs, learning_rate=cfg.lr, weight_decay=0.01,
        per_device_train_batch_size=cfg.batch_size, per_device_eval_batch_size=64,
        gradient_accumulation_steps=cfg.grad_accum, warmup_steps=int(0.06 * steps),
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="fake_f1",
        save_total_limit=1, fp16=cuda, logging_steps=100, report_to=[], seed=cfg.seed, dataloader_num_workers=2)
    trainer = ImbTrainer(model=model, args=args, train_dataset=enc(tr), eval_dataset=enc(va), processing_class=tok,
                         data_collator=DataCollatorWithPadding(tok), compute_metrics=_metrics, callbacks=[Mirror()])
    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    saved = _saved_epochs(ckpts[-1] if ckpts else None)
    if saved:
        log.info("loaded checkpoint %s: %d of %d epochs already done", ckpts[-1].name, saved, cfg.epochs)
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    hist = _epoch_table(trainer.state.log_history, "fake_f1", trainer.state.best_metric, saved)
    hist.to_csv(out_dir / "history.csv", index=False)
    log.info("training history (all epochs):\n%s", hist.to_string(index=False) if len(hist) else "-")
    best = trainer.state.best_model_checkpoint
    if best and Path(best).exists():
        # reload with from_pretrained: Trainer's in-place reload misses renamed keys (LayerNorm gamma/beta) on 5.x
        model = _load(best).to(trainer.model.device)
        log.info("best checkpoint reloaded: %s (val fake-F1 %.4f)", Path(best).name, trainer.state.best_metric or float("nan"))
    else:
        model = trainer.model
    return model.eval(), tok


@torch.no_grad()
def predict(model, tok, texts, max_len, bs=64) -> np.ndarray:
    dev = next(model.parameters()).device
    out = []
    for b in range(0, len(texts), bs):
        e = tok(list(texts[b:b + bs]), truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(torch.softmax(model(**e).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def baseline(tr: pd.DataFrame, texts_eval: list[pd.Series]):
    """TF-IDF (word 1-2 + char 3-5) + class-balanced logistic regression."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=150_000, sublinear_tf=True)
    t = tr.text.str.slice(0, 2000)
    X = sp.hstack([w.fit_transform(t), c.fit_transform(t)]).tocsr()
    lr = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000, solver="liblinear").fit(X, tr.label)
    return [lr.predict_proba(sp.hstack([w.transform(s.str.slice(0, 2000)), c.transform(s.str.slice(0, 2000))]).tocsr())[:, 1]
            for s in texts_eval]


def _epoch_table(log_history, best_metric_key, best_metric, saved_epochs):
    """Trainer log_history -> one row per epoch (train loss + validation metrics)."""
    import math
    loss, rows = {}, []
    for h in log_history:
        ep = h.get("epoch")
        if ep is None:
            continue
        if "loss" in h:
            loss[max(1, math.ceil(ep - 1e-9))] = h["loss"]
        if "eval_loss" in h:
            e = int(round(ep))
            r = {"epoch": e, "train_loss": loss.get(e)}
            r.update({k[5:]: v for k, v in h.items() if k.startswith("eval_") and
                      not k.endswith(("runtime", "per_second"))})
            r["source"] = "checkpoint" if e <= saved_epochs else "this session"
            rows.append(r)
    df = pd.DataFrame(rows)
    if len(df) and best_metric is not None and best_metric_key in df:
        hit = np.flatnonzero(np.isclose(df[best_metric_key], best_metric))
        df["best"] = ["★" if i == (hit[0] if len(hit) else -1) else "" for i in range(len(df))]   # first best = kept
    return df


def _saved_epochs(ckpt_dir) -> int:
    import json
    f = Path(ckpt_dir) / "trainer_state.json" if ckpt_dir else None
    return int(round(json.loads(f.read_text()).get("epoch", 0))) if f and f.exists() else 0


def history(out_dir) -> pd.DataFrame:
    """Per-epoch values of the last training in out_dir, including epochs restored from a checkpoint."""
    f = Path(out_dir) / "history.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame({"info": ["no training history yet"]})
