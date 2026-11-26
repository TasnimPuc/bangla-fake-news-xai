"""Load BanFakeNews + BanFakeNews-2.0, harmonise columns, deduplicate across both, split.
Internally label 1 = FAKE (the class we want to detect); the source files use 1 = authentic."""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
HEAD = ("headline", "title", "heading")
BODY = ("content", "text", "article", "news", "body", "description")
LABEL = ("label", "class", "target", "is_fake")
CAT = ("category", "topic")


def _col(cols, names):
    low = {c.lower().strip(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    for n in names:
        for k, c in low.items():
            if n in k:
                return c
    return None


def normalize(t) -> str:
    t = unicodedata.normalize("NFC", str(t) if t is not None and t == t else "")
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _label(v):
    """Source convention: 1 / 'authentic' / 'real' / 'true' = authentic, 0 / 'fake' = fake. Returns is_fake."""
    s = str(v).strip().lower()
    if s in ("1", "1.0", "authentic", "real", "true"):
        return 0
    if s in ("0", "0.0", "fake", "false"):
        return 1
    return None


def read_csv_any(path: Path) -> pd.DataFrame | None:
    """News text often contains newlines/quotes that break the fast C parser and silently drop rows.
    Try several parsers and keep the one that yields the most rows with a valid 0/1 label."""
    import csv
    import sys
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    tries = [dict(), dict(engine="python"), dict(engine="python", escapechar="\\"),
             dict(engine="python", quoting=csv.QUOTE_NONE, sep=None)]
    best, best_n = None, -1
    for enc in ("utf-8", "utf-8-sig"):
        for kw in tries:
            try:
                df = pd.read_csv(path, encoding=enc, on_bad_lines="skip", **({"low_memory": False} if not kw else {}), **kw)
            except Exception:
                continue
            lc = _col(df.columns, LABEL)
            n = int(fake_labels(df[lc])[0].notna().sum()) if lc else 0
            if n > best_n:
                best, best_n = df, n
        if best_n > 0:
            break
    if best is None:
        log.warning("skip %s: unreadable", path)
        return None
    mb = path.stat().st_size / 1e6
    log.info("    parsed %s: %d labelled rows (%.1f MB, %.1f KB/row)", path.name, best_n, mb, 1000 * mb / max(best_n, 1))
    return best


def fake_labels(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """-> (is_fake, subtype). BanFakeNews(v1): 1 = authentic, 0 = fake.
    BanFakeNews-2.0 *_cleaned files use 4 classes: 3 = authentic (~80%, regular news), 0/1/2 = fake sub-types
    (satire-like / fabricated / clickbait-misleading by inspection). Mapping inferred from class sizes + examples."""
    v = pd.to_numeric(series, errors="coerce")
    vals = set(v.dropna().astype(int).unique())
    if vals and vals <= {0, 1, 2, 3} and vals & {2, 3}:
        is_fake = v.map(lambda x: None if pd.isna(x) else int(int(x) != 3))
        sub = v.map(lambda x: "authentic" if x == 3 else f"v2_type{int(x)}" if not pd.isna(x) else None)
        return is_fake, sub
    is_fake = series.map(_label)
    return is_fake, is_fake.map(lambda x: None if x is None or pd.isna(x) else ("fake" if x else "authentic"))


def harmonise(df: pd.DataFrame, origin: str, file: str) -> pd.DataFrame | None:
    h, b, l, c = (_col(df.columns, x) for x in (HEAD, BODY, LABEL, CAT))
    if l is None or (h is None and b is None):
        return None
    is_fake, sub = fake_labels(df[l])
    out = pd.DataFrame({
        "headline": df[h].map(normalize) if h else "",
        "content": df[b].map(normalize) if b else "",
        "label": is_fake,
        "subtype": sub,
        "category": df[c].astype(str) if c else "unknown",
    })
    out["origin"], out["file"] = origin, file
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    return out[(out.headline.str.len() + out.content.str.len()) > 20]


def find_csvs(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*.csv") if p.stat().st_size > 0)


def download(cfg) -> dict[str, Path]:
    roots = {}
    try:
        import kagglehub
        for ds in cfg.kaggle_datasets:
            roots[ds] = Path(kagglehub.dataset_download(ds))
            log.info("kagglehub %s -> %s", ds, roots[ds])
    except Exception as e:
        log.warning("kagglehub unavailable (%s); scanning %s", e, cfg.local_dirs)
    for d in cfg.local_dirs:
        if Path(d).exists():
            roots.setdefault(f"local:{d}", Path(d))
    return roots


def origin_of(path: Path) -> str:
    n = str(path).lower()
    return "v2" if ("hrithik" in n or "bangla-fake-news" in n or "banfakenews-2" in n or "_cleaned" in n) else "v1"


def github_v2(cfg) -> Path | None:
    """Official BanFakeNews-2.0 release (train/val/test_cleaned.7z) from the authors' GitHub."""
    try:
        import py7zr
        import requests
    except ImportError:
        log.warning("py7zr not installed -> skipping the GitHub copy of BanFakeNews-2.0")
        return None
    out = Path(cfg.work_dir) / "data" / "banfakenews-2.0-github"
    out.mkdir(parents=True, exist_ok=True)
    for name in cfg.v2_github_files:
        if list(out.glob(name.replace(".7z", "*.csv"))):
            continue
        r = requests.get(cfg.v2_github_base + name, timeout=300)
        r.raise_for_status()
        arc = out / name
        arc.write_bytes(r.content)
        with py7zr.SevenZipFile(arc, "r") as z:
            z.extractall(out)
        arc.unlink()
        log.info("GitHub BanFakeNews-2.0: %s extracted", name)
    return out


def load_all(cfg, roots: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    parts, indep, seen = [], None, set()

    def scan(root):
        nonlocal indep
        for p in find_csvs(root):
            sig = (p.name, p.stat().st_size)
            if sig in seen:                       # same file mounted under another path
                continue
            seen.add(sig)
            raw = read_csv_any(p)
            h = harmonise(raw, origin_of(p), p.name) if raw is not None else None
            if h is None or h.empty:
                log.info("  %s: no usable text/label rows, skipped", p.name)
                continue
            if "new_test" in p.name.lower():
                indep = h
                continue
            log.info("  %s [%s]: %d rows (fake %.1f%%)", p.name, h.origin.iloc[0], len(h), 100 * h.label.mean())
            parts.append(h)
    for _, root in roots.items():
        scan(root)
    n_v2 = sum(len(x) for x in parts if x.origin.iloc[0] == "v2")
    if cfg.use_github_v2 and n_v2 < cfg.v2_min_rows:
        log.warning("Only %d BanFakeNews-2.0 rows found on Kaggle (paper: ~60k) -> trying the authors' GitHub release", n_v2)
        gh = github_v2(cfg)
        if gh is not None:
            parts = [x for x in parts if x.origin.iloc[0] != "v2"]          # replace the incomplete copy
            seen.clear()
            scan(gh)
    if indep is None and cfg.independent_test_url:
        try:
            import io

            import requests
            r = requests.get(cfg.independent_test_url, timeout=60)
            r.raise_for_status()
            indep = harmonise(pd.read_csv(io.BytesIO(r.content)), "v2", "New_Test_Dataset.csv")
            log.info("independent test set: %d rows (fake %.1f%%)", len(indep), 100 * indep.label.mean())
        except Exception as e:
            log.warning("independent test set not available: %s", e)
    if not parts:
        raise RuntimeError("No usable CSVs found. Add the two Kaggle datasets as inputs or check internet access.")
    return pd.concat(parts, ignore_index=True), indep


def _key(h, c):
    s = re.sub(r"\s+", "", (h + c)[:2000])
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def dedupe(df: pd.DataFrame, indep: pd.DataFrame | None, cfg) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    df = df.copy()
    df["key"] = [_key(h, c) for h, c in zip(df.headline, df.content)]
    agg = df.groupby("key").agg(label_n=("label", "nunique"), label=("label", "first"), headline=("headline", "first"),
                                content=("content", "first"), category=("category", "first"), subtype=("subtype", "first"),
                                origin=("origin", lambda s: "+".join(sorted(set(s)))))
    n_conf = int((agg.label_n > 1).sum())
    agg = agg[agg.label_n == 1].drop(columns="label_n").reset_index()
    log.info("Dedup: %d rows -> %d unique articles (%d with conflicting labels dropped)", len(df), len(agg), n_conf)
    if indep is not None:
        indep = indep.copy()
        indep["key"] = [_key(h, c) for h, c in zip(indep.headline, indep.content)]
        overlap = agg.key.isin(set(indep.key))
        log.info("Removed %d training-pool articles that also appear in the independent test set", int(overlap.sum()))
        agg = agg[~overlap]
    return agg, indep


def text_of(df: pd.DataFrame, max_chars: int) -> pd.Series:
    return (df.headline.fillna("") + " । " + df.content.fillna("")).str.slice(0, max_chars)


def split(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Stratified by (label, origin) so both datasets and both classes appear in every split."""
    rng = np.random.default_rng(cfg.seed)
    df = df.copy()
    df["split"] = "train"
    tr, va, _ = cfg.split
    for _, g in df.groupby(["label", "origin"]):
        idx = rng.permutation(g.index.to_numpy())
        n = len(idx)
        df.loc[idx[int(tr * n):int((tr + va) * n)], "split"] = "val"
        df.loc[idx[int((tr + va) * n):], "split"] = "test"
    return df


def build(cfg, roots=None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    cache = Path(cfg.work_dir) / "data"
    if (cache / "pool_v4.parquet").exists():
        pool = pd.read_parquet(cache / "pool_v4.parquet")
        ind = pd.read_parquet(cache / "indep_v4.parquet") if (cache / "indep_v4.parquet").exists() else None
        return pool, ind
    raw, indep = load_all(cfg, roots if roots is not None else download(cfg))
    pool, indep = dedupe(raw, indep, cfg)
    pool = split(pool, cfg)
    pool["text"] = text_of(pool, cfg.max_chars)
    if indep is not None:
        indep["text"] = text_of(indep, cfg.max_chars)
    cache.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(cache / "pool_v4.parquet", index=False)
    if indep is not None:
        indep.to_parquet(cache / "indep_v4.parquet", index=False)
    log.info("Pool %d | fake %.1f%% | %s", len(pool), 100 * pool.label.mean(),
             pool.groupby(["split", "origin"]).label.agg(["count", "mean"]).round(3).to_dict())
    return pool, indep


def fingerprint(pool: pd.DataFrame) -> str:
    """Short hash of the training data -> checkpoints from a different dataset are never resumed."""
    h = hashlib.md5(("".join(sorted(pool.key)) + str(pool.label.sum())).encode()).hexdigest()
    return h[:8]
