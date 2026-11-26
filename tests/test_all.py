import pytest
import numpy as np
import pandas as pd

from src import data
from src.config import Config

FAKE_W = ["চাঞ্চল্যকর", "গুজব", "অবিশ্বাস্য", "ভাইরাল", "শেয়ার করুন"]
REAL_W = ["মন্ত্রণালয়", "সংবাদ সম্মেলনে", "জানিয়েছেন", "প্রতিবেদন", "কর্মকর্তা"]


def _article(r, fake, i):
    w = FAKE_W if fake else REAL_W
    return " ".join(r.choice(w, 6)) + f" খবর {i} " + " ".join(r.choice(FAKE_W + REAL_W, 6))


def _roots(tmp):
    r = np.random.default_rng(0)
    v1, v2 = tmp / "banfakenews", tmp / "bangla-fake-news"
    v1.mkdir(); v2.mkdir()
    mk = lambda n, fake, off: pd.DataFrame({"articleID": range(off, off + n), "domain": "x", "date": "2019",
                                            "category": r.choice(["Politics", "Sports"], n),
                                            "headline": [f"শিরোনাম {off + i}" for i in range(n)],
                                            "content": [_article(r, fake, off + i) for i in range(n)],
                                            "label": 0 if fake else 1})
    auth = mk(300, False, 0)
    auth.to_csv(v1 / "Authentic-48K.csv", index=False)
    mk(40, True, 1000).to_csv(v1 / "Fake-1K.csv", index=False)
    fake2 = mk(120, True, 2000)
    v2df = pd.concat([auth.head(150), fake2])                            # 150 authentic overlap v1 (dedup test)
    v2df.rename(columns={"content": "Content", "label": "Label"}).to_csv(v2 / "train.csv", index=False)
    pd.concat([fake2.head(10), mk(10, False, 5000)]).to_csv(v2 / "New_Test_Dataset.csv", index=False)  # 10 overlap v2 fakes
    return {"cryptexcode/banfakenews": v1, "hrithikmajumdar/bangla-fake-news": v2}


def test_load_dedupe_split(tmp_path):
    cfg = Config(work_dir=str(tmp_path / "o"), independent_test_url="", use_github_v2=False)
    pool, indep = data.build(cfg, _roots(tmp_path))
    assert len(indep) == 20
    assert pool.key.is_unique and not pool.key.isin(set(indep.key)).any()        # no leakage into independent test
    assert set(pool.origin) >= {"v1", "v1+v2", "v2"}
    assert set(pool.split) == {"train", "val", "test"} and pool.label.isin([0, 1]).all()
    assert len(pool) == 300 + 40 + 110 and pool.label.sum() == 40 + 110   # 10 fakes moved out (in independent test)                # 1 = FAKE internally


def _tiny(path, texts):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
    t = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]")
    BertForSequenceClassification(BertConfig(vocab_size=t.get_vocab_size(), hidden_size=32, num_hidden_layers=1,
                                             num_attention_heads=2, intermediate_size=64)).save_pretrained(path)
    tok.save_pretrained(path)


@pytest.mark.parametrize("mode", ["run", "notebook"])
def test_pipeline(tmp_path, mode):
    from src.pipeline import run
    roots = _roots(tmp_path)
    allt = [f"{w} খবর শিরোনাম ।" for w in FAKE_W + REAL_W] + [str(i) for i in range(3000)]
    _tiny(tmp_path / "tiny", allt)
    cfg = Config(work_dir=str(tmp_path / "o"), model_name=str(tmp_path / "tiny"), epochs=2, batch_size=16,
                 shap_n=6, shap_max_evals=40, independent_test_url="", use_github_v2=False, local_store=str(tmp_path / "bucket"))
    s = (run if mode == "run" else _steps)(cfg, roots=roots)
    assert {r["eval_set"] for r in s["results"]} == {"pooled test", "independent test (BanFakeNews-2.0)"}
    assert (tmp_path / "o" / "model" / "banglabert_fake_news.pth").exists()
    assert (tmp_path / "o" / "report" / "shap_examples.html").stat().st_size > 1000
    assert "Why does it say" in (tmp_path / "o" / "report" / "REPORT.md").read_text()
    assert list((tmp_path / "bucket").rglob("LATEST"))            # checkpoints mirrored -> resumable


def test_robust_csv_and_double_mount(tmp_path):
    """Backslash-escaped quotes + embedded newlines (as in the BanFakeNews-2.0 export) must not lose rows,
    and the same file mounted twice must be read once."""
    import csv
    d = tmp_path / "hrithikmajumdar" / "bangla-fake-news"
    d.mkdir(parents=True)
    rows = [{"headline": f"শিরোনাম {i}", "content": f'তিনি বলেন \"সত্য\" কথা\nনতুন লাইন {i}', "label": i % 2} for i in range(200)]
    with open(d / "train_cleaned.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["headline", "content", "label"], quoting=csv.QUOTE_ALL, escapechar="\\",
                           doublequote=False)
        w.writeheader(); w.writerows(rows)
    raw = data.read_csv_any(d / "train_cleaned.csv")
    assert len(raw) == 200
    mirror = tmp_path / "datasets"
    import shutil
    shutil.copytree(tmp_path / "hrithikmajumdar", mirror / "hrithikmajumdar")
    cfg = Config(work_dir=str(tmp_path / "o"), independent_test_url="", use_github_v2=False)
    pool, _ = data.load_all(cfg, {"a": tmp_path / "hrithikmajumdar", "local": mirror})
    assert len(pool) == 200 and set(pool.origin) == {"v2"}


def test_four_class_labels_of_banfakenews2():
    raw = pd.DataFrame({"Headline": ["ক", "খ", "গ", "ঘ"], "Content": ["এটি একটি খবরের লেখা যথেষ্ট লম্বা"] * 4,
                        "Label": [3, 0, 1, 2]})
    h = data.harmonise(raw, "v2", "val_cleaned.csv")
    assert h.label.tolist() == [0, 1, 1, 1]                       # 3 = authentic, 0/1/2 = fake sub-types
    assert h.subtype.tolist() == ["authentic", "v2_type0", "v2_type1", "v2_type2"]
    v1 = data.harmonise(raw.assign(Label=[1, 0, 1, 0]), "v1", "x.csv")
    assert v1.label.tolist() == [0, 1, 0, 1]                      # v1: 1 = authentic


def _steps(cfg, **kw):
    """Runs the notebook's step cells (built from src/pipeline.py::run) instead of run()."""
    import sys
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    from nbsteps import exec_steps
    return exec_steps(root / "src" / "pipeline.py", cfg, **kw)["result"]


def test_rerun_loads_checkpoint_and_shows_saved_epochs(tmp_path):
    """New session (empty work_dir, same bucket) -> training is not repeated, saved epochs are shown."""
    from src.pipeline import run
    from src.train import history
    roots = _roots(tmp_path)
    _tiny(tmp_path / "tiny", [f"{w} খবর শিরোনাম ।" for w in FAKE_W + REAL_W] + [str(i) for i in range(3000)])
    kw = dict(model_name=str(tmp_path / "tiny"), epochs=2, batch_size=16, shap_n=4, shap_max_evals=20,
              independent_test_url="", use_github_v2=False, local_store=str(tmp_path / "bucket"), run_baseline=False)
    run(Config(work_dir=str(tmp_path / "s1"), **kw), roots)
    h1 = history(tmp_path / "s1" / "ckpt")
    assert len(h1) == 2 and set(h1.source) == {"this session"} and (h1.best == "★").sum() >= 1
    run(Config(work_dir=str(tmp_path / "s2"), **kw), roots)
    h2 = history(tmp_path / "s2" / "ckpt")
    assert list(h2.epoch) == [1, 2] and set(h2.source) == {"checkpoint"}
    assert np.allclose(h1.fake_f1, h2.fake_f1)
