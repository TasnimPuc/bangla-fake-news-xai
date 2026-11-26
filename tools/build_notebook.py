"""Builds notebooks/kaggle_run.ipynb from the repo (self-contained). Run: python tools/build_notebook.py"""
from pathlib import Path

import sys

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbsteps import cells as step_cells  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Bangla Fake News Detection + Explainability (Kaggle)

**Before Run All:** GPU **T4** · Internet **On**. Both datasets are pulled automatically with `kagglehub`
(`cryptexcode/banfakenews` + `hrithikmajumdar/bangla-fake-news`); you can also attach them via *Add Input*.
Optional: Add-ons → Google Cloud (Cloud Storage) for checkpoint backup/resume, and secret `HF_TOKEN` to publish.

Pipeline (~1–1.5 h): merge + dedupe both datasets → TF-IDF baseline → BanglaBERT with focal loss (checkpoint each
epoch → bucket) → pooled test + BanFakeNews-2.0 independent test → SHAP → `.pth`, HF model, Streamlit Space, report."""

INSTALL = """!pip install -q shap py7zr "huggingface_hub>=1.3,<2.0" 2>&1 | tail -1
import importlib.metadata as md, torch
print({p: md.version(p) for p in ("transformers", "huggingface_hub", "shap", "torch")}, "| GPUs:", torch.cuda.device_count())"""

CONFIG = """import sys
sys.path.insert(0, "/kaggle/working/repo")
from src.config import Config

HF_USER = "YOUR_HF_USERNAME"          # <-- set to publish model + demo (needs HF_TOKEN secret)
publish = HF_USER != "YOUR_HF_USERNAME"
cfg = Config(
    work_dir="/kaggle/working/bfn",
    project_id="mimic-cxr-training",   # GCS bucket for checkpoints/resume ("" to skip)
    loss="focal",                      # focal | weighted_ce | ce
    epochs=3, max_len=384,
    hub_model_repo=f"{HF_USER}/banglabert-fake-news" if publish else "",
    hub_space_repo=f"{HF_USER}/bangla-fake-news-demo" if publish else "",
)"""


SHOW = """import re, shutil, glob
from IPython.display import Markdown, Image, HTML, display
rep = f"{cfg.work_dir}/report"
md = open(f"{rep}/REPORT.md", encoding="utf-8").read()
display(Markdown(re.sub(r"!\\[.*?\\]\\(figures/.*?\\)", "", md)))
for f in sorted(glob.glob(f"{rep}/figures/*.png")):
    display(Image(f))
display(HTML(open(f"{rep}/shap_examples.html", encoding="utf-8").read()[:200000]))
shutil.make_archive("/kaggle/working/results_for_github", "zip", rep)
print("Results zip  : /kaggle/working/results_for_github.zip -> unzip into the repo as results/")
print("Model (.pth) :", f"{cfg.work_dir}/model/banglabert_fake_news.pth")"""


def main():
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_markdown_cell("## 1 · Setup"), nbf.v4.new_code_cell(INSTALL),
             nbf.v4.new_markdown_cell("## 2 · Project code\nThese cells write the package to `/kaggle/working/repo`. "
                                      "No need to edit them; the steps further down call this code."),
             nbf.v4.new_code_cell("!mkdir -p /kaggle/working/repo/src /kaggle/working/repo/app /kaggle/working/repo/space")]
    files = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "app").glob("*.py")) + sorted((ROOT / "space").glob("*"))
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        cells.append(nbf.v4.new_code_cell(f"%%writefile /kaggle/working/repo/{rel}\n" + f.read_text(encoding="utf-8")))
    cells += [nbf.v4.new_markdown_cell("## 3 · Config\nChange only this cell."), nbf.v4.new_code_cell(CONFIG)]
    steps, n = step_cells(ROOT / "src" / "pipeline.py", 4)
    cells += steps + [nbf.v4.new_markdown_cell(f"## {n} · Results report\nAggregate results only (safe for GitHub)."),
                      nbf.v4.new_code_cell(SHOW)]
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}
    nbf.write(nb, ROOT / "notebooks" / "kaggle_run.ipynb")
    print("wrote notebooks/kaggle_run.ipynb")


if __name__ == "__main__":
    main()
