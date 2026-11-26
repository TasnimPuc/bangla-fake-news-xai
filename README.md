# Bangla Fake News Detection + Explainability 📰🔍

A BanglaBERT classifier for Bangla fake news that also **shows its reasons**: SHAP highlights the words that pushed
an article towards "fake". The project reports more than accuracy:

- **Both datasets together, without leakage.** BanFakeNews (LREC 2020, ~48.7k authentic / ~1.3k fake) and
  BanFakeNews-2.0 (2025, ~47k authentic / ~13k fake) overlap heavily. Articles are matched across both by a
  content hash, so the same article never sits in train and test. Anything that appears in the BanFakeNews-2.0
  **independent test set** is removed from training, and that set is used as an external test.
- **Imbalance:** focal loss, with the decision threshold tuned on validation for fake-F1. `weighted_ce` and `ce`
  are switchable for comparison. Results report fake-F1, macro-F1 and AUPRC, never accuracy alone.
- **Does the model just learn which dataset an article came from?** Results are broken down by source
  (v1 / v2 / both) and by news category.
- **Explainability:**
  - SHAP Partition explanations for correct and wrong predictions, saved as an HTML page with red and blue word highlights;
  - a global chart of the words that push most strongly towards FAKE or AUTHENTIC.
- A **TF-IDF + logistic regression baseline** gives an honest comparison.

## Run on Kaggle (~1–1.5 h on one T4)
1. Import `notebooks/kaggle_run.ipynb`. Set GPU T4 and Internet On. Both datasets download automatically via
   `kagglehub`; you can also attach `cryptexcode/banfakenews` and `hrithikmajumdar/bangla-fake-news` as inputs.
2. Optional:
   - **Checkpoint backup:** add the GCS add-on and set `project_id`. Every epoch is copied to the bucket, so a dead session resumes.
   - **Publishing:** add the `HF_TOKEN` secret and set `HF_USER`.
3. Run All. Then download `results_for_github.zip` and unzip it here as `results/`.

## Streamlit demo
```bash
pip install -r requirements.txt
MODEL_ID=HF_USER/banglabert-fake-news streamlit run app/streamlit_app.py
```
Paste an article to get P(fake) and a SHAP word highlight. With `HF_USER` set, the notebook also deploys this
app as a Hugging Face Space (Docker, `space/`).

## Model files
- `banglabert_fake_news.pth`: a `state_dict` bundle with the base model name, `max_len`, the tuned threshold and labels.
- `model/hf/`: a Hugging Face folder plus `threshold.json`.

## Results
See **[results/REPORT.md](results/REPORT.md)** and **[results/shap_examples.html](results/shap_examples.html)**.

## Local / CI
```bash
python -m pytest -q             # synthetic Bangla data + tiny model: dedup/leakage, focal training, resume, SHAP
python tools/build_notebook.py  # regenerate the Kaggle notebook after editing the code
```

## Data & licences
- BanFakeNews: Hossain et al., *LREC 2020* (github.com/Rowan1224/FakeNews).
- BanFakeNews-2.0: Shibu et al., 2025, CC BY 4.0 (Mendeley DOI 10.17632/kjh887ct4j.1).
- Base model: `csebuetnlp/banglabert`.

Check each licence before any commercial use. This repo contains code and aggregate results only.

**Limitation:** the model learns style and topic cues from these datasets. It is not a fact-checker.
