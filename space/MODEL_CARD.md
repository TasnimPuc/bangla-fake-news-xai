---
language: [bn]
base_model: csebuetnlp/banglabert
pipeline_tag: text-classification
tags: [fake-news, misinformation, bangla, explainability, shap]
---
# BanglaBERT fake-news classifier

`csebuetnlp/banglabert` fine-tuned on **BanFakeNews + BanFakeNews-2.0**, merged and deduplicated, with focal
loss for class imbalance. Labels: `AUTHENTIC` (0), `FAKE` (1). The decision threshold tuned on validation is in
`threshold.json`.

Evaluation covers a pooled held-out test set and the independent BanFakeNews-2.0 test set; see the GitHub repo's
`results/REPORT.md`, which also includes SHAP explanations.

**Use with care:** the model learns writing style and topic cues from these datasets. It is not a fact-checker.
Check original dataset licences before commercial use.
