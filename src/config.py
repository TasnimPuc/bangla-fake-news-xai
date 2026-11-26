from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # --- data: both datasets are used together (deduplicated) ---
    kaggle_datasets: list = field(default_factory=lambda: ["cryptexcode/banfakenews",          # BanFakeNews (LREC 2020)
                                                           "hrithikmajumdar/bangla-fake-news"])  # BanFakeNews-2.0
    independent_test_url: str = "https://raw.githubusercontent.com/Shibu4064/IndoNLP/main/New_Test_Dataset.csv"
    local_dirs: list = field(default_factory=lambda: ["/kaggle/input"])  # also scanned (datasets added via "Add Input")
    use_github_v2: bool = True            # Kaggle copy of BanFakeNews-2.0 incomplete? -> authors' GitHub .7z release
    v2_min_rows: int = 20000
    v2_github_base: str = "https://raw.githubusercontent.com/Shibu4064/IndoNLP/main/Datasets/"
    v2_github_files: tuple = ("train_cleaned.7z", "val_cleaned.7z", "test_cleaned.7z")
    max_chars: int = 3000                 # headline + first part of the article
    split: tuple = (0.8, 0.1, 0.1)
    seed: int = 42

    # --- model ---
    model_name: str = "csebuetnlp/banglabert"
    max_len: int = 384
    epochs: int = 3
    lr: float = 3e-5
    batch_size: int = 16
    grad_accum: int = 1
    loss: str = "focal"                   # focal | weighted_ce | ce   (imbalance handling)
    focal_gamma: float = 2.0
    focal_alpha_fake: float = 0.75        # weight of the minority (fake) class in focal loss

    # --- baseline ---
    run_baseline: bool = True             # TF-IDF + logistic regression

    # --- explainability ---
    shap_n: int = 40                      # test articles explained with SHAP
    shap_words: int = 120                 # words per article fed to SHAP (cost grows with length)
    shap_max_evals: int = 300

    # --- outputs ---
    work_dir: str = "/kaggle/working/bfn"
    hub_model_repo: str = ""              # e.g. "your-hf-user/banglabert-fake-news"  (needs HF_TOKEN secret)
    hub_space_repo: str = ""              # e.g. "your-hf-user/bangla-fake-news-demo" (Streamlit, Docker Space)
    hf_token_secret: str = "HF_TOKEN"

    # --- checkpoints in your GCS bucket (same as the other projects) -> resume after a crash ---
    project_id: str = ""
    bucket: str = ""
    bucket_location: str = "US"
    gcs_prefix: str = "bangla-fake-news"
    run_name: str = "main"
    local_store: str = ""

    def to_dict(self):
        return asdict(self)
