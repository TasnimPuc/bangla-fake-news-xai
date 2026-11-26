from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def push_all(cfg, out: Path, token: str) -> dict:
    from huggingface_hub import HfApi
    api, res = HfApi(token=token), {}
    if cfg.hub_model_repo:
        api.create_repo(cfg.hub_model_repo, exist_ok=True)
        (out / "model" / "hf" / "README.md").write_text((REPO / "space" / "MODEL_CARD.md").read_text(encoding="utf-8"),
                                                        encoding="utf-8")
        api.upload_folder(folder_path=str(out / "model" / "hf"), repo_id=cfg.hub_model_repo)
        api.upload_file(path_or_fileobj=str(out / "model" / "banglabert_fake_news.pth"),
                        path_in_repo="banglabert_fake_news.pth", repo_id=cfg.hub_model_repo)
        res["model"] = f"https://huggingface.co/{cfg.hub_model_repo}"
    if cfg.hub_space_repo:
        api.create_repo(cfg.hub_space_repo, repo_type="space", space_sdk="docker", exist_ok=True)
        for f in ("Dockerfile", "README.md", "requirements.txt"):
            api.upload_file(path_or_fileobj=str(REPO / "space" / f), path_in_repo=f, repo_id=cfg.hub_space_repo,
                            repo_type="space")
        api.upload_file(path_or_fileobj=str(REPO / "app" / "streamlit_app.py"), path_in_repo="streamlit_app.py",
                        repo_id=cfg.hub_space_repo, repo_type="space")
        if cfg.hub_model_repo:
            api.add_space_variable(cfg.hub_space_repo, "MODEL_ID", cfg.hub_model_repo)
        res["space"] = f"https://huggingface.co/spaces/{cfg.hub_space_repo}"
    return res
