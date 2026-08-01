from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "deployment" / "house-price-space"
MODEL_PATH = ROOT / "models" / "house_price.pkl"


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_SPACE_ID")
    if not token:
        raise RuntimeError("Set HF_TOKEN to a Hugging Face write token before publishing.")
    if not repo_id:
        raise RuntimeError("Set HF_SPACE_ID, for example: YUST777/iti-house-price-api")
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=SPACE_DIR,
        commit_message="Deploy house price API container",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="space",
        path_or_fileobj=MODEL_PATH,
        path_in_repo="models/house_price.pkl",
        commit_message="Upload trained house price model",
    )
    print(f"Published https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
