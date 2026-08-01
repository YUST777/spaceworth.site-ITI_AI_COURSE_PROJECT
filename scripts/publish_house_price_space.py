from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, get_token


ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "deployment" / "house-price-space"


def main() -> None:
    token = os.environ.get("HF_TOKEN") or get_token()
    repo_id = os.environ.get("HF_SPACE_ID")
    if not token:
        raise RuntimeError("Set HF_TOKEN to a Hugging Face write token before publishing.")
    if not repo_id:
        raise RuntimeError("Set HF_SPACE_ID, for example: YUST777/iti-house-price-api")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=SPACE_DIR,
        commit_message="Deploy house price API container",
    )
    print(f"Published https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
