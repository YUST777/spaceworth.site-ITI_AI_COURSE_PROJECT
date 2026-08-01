from __future__ import annotations

import os
import argparse
from pathlib import Path

from huggingface_hub import HfApi, get_token


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "house_price.pkl"
METRICS_PATH = ROOT / "models" / "model_metrics.json"
MODEL_CARD_PATH = ROOT / "deployment" / "house-price-model" / "README.md"
CHUNK_SIZE_BYTES = 2 * 1024 * 1024


def upload(api: HfApi, repo_id: str, local_path: Path, remote_path: str) -> None:
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        commit_message=f"Upload {remote_path}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the ITI house-price model to Hugging Face.")
    parser.add_argument("--part", type=int, help="Upload one 2 MiB model part (zero-based).")
    arguments = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or get_token()
    repo_id = os.environ.get("HF_MODEL_ID")
    if not token:
        raise RuntimeError("Log in with `hf auth login` or set HF_TOKEN before publishing.")
    if not repo_id:
        raise RuntimeError("Set HF_MODEL_ID, for example: duck233/iti-house-price-model")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    if arguments.part is None:
        upload(api, repo_id, MODEL_CARD_PATH, "README.md")
        upload(api, repo_id, METRICS_PATH, "model_metrics.json")
        print(f"Published metadata to https://huggingface.co/{repo_id}")
        return

    total_parts = (MODEL_PATH.stat().st_size + CHUNK_SIZE_BYTES - 1) // CHUNK_SIZE_BYTES
    if not 0 <= arguments.part < total_parts:
        raise ValueError(f"Part must be from 0 to {total_parts - 1}.")

    with MODEL_PATH.open("rb") as model_file:
        model_file.seek(arguments.part * CHUNK_SIZE_BYTES)
        chunk = model_file.read(CHUNK_SIZE_BYTES)

    remote_path = f"house_price.pkl.parts/part-{arguments.part:03d}"
    upload(api, repo_id, chunk, remote_path)
    print(f"Uploaded {remote_path} ({arguments.part + 1}/{total_parts})")


if __name__ == "__main__":
    main()
