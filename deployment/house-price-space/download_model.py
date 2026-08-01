from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


repo_id = os.environ["HF_MODEL_REPO"]
part_paths = sorted(
    path
    for path in HfApi().list_repo_files(repo_id, repo_type="model")
    if path.startswith("house_price.pkl.parts/")
)
if not part_paths:
    raise RuntimeError(f"No model parts found in {repo_id}.")

model_path = Path("/app/models/house_price.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)
with model_path.open("wb") as output_file:
    for part_path in part_paths:
        local_part = hf_hub_download(repo_id, part_path, repo_type="model")
        output_file.write(Path(local_part).read_bytes())
