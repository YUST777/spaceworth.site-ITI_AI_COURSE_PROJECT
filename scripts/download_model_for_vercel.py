from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path


MODEL_URL = "https://huggingface.co/duck233/iti-house-price-model/resolve/main/house_price.pkl?download=true"
EXPECTED_SHA256 = "872f0852fe6ff615c14a7cb5d4b243ddc7032613cc94a991d2fe1a01686e6ca4"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "house_price.pkl"


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_suffix(".pkl.download")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "iti-house-price-vercel-build"})
    with urllib.request.urlopen(request, timeout=300) as response, temporary_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)

    checksum = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
    if checksum != EXPECTED_SHA256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded model checksum mismatch: {checksum}")

    temporary_path.replace(MODEL_PATH)
    print(f"Downloaded verified model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
