from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "house_price.pkl"
APP_PATH = ROOT / "deployment" / "house-price-space" / "app.py"

os.environ.setdefault("MODEL_PATH", str(MODEL_PATH))

spec = importlib.util.spec_from_file_location("house_price_api", APP_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load house price API from {APP_PATH}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

app = module.app
