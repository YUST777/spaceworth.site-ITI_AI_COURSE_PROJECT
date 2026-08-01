from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features, simple_category


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "notebooks" / "data" / "house_prices.csv"
MODEL_DIR = ROOT / "models"


def normalise_location_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    text = re.sub(r"[^a-z0-9 ]+", " ", value.strip().lower())
    return re.sub(r"\s+", " ", text).strip() or "unknown"


def locality_hint(title: object, society: object = None) -> str:
    if not isinstance(title, str) or not title.strip():
        return "unknown"
    text = normalise_location_text(title)
    value = text.split(" for sale ", 1)[-1]
    society_text = normalise_location_text(society)
    value = re.sub(r"^in\s+", "", value)
    if society_text != "unknown" and value.startswith(society_text):
        value = value[len(society_text) :].strip()
    return value.replace(" ", "_") or "unknown"


def locality_tail(locality: object, token_count: int) -> str:
    if not isinstance(locality, str) or not locality.strip() or locality == "unknown":
        return "unknown"
    tokens = locality.split("_")
    return "_".join(tokens[-token_count:])


def prepare_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = build_features(raw)
    frame["society"] = raw["Society"].apply(simple_category)
    frame["locality_hint"] = [
        locality_hint(title, society)
        for title, society in zip(raw["Title"], raw["Society"])
    ]
    frame["locality_tail_1"] = frame["locality_hint"].apply(lambda value: locality_tail(value, 1))
    frame["locality_tail_2"] = frame["locality_hint"].apply(lambda value: locality_tail(value, 2))
    frame["locality_tail_3"] = frame["locality_hint"].apply(lambda value: locality_tail(value, 3))

    frame = frame.dropna(subset=["price", "area_sqft"])
    frame = frame[
        frame["area_sqft"].between(100, 25_000)
        & frame["price"].between(100_000, 1_000_000_000)
    ].copy()

    price_per_sqft = frame["price"] / frame["area_sqft"]
    lower, upper = price_per_sqft.quantile([0.01, 0.99])
    frame = frame[price_per_sqft.between(lower, upper)].copy()

    frame["log_area"] = np.log1p(frame["area_sqft"])
    frame["area_per_bedroom"] = frame["area_sqft"] / frame["bedrooms"].replace(0, np.nan)
    frame["bathroom_per_bedroom"] = frame["bathroom"] / frame["bedrooms"].replace(0, np.nan)
    frame["is_ground_floor"] = (frame["floor_num"] == 0).astype(float)
    frame["is_top_floor"] = (
        frame["floor_num"].notna()
        & frame["total_floors"].notna()
        & (frame["floor_num"] == frame["total_floors"])
    ).astype(float)

    numeric_features = NUMERIC_FEATURES + [
        "log_area",
        "area_per_bedroom",
        "bathroom_per_bedroom",
        "is_ground_floor",
        "is_top_floor",
    ]
    categorical_features = CATEGORICAL_FEATURES + [
        "society",
        "locality_hint",
        "locality_tail_1",
        "locality_tail_2",
        "locality_tail_3",
    ]
    model_features = numeric_features + categorical_features

    for column in categorical_features:
        frame[column] = frame[column].fillna("unknown").astype(str)

    fingerprint = [
        "location",
        "society",
        "locality_hint",
        "locality_tail_1",
        "locality_tail_2",
        "locality_tail_3",
        "area_sqft",
        "bedrooms",
        "bathroom",
        "floor_num",
        "total_floors",
        "property_type",
    ]
    frame = (
        frame.groupby(fingerprint, dropna=False, observed=True)
        .agg(
            price=("price", "median"),
            **{
                column: (column, "first")
                for column in model_features
                if column not in fingerprint
            },
        )
        .reset_index()
    )

    return frame, numeric_features, categorical_features


def main() -> None:
    started = time.perf_counter()
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    frame, numeric_features, categorical_features = prepare_frame(raw)
    model_features = numeric_features + categorical_features

    X = frame[model_features]
    y = frame["price"].astype(float)
    bins = pd.qcut(y.rank(method="first"), q=10, labels=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=bins,
    )
    train_bins = pd.qcut(y_train.rank(method="first"), q=10, labels=False)
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.1764705882,
        random_state=42,
        stratify=train_bins,
    )

    categorical_indices = [model_features.index(column) for column in categorical_features]
    model = CatBoostRegressor(
        iterations=1_200,
        depth=8,
        learning_rate=0.06,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=6.0,
        random_strength=0.3,
        bagging_temperature=0.7,
        random_seed=42,
        task_type="GPU",
        devices="0",
        border_count=128,
        allow_writing_files=False,
        verbose=100,
    )

    print(
        f"Training rows: {len(X_fit):,}; validation rows: {len(X_valid):,}; "
        f"test rows: {len(X_test):,}; features: {len(model_features)}",
        flush=True,
    )
    model.fit(
        X_fit,
        y_fit,
        cat_features=categorical_indices,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=150,
        use_best_model=True,
    )

    prediction = np.clip(model.predict(X_test), 0, None)
    metrics = {
        "model": "CatBoostRegressor",
        "device": "GPU",
        "rows_total": int(len(frame)),
        "rows_train": int(len(X_fit)),
        "rows_validation": int(len(X_valid)),
        "rows_test": int(len(X_test)),
        "features": model_features,
        "best_iteration": int(model.get_best_iteration()),
        "r2": float(r2_score(y_test, prediction)),
        "accuracy_percent_r2": float(r2_score(y_test, prediction) * 100),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
        "training_seconds": float(time.perf_counter() - started),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_DIR / "house_price_catboost.cbm")
    joblib.dump(
        {
            "model": model,
            "features": model_features,
            "categorical_features": categorical_features,
        },
        MODEL_DIR / "house_price.pkl",
    )
    (MODEL_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "locations.json").write_text(
        json.dumps(sorted(frame["location"].unique().tolist()), indent=2),
        encoding="utf-8",
    )

    print("FINAL_METRICS")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
