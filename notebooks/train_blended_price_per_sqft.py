from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from train_catboost import DATA_PATH, MODEL_DIR, prepare_frame


def main() -> None:
    started = time.perf_counter()
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    frame, numeric_features, categorical_features = prepare_frame(raw)
    model_features = numeric_features + categorical_features

    X = frame[model_features].copy()
    for column in categorical_features:
        X[column] = X[column].astype(str)
    total_price = frame["price"].astype(float)
    area = frame["area_sqft"].astype(float)
    target = np.log1p(total_price / area)

    bins = pd.qcut(total_price.rank(method="first"), q=10, labels=False)
    (
        X_train,
        X_test,
        price_train,
        price_test,
        area_train,
        area_test,
        target_train,
        target_test,
    ) = train_test_split(
        X,
        total_price,
        area,
        target,
        test_size=0.15,
        random_state=42,
        stratify=bins,
    )
    train_bins = pd.qcut(price_train.rank(method="first"), q=10, labels=False)
    (
        X_fit,
        X_valid,
        price_fit,
        price_valid,
        area_fit,
        area_valid,
        target_fit,
        target_valid,
    ) = train_test_split(
        X_train,
        price_train,
        area_train,
        target_train,
        test_size=0.1764705882,
        random_state=42,
        stratify=train_bins,
    )

    cat_indices = [model_features.index(column) for column in categorical_features]
    catboost = CatBoostRegressor(
        iterations=2_500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=4.0,
        random_strength=0.2,
        bagging_temperature=0.4,
        random_seed=42,
        task_type="GPU",
        devices="0",
        border_count=128,
        allow_writing_files=False,
        verbose=200,
    )
    print(
        f"Training rows: {len(X_fit):,}; validation rows: {len(X_valid):,}; "
        f"test rows: {len(X_test):,}; features: {len(model_features)}",
        flush=True,
    )
    catboost.fit(
        X_fit,
        target_fit,
        cat_features=cat_indices,
        eval_set=(X_valid, target_valid),
        early_stopping_rounds=180,
        use_best_model=True,
    )

    # Load the already validated LightGBM model trained on the identical fit split.
    lgb_bundle = joblib.load(MODEL_DIR / "house_price_per_sqft.pkl")
    lgb_model = lgb_bundle["model"]
    lgb_features = lgb_bundle["features"]
    for dataset in [X_valid, X_test]:
        for column in lgb_bundle["categorical_features"]:
            dataset[column] = dataset[column].astype("category")

    lgb_valid = np.expm1(lgb_model.predict(X_valid[lgb_features])) * area_valid.to_numpy()
    lgb_test = np.expm1(lgb_model.predict(X_test[lgb_features])) * area_test.to_numpy()
    cat_valid = np.expm1(catboost.predict(X_valid)) * area_valid.to_numpy()
    cat_test = np.expm1(catboost.predict(X_test)) * area_test.to_numpy()

    weights = np.linspace(0, 1, 101)
    best_weight = min(
        weights,
        key=lambda weight: mean_squared_error(price_valid, weight * lgb_valid + (1 - weight) * cat_valid),
    )
    blended = np.clip(best_weight * lgb_test + (1 - best_weight) * cat_test, 0, None)
    r2 = float(r2_score(price_test, blended))
    metrics = {
        "model": "LightGBMAndCatBoostPricePerSqftBlend",
        "device": "LightGBM CPU + CatBoost GPU",
        "rows_total": int(len(frame)),
        "rows_train": int(len(X_train)),
        "rows_validation": int(len(X_valid)),
        "rows_test": int(len(X_test)),
        "lightgbm_weight": float(best_weight),
        "catboost_weight": float(1 - best_weight),
        "catboost_best_iteration": int(catboost.get_best_iteration()),
        "r2": r2,
        "accuracy_percent_r2": r2 * 100,
        "mae": float(mean_absolute_error(price_test, blended)),
        "rmse": float(mean_squared_error(price_test, blended) ** 0.5),
        "training_seconds": float(time.perf_counter() - started),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "lightgbm_bundle": lgb_bundle,
            "catboost_model": catboost,
            "lightgbm_weight": float(best_weight),
            "catboost_weight": float(1 - best_weight),
            "prediction_target": "log1p(price_per_sqft)",
        },
        MODEL_DIR / "house_price_blend.pkl",
    )
    catboost.save_model(MODEL_DIR / "house_price_price_per_sqft_catboost.cbm")
    (MODEL_DIR / "model_metrics_blended_price_per_sqft.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("FINAL_METRICS")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
