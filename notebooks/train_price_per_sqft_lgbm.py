from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
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
        X[column] = X[column].astype("category")

    total_price = frame["price"].astype(float)
    price_per_sqft = total_price / frame["area_sqft"].astype(float)
    target = np.log1p(price_per_sqft)

    bins = pd.qcut(total_price.rank(method="first"), q=10, labels=False)
    X_train, X_test, y_train, y_test, area_train, area_test, target_train, target_test = train_test_split(
        X,
        total_price,
        frame["area_sqft"].astype(float),
        target,
        test_size=0.15,
        random_state=42,
        stratify=bins,
    )
    train_bins = pd.qcut(y_train.rank(method="first"), q=10, labels=False)
    (
        X_fit,
        X_valid,
        y_fit,
        y_valid,
        area_fit,
        area_valid,
        target_fit,
        target_valid,
    ) = train_test_split(
        X_train,
        y_train,
        area_train,
        target_train,
        test_size=0.1764705882,
        random_state=42,
        stratify=train_bins,
    )

    model = LGBMRegressor(
        objective="regression_l2",
        n_estimators=3_000,
        learning_rate=0.02,
        num_leaves=127,
        min_child_samples=10,
        max_bin=255,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.02,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    print(
        f"Training rows: {len(X_fit):,}; validation rows: {len(X_valid):,}; "
        f"test rows: {len(X_test):,}; features: {len(model_features)}",
        flush=True,
    )
    model.fit(
        X_fit,
        target_fit,
        categorical_feature=categorical_features,
        eval_set=[(X_valid, target_valid)],
        callbacks=[early_stopping(180, verbose=True), log_evaluation(200)],
    )

    predicted_price_per_sqft = np.expm1(model.predict(X_test))
    prediction = np.clip(predicted_price_per_sqft * area_test.to_numpy(), 0, None)
    r2 = float(r2_score(y_test, prediction))
    metrics = {
        "model": "LGBMRegressorPricePerSqft",
        "device": "CPU",
        "target": "log1p(price / area_sqft)",
        "rows_total": int(len(frame)),
        "rows_train": int(len(X_train)),
        "rows_validation": int(len(X_valid)),
        "rows_test": int(len(X_test)),
        "features": model_features,
        "best_iteration": int(model.best_iteration_),
        "r2": r2,
        "accuracy_percent_r2": r2 * 100,
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
        "training_seconds": float(time.perf_counter() - started),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(MODEL_DIR / "house_price_per_sqft_lightgbm.txt")
    joblib.dump(
        {
            "model": model,
            "features": model_features,
            "categorical_features": categorical_features,
            "prediction_target": "log1p(price_per_sqft)",
        },
        MODEL_DIR / "house_price_per_sqft.pkl",
    )
    (MODEL_DIR / "model_metrics_price_per_sqft_lgbm.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("FINAL_METRICS")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
