from __future__ import annotations

import json
import time
from pathlib import Path

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
    y = frame["price"].astype(float)
    for column in categorical_features:
        X[column] = X[column].astype("category")

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

    model = LGBMRegressor(
        objective="regression_l2",
        n_estimators=3_000,
        learning_rate=0.025,
        num_leaves=127,
        min_child_samples=12,
        max_bin=255,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=2.0,
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
        y_fit,
        categorical_feature=categorical_features,
        eval_set=[(X_valid, y_valid)],
        callbacks=[early_stopping(150, verbose=True), log_evaluation(100)],
    )

    prediction = np.clip(model.predict(X_test), 0, None)
    r2 = float(r2_score(y_test, prediction))
    metrics = {
        "model": "LGBMRegressor",
        "device": "CPU",
        "rows_total": int(len(frame)),
        "rows_train": int(len(X_fit)),
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
    model.booster_.save_model(MODEL_DIR / "house_price_lightgbm.txt")
    joblib.dump(
        {
            "model": model,
            "features": model_features,
            "categorical_features": categorical_features,
        },
        MODEL_DIR / "house_price.pkl",
    )
    (MODEL_DIR / "model_metrics_lightgbm.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("FINAL_METRICS")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
