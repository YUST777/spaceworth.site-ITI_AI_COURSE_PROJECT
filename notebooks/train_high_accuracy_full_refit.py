from __future__ import annotations

import json
import math
import time

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from train_catboost import DATA_PATH, MODEL_DIR, prepare_frame
from train_full_refit_ensemble import predict_network, train_fixed_network
from train_high_accuracy_ensemble import (
    CARPET_TO_SALEABLE_FACTOR,
    OUTLIER_LOWER_QUANTILE,
    OUTLIER_UPPER_QUANTILE,
    add_cross_fitted_target_encodings,
    add_property_features,
    category_frames,
    filter_noisy_price_per_sqft,
)
from train_neural_price_per_sqft_blend import encode_categories, prepare_numeric


LIGHTGBM_WEIGHT = 0.325
CATBOOST_WEIGHT = 0.525
NEURAL_WEIGHT = 0.15
LIGHTGBM_ITERATIONS = 1_530
CATBOOST_ITERATIONS = 351
NEURAL_EPOCHS = [24, 24, 24]


def full_target_encoding_maps(X: pd.DataFrame, target: pd.Series) -> dict[str, dict[str, float]]:
    mappings: dict[str, dict[str, float]] = {}
    global_mean = float(target.mean())
    for column in [
        "society",
        "locality_hint",
        "locality_tail_1",
        "locality_tail_2",
        "locality_tail_3",
        "location",
        "location_bedrooms",
        "society_bedrooms",
    ]:
        values = X[column].fillna("unknown").astype(str)
        stats = pd.DataFrame({"value": values.to_numpy(), "target": target.to_numpy()}).groupby(
            "value", observed=True
        )["target"].agg(["mean", "count"])
        smoothing = 12.0 if column in {"society", "locality_hint", "society_bedrooms"} else 4.0
        encoded = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        mappings[column] = {str(key): float(value) for key, value in encoded.items()}
    return mappings


def main() -> None:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    frame, numeric_features, categorical_features = prepare_frame(raw)
    frame = filter_noisy_price_per_sqft(frame)
    frame, numeric_features, categorical_features = add_property_features(
        frame, numeric_features, categorical_features
    )
    base_model_features = numeric_features + categorical_features
    X = frame[base_model_features].copy()
    price = frame["price"].astype(float)
    effective_area = frame["effective_area_sqft"].astype(float)
    target = np.log1p(price / effective_area)

    bins = pd.qcut(price.rank(method="first"), q=10, labels=False)
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
        price,
        effective_area,
        target,
        test_size=0.15,
        random_state=42,
        stratify=bins,
    )
    X_train, _, X_test, target_encoding_features = add_cross_fitted_target_encodings(
        X_train,
        X_test.iloc[:0].copy(),
        X_test,
        target_train.reset_index(drop=True),
    )
    target_train = target_train.reset_index(drop=True)
    numeric_features = numeric_features + target_encoding_features
    model_features = numeric_features + categorical_features
    print(f"Refitting on all {len(X_train):,} training rows; test rows: {len(X_test):,}", flush=True)

    lightgbm_train, _, lightgbm_test = category_frames(
        X_train, X_test.iloc[:0].copy(), X_test, categorical_features
    )
    lightgbm = LGBMRegressor(
        objective="regression_l2",
        n_estimators=LIGHTGBM_ITERATIONS,
        learning_rate=0.018,
        num_leaves=191,
        min_child_samples=9,
        max_bin=511,
        subsample=0.9,
        colsample_bytree=0.95,
        reg_alpha=0.02,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    lightgbm.fit(lightgbm_train[model_features], target_train, categorical_feature=categorical_features)

    catboost_train = X_train[model_features].copy()
    catboost_test = X_test[model_features].copy()
    for column in categorical_features:
        catboost_train[column] = catboost_train[column].fillna("unknown").astype(str)
        catboost_test[column] = catboost_test[column].fillna("unknown").astype(str)
    catboost = CatBoostRegressor(
        iterations=CATBOOST_ITERATIONS,
        depth=8,
        learning_rate=0.045,
        loss_function="RMSE",
        l2_leaf_reg=4.5,
        random_strength=0.18,
        bagging_temperature=0.35,
        random_seed=42,
        task_type="GPU",
        devices="0",
        border_count=128,
        allow_writing_files=False,
        verbose=100,
    )
    catboost.fit(
        catboost_train,
        target_train,
        cat_features=[model_features.index(column) for column in categorical_features],
    )

    train_categories, _, test_categories, vocabularies, cardinalities = encode_categories(
        X_train, X_test.iloc[:0].copy(), X_test, categorical_features
    )
    train_numeric, _, test_numeric, numeric_metadata = prepare_numeric(
        X_train, X_test.iloc[:0].copy(), X_test, numeric_features, categorical_features
    )
    embedding_dims = [min(56, max(4, int(round(cardinality ** 0.25 * 4)))) for cardinality in cardinalities]
    target_mean = float(target_train.mean())
    target_std = float(target_train.std())
    neural_state_dicts = []
    neural_predictions = []
    for seed, epochs in zip([42, 77, 2026], NEURAL_EPOCHS):
        model = train_fixed_network(
            seed,
            epochs,
            train_numeric,
            train_categories,
            target_train.to_numpy(),
            cardinalities,
            embedding_dims,
            target_mean,
            target_std,
            device,
        )
        neural_state_dicts.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        neural_predictions.append(
            predict_network(model, test_numeric, test_categories, target_mean, target_std, device)
        )

    lightgbm_price = np.expm1(lightgbm.predict(lightgbm_test[model_features])) * area_test.to_numpy()
    catboost_price = np.expm1(catboost.predict(catboost_test)) * area_test.to_numpy()
    neural_price = np.expm1(np.column_stack(neural_predictions).mean(axis=1)) * area_test.to_numpy()
    prediction = np.clip(
        LIGHTGBM_WEIGHT * lightgbm_price
        + CATBOOST_WEIGHT * catboost_price
        + NEURAL_WEIGHT * neural_price,
        0,
        None,
    )
    r2 = float(r2_score(price_test, prediction))
    metrics = {
        "model": "FullRefitHighAccuracyTreeAndNeuralEnsemble",
        "device": str(device),
        "rows_total": int(len(frame)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "carpet_to_saleable_factor": CARPET_TO_SALEABLE_FACTOR,
        "extra_price_per_sqft_filter_quantiles": [OUTLIER_LOWER_QUANTILE, OUTLIER_UPPER_QUANTILE],
        "lightgbm_iterations": LIGHTGBM_ITERATIONS,
        "catboost_iterations": CATBOOST_ITERATIONS,
        "network_epochs": NEURAL_EPOCHS,
        "blend_weights": {
            "lightgbm": LIGHTGBM_WEIGHT,
            "catboost": CATBOOST_WEIGHT,
            "neural": NEURAL_WEIGHT,
        },
        "r2": r2,
        "accuracy_percent_r2": r2 * 100.0,
        "mae": float(mean_absolute_error(price_test, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(price_test, prediction))),
        "training_seconds": float(time.perf_counter() - started),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "lightgbm_model": lightgbm,
            "catboost_model": catboost,
            "neural_state_dicts": neural_state_dicts,
            "model_features": model_features,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target_encoding_features": target_encoding_features,
            "target_encoding_maps": full_target_encoding_maps(X_train, target_train),
            "vocabularies": vocabularies,
            "cardinalities": cardinalities,
            "embedding_dims": embedding_dims,
            "numeric_metadata": numeric_metadata,
            "target_mean": target_mean,
            "target_std": target_std,
            "carpet_to_saleable_factor": CARPET_TO_SALEABLE_FACTOR,
            "blend_weights": {
                "lightgbm": LIGHTGBM_WEIGHT,
                "catboost": CATBOOST_WEIGHT,
                "neural": NEURAL_WEIGHT,
            },
            "prediction_target": "log1p(price_per_effective_area_sqft)",
        },
        MODEL_DIR / "house_price_high_accuracy_full_refit.pkl",
        compress=3,
    )
    (MODEL_DIR / "model_metrics_high_accuracy_full_refit.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("FINAL_METRICS")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
