from __future__ import annotations

import json
import math
import time

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from train_catboost import DATA_PATH, MODEL_DIR, prepare_frame
from train_neural_price_per_sqft_blend import (
    encode_categories,
    predict_network,
    prepare_numeric,
    train_network,
)


CARPET_TO_SALEABLE_FACTOR = 1.2
OUTLIER_LOWER_QUANTILE = 0.02
OUTLIER_UPPER_QUANTILE = 0.98
TARGET_ENCODE_COLUMNS = [
    "society",
    "locality_hint",
    "locality_tail_1",
    "locality_tail_2",
    "locality_tail_3",
    "location",
    "location_bedrooms",
    "society_bedrooms",
]


def add_property_features(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = frame.copy()
    carpet_multiplier = np.where(frame["area_type"].eq("carpet"), CARPET_TO_SALEABLE_FACTOR, 1.0)
    frame["effective_area_sqft"] = frame["area_sqft"] * carpet_multiplier
    frame["log_effective_area"] = np.log1p(frame["effective_area_sqft"])
    frame["location_bedrooms"] = (
        frame["location"].astype(str) + "__" + frame["bedrooms"].fillna(-1).astype(str)
    )
    frame["society_bedrooms"] = (
        frame["society"].astype(str) + "__" + frame["bedrooms"].fillna(-1).astype(str)
    )
    frame["locality_bedrooms"] = (
        frame["locality_hint"].astype(str) + "__" + frame["bedrooms"].fillna(-1).astype(str)
    )
    return (
        frame,
        numeric_features + ["effective_area_sqft", "log_effective_area"],
        categorical_features + ["location_bedrooms", "society_bedrooms", "locality_bedrooms"],
    )


def filter_noisy_price_per_sqft(frame: pd.DataFrame) -> pd.DataFrame:
    effective_area = frame["area_sqft"] * np.where(frame["area_type"].eq("carpet"), CARPET_TO_SALEABLE_FACTOR, 1.0)
    price_per_sqft = frame["price"] / effective_area
    lower, upper = price_per_sqft.quantile([OUTLIER_LOWER_QUANTILE, OUTLIER_UPPER_QUANTILE])
    filtered = frame.loc[price_per_sqft.between(lower, upper)].copy()
    print(
        f"quality filter kept {len(filtered):,}/{len(frame):,} rows "
        f"between Rs {lower:,.0f} and Rs {upper:,.0f} per effective sqft",
        flush=True,
    )
    return filtered


def add_cross_fitted_target_encodings(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    target_fit: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    fit = X_fit.copy()
    valid = X_valid.copy()
    test = X_test.copy()
    global_mean = float(target_fit.mean())
    added: list[str] = []
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    for column in TARGET_ENCODE_COLUMNS:
        feature_name = f"{column}_target_encoding"
        fit[feature_name] = global_mean
        values = X_fit[column].fillna("unknown").astype(str)
        for train_indices, holdout_indices in folds.split(X_fit):
            fold_values = values.iloc[train_indices]
            fold_target = target_fit.iloc[train_indices]
            stats = pd.DataFrame({"value": fold_values.to_numpy(), "target": fold_target.to_numpy()}).groupby(
                "value", observed=True
            )["target"].agg(["mean", "count"])
            smoothing = 12.0 if column in {"society", "locality_hint", "society_bedrooms"} else 4.0
            encoded = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            fit.iloc[holdout_indices, fit.columns.get_loc(feature_name)] = values.iloc[holdout_indices].map(encoded).fillna(
                global_mean
            )
        full_stats = pd.DataFrame({"value": values.to_numpy(), "target": target_fit.to_numpy()}).groupby(
            "value", observed=True
        )["target"].agg(["mean", "count"])
        smoothing = 12.0 if column in {"society", "locality_hint", "society_bedrooms"} else 4.0
        encoded = (full_stats["mean"] * full_stats["count"] + global_mean * smoothing) / (
            full_stats["count"] + smoothing
        )
        valid[feature_name] = valid[column].fillna("unknown").astype(str).map(encoded).fillna(global_mean)
        test[feature_name] = test[column].fillna("unknown").astype(str).map(encoded).fillna(global_mean)
        added.append(feature_name)
    return fit, valid, test, added


def category_frames(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit = X_fit.copy()
    valid = X_valid.copy()
    test = X_test.copy()
    for column in columns:
        categories = pd.Index(fit[column].fillna("unknown").astype(str).unique())
        dtype = pd.CategoricalDtype(categories=categories)
        fit[column] = fit[column].fillna("unknown").astype(str).astype(dtype)
        valid[column] = valid[column].fillna("unknown").astype(str).astype(dtype)
        test[column] = test[column].fillna("unknown").astype(str).astype(dtype)
    return fit, valid, test


def choose_blend_weights(validation_predictions: list[np.ndarray], validation_target: pd.Series) -> np.ndarray:
    candidates = np.linspace(0.0, 1.0, 41)
    best_weights = np.array([1.0, 0.0, 0.0])
    best_error = float("inf")
    for lightgbm_weight in candidates:
        for catboost_weight in candidates:
            neural_weight = 1.0 - lightgbm_weight - catboost_weight
            if neural_weight < 0.0:
                continue
            prediction = (
                lightgbm_weight * validation_predictions[0]
                + catboost_weight * validation_predictions[1]
                + neural_weight * validation_predictions[2]
            )
            error = mean_squared_error(validation_target, prediction)
            if error < best_error:
                best_error = error
                best_weights = np.array([lightgbm_weight, catboost_weight, neural_weight])
    return best_weights


def choose_final_blend_weights(
    base_prediction: np.ndarray,
    direct_lightgbm_prediction: np.ndarray,
    direct_catboost_prediction: np.ndarray,
    validation_target: pd.Series,
) -> np.ndarray:
    candidates = np.linspace(0.0, 1.0, 41)
    best_weights = np.array([1.0, 0.0, 0.0])
    best_error = float("inf")
    for base_weight in candidates:
        for direct_lightgbm_weight in candidates:
            direct_catboost_weight = 1.0 - base_weight - direct_lightgbm_weight
            if direct_catboost_weight < 0.0:
                continue
            prediction = (
                base_weight * base_prediction
                + direct_lightgbm_weight * direct_lightgbm_prediction
                + direct_catboost_weight * direct_catboost_prediction
            )
            error = mean_squared_error(validation_target, prediction)
            if error < best_error:
                best_error = error
                best_weights = np.array([base_weight, direct_lightgbm_weight, direct_catboost_weight])
    return best_weights


def main() -> None:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    frame, numeric_features, categorical_features = prepare_frame(raw)
    frame = filter_noisy_price_per_sqft(frame)
    frame, numeric_features, categorical_features = add_property_features(
        frame, numeric_features, categorical_features
    )
    model_features = numeric_features + categorical_features
    X = frame[model_features].copy()
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
    X_fit, X_valid, X_test, target_encoding_features = add_cross_fitted_target_encodings(
        X_fit, X_valid, X_test, target_fit.reset_index(drop=True)
    )
    target_fit = target_fit.reset_index(drop=True)
    numeric_features = numeric_features + target_encoding_features
    model_features = numeric_features + categorical_features

    lightgbm_fit, lightgbm_valid, lightgbm_test = category_frames(X_fit, X_valid, X_test, categorical_features)
    lightgbm = LGBMRegressor(
        objective="regression_l2",
        n_estimators=5_000,
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
    lightgbm.fit(
        lightgbm_fit[model_features],
        target_fit,
        categorical_feature=categorical_features,
        eval_set=[(lightgbm_valid[model_features], target_valid)],
        callbacks=[early_stopping(250, verbose=True), log_evaluation(250)],
    )

    direct_target_fit = np.log1p(price_fit)
    direct_target_valid = np.log1p(price_valid)
    direct_lightgbm = LGBMRegressor(
        objective="regression_l2",
        n_estimators=4_000,
        learning_rate=0.02,
        num_leaves=191,
        min_child_samples=10,
        max_bin=511,
        subsample=0.9,
        colsample_bytree=0.95,
        reg_alpha=0.02,
        reg_lambda=1.2,
        random_state=77,
        n_jobs=-1,
        verbosity=-1,
    )
    direct_lightgbm.fit(
        lightgbm_fit[model_features],
        direct_target_fit,
        categorical_feature=categorical_features,
        eval_set=[(lightgbm_valid[model_features], direct_target_valid)],
        callbacks=[early_stopping(250, verbose=True), log_evaluation(250)],
    )

    catboost_fit = X_fit[model_features].copy()
    catboost_valid = X_valid[model_features].copy()
    catboost_test = X_test[model_features].copy()
    for column in categorical_features:
        catboost_fit[column] = catboost_fit[column].fillna("unknown").astype(str)
        catboost_valid[column] = catboost_valid[column].fillna("unknown").astype(str)
        catboost_test[column] = catboost_test[column].fillna("unknown").astype(str)
    catboost = CatBoostRegressor(
        iterations=3_500,
        depth=8,
        learning_rate=0.045,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=4.5,
        random_strength=0.18,
        bagging_temperature=0.35,
        random_seed=42,
        task_type="GPU",
        devices="0",
        border_count=128,
        allow_writing_files=False,
        verbose=250,
    )
    catboost.fit(
        catboost_fit,
        target_fit,
        cat_features=[model_features.index(column) for column in categorical_features],
        eval_set=(catboost_valid, target_valid),
        early_stopping_rounds=250,
        use_best_model=True,
    )

    direct_catboost = CatBoostRegressor(
        iterations=2_000,
        depth=8,
        learning_rate=0.045,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=4.5,
        random_strength=0.18,
        bagging_temperature=0.35,
        random_seed=77,
        task_type="GPU",
        devices="0",
        border_count=128,
        allow_writing_files=False,
        verbose=250,
    )
    direct_catboost.fit(
        catboost_fit,
        target_fit,
        cat_features=[model_features.index(column) for column in categorical_features],
        eval_set=(catboost_valid, target_valid),
        early_stopping_rounds=250,
        use_best_model=True,
    )

    fit_categories, valid_categories, test_categories, vocabularies, cardinalities = encode_categories(
        X_fit, X_valid, X_test, categorical_features
    )
    fit_numeric, valid_numeric, test_numeric, numeric_metadata = prepare_numeric(
        X_fit, X_valid, X_test, numeric_features, categorical_features
    )
    embedding_dims = [min(56, max(4, int(round(cardinality ** 0.25 * 4)))) for cardinality in cardinalities]
    target_mean = float(target_fit.mean())
    target_std = float(target_fit.std())
    neural_valid_logs: list[np.ndarray] = []
    neural_test_logs: list[np.ndarray] = []
    neural_states = []
    validation_losses = []
    for seed in [42, 77, 2026]:
        model, validation_loss = train_network(
            seed,
            fit_numeric,
            fit_categories,
            target_fit.to_numpy(),
            valid_numeric,
            valid_categories,
            target_valid.to_numpy(),
            cardinalities,
            embedding_dims,
            target_mean,
            target_std,
            device,
        )
        neural_states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        validation_losses.append(float(validation_loss))
        neural_valid_logs.append(
            predict_network(model, valid_numeric, valid_categories, target_mean, target_std, device)
        )
        neural_test_logs.append(
            predict_network(model, test_numeric, test_categories, target_mean, target_std, device)
        )

    lightgbm_valid_price = np.expm1(lightgbm.predict(lightgbm_valid[model_features])) * area_valid.to_numpy()
    lightgbm_test_price = np.expm1(lightgbm.predict(lightgbm_test[model_features])) * area_test.to_numpy()
    catboost_valid_price = (
        np.expm1(catboost.predict(catboost_valid))
        + np.expm1(direct_catboost.predict(catboost_valid))
    ) * area_valid.to_numpy() / 2.0
    catboost_test_price = (
        np.expm1(catboost.predict(catboost_test))
        + np.expm1(direct_catboost.predict(catboost_test))
    ) * area_test.to_numpy() / 2.0
    direct_lightgbm_valid_price = np.expm1(direct_lightgbm.predict(lightgbm_valid[model_features]))
    direct_lightgbm_test_price = np.expm1(direct_lightgbm.predict(lightgbm_test[model_features]))
    direct_catboost_valid_price = np.expm1(direct_catboost.predict(catboost_valid)) * area_valid.to_numpy()
    direct_catboost_test_price = np.expm1(direct_catboost.predict(catboost_test)) * area_test.to_numpy()
    neural_valid_price = np.expm1(np.column_stack(neural_valid_logs).mean(axis=1)) * area_valid.to_numpy()
    neural_test_price = np.expm1(np.column_stack(neural_test_logs).mean(axis=1)) * area_test.to_numpy()
    blend_weights = choose_blend_weights(
        [lightgbm_valid_price, catboost_valid_price, neural_valid_price], price_valid
    )
    base_valid_prediction = (
        blend_weights[0] * lightgbm_test_price
        + blend_weights[1] * catboost_test_price
        + blend_weights[2] * neural_test_price,
    )
    base_test_prediction = (
        blend_weights[0] * lightgbm_test_price
        + blend_weights[1] * catboost_test_price
        + blend_weights[2] * neural_test_price
    )
    base_valid_prediction = (
        blend_weights[0] * lightgbm_valid_price
        + blend_weights[1] * catboost_valid_price
        + blend_weights[2] * neural_valid_price
    )
    final_blend_weights = np.array([1.0, 0.0, 0.0])
    prediction = np.clip(
        final_blend_weights[0] * base_test_prediction
        + final_blend_weights[1] * direct_lightgbm_test_price
        + final_blend_weights[2] * direct_catboost_test_price,
        0,
        None,
    )
    r2 = float(r2_score(price_test, prediction))
    metrics = {
        "model": "HighAccuracyTreeAndNeuralEnsemble",
        "device": str(device),
        "carpet_to_saleable_factor": CARPET_TO_SALEABLE_FACTOR,
        "extra_price_per_sqft_filter_quantiles": [OUTLIER_LOWER_QUANTILE, OUTLIER_UPPER_QUANTILE],
        "rows_total": int(len(frame)),
        "rows_train": int(len(X_train)),
        "rows_validation": int(len(X_valid)),
        "rows_test": int(len(X_test)),
        "lightgbm_best_iteration": int(lightgbm.best_iteration_),
        "catboost_best_iteration": int(catboost.get_best_iteration()),
        "direct_lightgbm_best_iteration": int(direct_lightgbm.best_iteration_),
        "direct_catboost_best_iteration": int(direct_catboost.get_best_iteration()),
        "network_count": len(neural_states),
        "validation_losses": validation_losses,
        "blend_weights": {
            "lightgbm": float(blend_weights[0]),
            "catboost": float(blend_weights[1]),
            "neural": float(blend_weights[2]),
        },
        "final_blend_weights": {
            "price_per_sqft_ensemble": float(final_blend_weights[0]),
            "direct_price_lightgbm": float(final_blend_weights[1]),
            "direct_price_catboost": float(final_blend_weights[2]),
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
            "direct_lightgbm_model": direct_lightgbm,
            "direct_catboost_model": direct_catboost,
            "neural_state_dicts": neural_states,
            "model_features": model_features,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target_encoding_features": target_encoding_features,
            "vocabularies": vocabularies,
            "cardinalities": cardinalities,
            "embedding_dims": embedding_dims,
            "numeric_metadata": numeric_metadata,
            "target_mean": target_mean,
            "target_std": target_std,
            "blend_weights": blend_weights.tolist(),
            "final_blend_weights": final_blend_weights.tolist(),
            "carpet_to_saleable_factor": CARPET_TO_SALEABLE_FACTOR,
            "prediction_target": "log1p(price_per_effective_area_sqft)",
        },
        MODEL_DIR / "house_price_high_accuracy_ensemble.pkl",
        compress=3,
    )
    (MODEL_DIR / "model_metrics_high_accuracy_ensemble.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("FINAL_METRICS")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
