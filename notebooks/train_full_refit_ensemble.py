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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_catboost import DATA_PATH, MODEL_DIR, prepare_frame
from train_neural_price_per_sqft_blend import (
    EntityEmbeddingRegressor,
    encode_categories,
    prepare_numeric,
)


TREE_LIGHTGBM_WEIGHT = 0.66
TREE_CATBOOST_WEIGHT = 0.34
FINAL_TREE_WEIGHT = 0.45
FINAL_NEURAL_WEIGHT = 0.55
NEURAL_EPOCHS = [24, 24, 28]


def train_fixed_network(
    seed: int,
    epochs: int,
    numeric: np.ndarray,
    categorical: np.ndarray,
    target: np.ndarray,
    cardinalities: list[int],
    embedding_dims: list[int],
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> EntityEmbeddingRegressor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EntityEmbeddingRegressor(cardinalities, embedding_dims, numeric.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
    loss_function = nn.SmoothL1Loss(beta=0.35)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    normalized_target = ((target - target_mean) / target_std).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(numeric),
            torch.from_numpy(categorical),
            torch.from_numpy(normalized_target),
        ),
        batch_size=1_024,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for numeric_batch, categorical_batch, target_batch in loader:
            numeric_batch = numeric_batch.to(device, non_blocking=True)
            categorical_batch = categorical_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(numeric_batch, categorical_batch)
                loss = loss_function(prediction, target_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
        scheduler.step()
        if epoch % 8 == 0 or epoch == epochs:
            print(f"seed={seed} epoch={epoch}/{epochs} train_loss={running_loss / len(loader):.6f}", flush=True)
    return model


def predict_network(
    model: EntityEmbeddingRegressor,
    numeric: np.ndarray,
    categorical: np.ndarray,
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(numeric), 4_096):
        numeric_tensor = torch.from_numpy(numeric[start : start + 4_096]).to(device)
        categorical_tensor = torch.from_numpy(categorical[start : start + 4_096]).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            prediction = model(numeric_tensor, categorical_tensor).float().cpu().numpy()
        outputs.append(prediction * target_std + target_mean)
    return np.concatenate(outputs)


def main() -> None:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = pd.read_csv(DATA_PATH, low_memory=False)
    frame, numeric_features, categorical_features = prepare_frame(raw)
    model_features = numeric_features + categorical_features
    X = frame[model_features].copy()
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
    ) = train_test_split(X, total_price, area, target, test_size=0.15, random_state=42, stratify=bins)

    print(f"Refitting on all {len(X_train):,} training rows; test rows: {len(X_test):,}", flush=True)

    lightgbm_train = X_train.copy()
    lightgbm_test = X_test.copy()
    for column in categorical_features:
        lightgbm_train[column] = lightgbm_train[column].astype("category")
        lightgbm_test[column] = lightgbm_test[column].astype("category")
    lightgbm = LGBMRegressor(
        objective="regression_l2",
        n_estimators=2_063,
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
    lightgbm.fit(lightgbm_train, target_train, categorical_feature=categorical_features)

    catboost_train = X_train.copy()
    catboost_test = X_test.copy()
    for column in categorical_features:
        catboost_train[column] = catboost_train[column].fillna("unknown").astype(str)
        catboost_test[column] = catboost_test[column].fillna("unknown").astype(str)
    cat_indices = [model_features.index(column) for column in categorical_features]
    catboost = CatBoostRegressor(
        iterations=2_494,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        l2_leaf_reg=4.0,
        random_strength=0.2,
        bagging_temperature=0.4,
        random_seed=42,
        task_type="GPU",
        devices="0",
        border_count=128,
        allow_writing_files=False,
        verbose=250,
    )
    catboost.fit(catboost_train, target_train, cat_features=cat_indices)

    lightgbm_test_log = lightgbm.predict(lightgbm_test)
    catboost_test_log = catboost.predict(catboost_test)
    tree_test_log = TREE_LIGHTGBM_WEIGHT * lightgbm_test_log + TREE_CATBOOST_WEIGHT * catboost_test_log
    tree_test_price = np.expm1(tree_test_log) * area_test.to_numpy()

    train_categories, unused_categories, test_categories, vocabularies, cardinalities = encode_categories(
        X_train, X_test.iloc[:0].copy(), X_test, categorical_features
    )
    train_numeric, unused_numeric, test_numeric, numeric_metadata = prepare_numeric(
        X_train, X_test.iloc[:0].copy(), X_test, numeric_features, categorical_features
    )
    embedding_dims = [min(48, max(4, int(round(cardinality ** 0.25 * 4)))) for cardinality in cardinalities]
    target_mean = float(target_train.mean())
    target_std = float(target_train.std())
    neural_states = []
    neural_test_logs = []
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
        neural_states.append({key: value.detach().cpu() for key, value in model.state_dict().items()})
        neural_test_logs.append(
            predict_network(model, test_numeric, test_categories, target_mean, target_std, device)
        )
    neural_test_log = np.column_stack(neural_test_logs).mean(axis=1)
    neural_test_price = np.expm1(neural_test_log) * area_test.to_numpy()

    prediction = np.clip(
        FINAL_TREE_WEIGHT * tree_test_price + FINAL_NEURAL_WEIGHT * neural_test_price,
        0,
        None,
    )
    r2 = float(r2_score(price_test, prediction))
    metrics = {
        "model": "FullRefitTreeAndNeuralEnsemble",
        "device": str(device),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "lightgbm_weight_inside_tree": TREE_LIGHTGBM_WEIGHT,
        "catboost_weight_inside_tree": TREE_CATBOOST_WEIGHT,
        "tree_weight": FINAL_TREE_WEIGHT,
        "neural_weight": FINAL_NEURAL_WEIGHT,
        "network_count": len(neural_states),
        "network_epochs": NEURAL_EPOCHS,
        "r2": r2,
        "accuracy_percent_r2": r2 * 100,
        "mae": float(mean_absolute_error(price_test, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(price_test, prediction))),
        "training_seconds": float(time.perf_counter() - started),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lightgbm_model": lightgbm,
            "catboost_model": catboost,
            "neural_state_dicts": neural_states,
            "cardinalities": cardinalities,
            "embedding_dims": embedding_dims,
            "vocabularies": vocabularies,
            "numeric_metadata": numeric_metadata,
            "categorical_features": categorical_features,
            "model_features": model_features,
            "target_mean": target_mean,
            "target_std": target_std,
            "tree_lightgbm_weight": TREE_LIGHTGBM_WEIGHT,
            "tree_catboost_weight": TREE_CATBOOST_WEIGHT,
            "final_tree_weight": FINAL_TREE_WEIGHT,
            "final_neural_weight": FINAL_NEURAL_WEIGHT,
            "prediction_target": "log1p(price_per_sqft)",
        },
        MODEL_DIR / "house_price_full_refit_ensemble.pt",
    )
    (MODEL_DIR / "model_metrics_full_refit_ensemble.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("FINAL_METRICS")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
