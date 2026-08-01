from __future__ import annotations

import copy
import json
import math
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_catboost import DATA_PATH, MODEL_DIR, prepare_frame


class EntityEmbeddingRegressor(nn.Module):
    def __init__(self, cardinalities: list[int], embedding_dims: list[int], numeric_count: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, dimension) for cardinality, dimension in zip(cardinalities, embedding_dims)]
        )
        input_size = sum(embedding_dims) + numeric_count
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(128, 1),
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        embedded = [embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings)]
        return self.network(torch.cat([numeric, *embedded], dim=1)).squeeze(1)


def encode_categories(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, int]], list[int]]:
    fit_encoded = np.zeros((len(X_fit), len(columns)), dtype=np.int64)
    valid_encoded = np.zeros((len(X_valid), len(columns)), dtype=np.int64)
    test_encoded = np.zeros((len(X_test), len(columns)), dtype=np.int64)
    vocabularies: dict[str, dict[str, int]] = {}
    cardinalities: list[int] = []
    for index, column in enumerate(columns):
        values = X_fit[column].fillna("unknown").astype(str)
        vocabulary = {value: value_index + 1 for value_index, value in enumerate(values.unique())}
        vocabularies[column] = vocabulary
        cardinalities.append(len(vocabulary) + 1)
        fit_encoded[:, index] = values.map(vocabulary).fillna(0).astype(int)
        valid_encoded[:, index] = X_valid[column].fillna("unknown").astype(str).map(vocabulary).fillna(0).astype(int)
        test_encoded[:, index] = X_test[column].fillna("unknown").astype(str).map(vocabulary).fillna(0).astype(int)
    return fit_encoded, valid_encoded, test_encoded, vocabularies, cardinalities


def prepare_numeric(
    X_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    fit = X_fit[numeric_columns].copy()
    valid = X_valid[numeric_columns].copy()
    test = X_test[numeric_columns].copy()
    medians = fit.median().fillna(0)
    fit = fit.fillna(medians)
    valid = valid.fillna(medians)
    test = test.fillna(medians)

    frequency_maps: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        counts = X_fit[column].fillna("unknown").astype(str).value_counts().to_dict()
        frequency_maps[column] = {str(key): int(value) for key, value in counts.items()}
        fit[f"{column}_frequency"] = np.log1p(X_fit[column].fillna("unknown").astype(str).map(counts).fillna(0))
        valid[f"{column}_frequency"] = np.log1p(X_valid[column].fillna("unknown").astype(str).map(counts).fillna(0))
        test[f"{column}_frequency"] = np.log1p(X_test[column].fillna("unknown").astype(str).map(counts).fillna(0))

    means = fit.mean()
    stds = fit.std().replace(0, 1).fillna(1)
    fit = (fit - means) / stds
    valid = (valid - means) / stds
    test = (test - means) / stds
    metadata = {
        "numeric_columns": fit.columns.tolist(),
        "medians": medians.to_dict(),
        "means": means.to_dict(),
        "stds": stds.to_dict(),
        "frequency_maps": frequency_maps,
    }
    return fit.to_numpy(np.float32), valid.to_numpy(np.float32), test.to_numpy(np.float32), metadata


def global_blend_prediction(bundle: dict, X: pd.DataFrame, area: pd.Series) -> np.ndarray:
    lightgbm_bundle = bundle["lightgbm_bundle"]
    lightgbm_X = X[lightgbm_bundle["features"]].copy()
    for column in lightgbm_bundle["categorical_features"]:
        lightgbm_X[column] = lightgbm_X[column].astype("category")
    catboost_X = X[lightgbm_bundle["features"]].copy()
    for column in lightgbm_bundle["categorical_features"]:
        catboost_X[column] = catboost_X[column].astype(str)
    lightgbm_log = lightgbm_bundle["model"].predict(lightgbm_X)
    catboost_log = bundle["catboost_model"].predict(catboost_X)
    log_prediction = bundle["lightgbm_weight"] * lightgbm_log + bundle["catboost_weight"] * catboost_log
    return np.expm1(log_prediction) * area.to_numpy()


def train_network(
    seed: int,
    fit_numeric: np.ndarray,
    fit_categorical: np.ndarray,
    fit_target: np.ndarray,
    valid_numeric: np.ndarray,
    valid_categorical: np.ndarray,
    valid_target: np.ndarray,
    cardinalities: list[int],
    embedding_dims: list[int],
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> tuple[EntityEmbeddingRegressor, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EntityEmbeddingRegressor(cardinalities, embedding_dims, fit_numeric.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, min_lr=1e-5)
    loss_function = nn.SmoothL1Loss(beta=0.35)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    normalized_target = ((fit_target - target_mean) / target_std).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(fit_numeric),
        torch.from_numpy(fit_categorical),
        torch.from_numpy(normalized_target),
    )
    loader = DataLoader(dataset, batch_size=1_024, shuffle=True, num_workers=2, pin_memory=True)
    valid_numeric_tensor = torch.from_numpy(valid_numeric).to(device)
    valid_categorical_tensor = torch.from_numpy(valid_categorical).to(device)
    valid_target_tensor = torch.from_numpy(((valid_target - target_mean) / target_std).astype(np.float32)).to(device)

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(1, 101):
        model.train()
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

        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            valid_prediction = model(valid_numeric_tensor, valid_categorical_tensor)
            valid_loss = torch.mean((valid_prediction - valid_target_tensor) ** 2).item()
        scheduler.step(valid_loss)
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch % 10 == 0:
            print(f"seed={seed} epoch={epoch} valid_mse={valid_loss:.6f}", flush=True)
        if stale_epochs >= 12:
            break
    if best_state is None:
        raise RuntimeError("Neural network did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, best_loss


def predict_network(
    model: EntityEmbeddingRegressor,
    numeric: np.ndarray,
    categorical: np.ndarray,
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions = []
    for start in range(0, len(numeric), 4_096):
        numeric_tensor = torch.from_numpy(numeric[start : start + 4_096]).to(device)
        categorical_tensor = torch.from_numpy(categorical[start : start + 4_096]).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            prediction = model(numeric_tensor, categorical_tensor).float().cpu().numpy()
        predictions.append(prediction * target_std + target_mean)
    return np.concatenate(predictions)


def main() -> None:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}", flush=True)
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

    fit_categories, valid_categories, test_categories, vocabularies, cardinalities = encode_categories(
        X_fit, X_valid, X_test, categorical_features
    )
    fit_numeric, valid_numeric, test_numeric, numeric_metadata = prepare_numeric(
        X_fit, X_valid, X_test, numeric_features, categorical_features
    )
    embedding_dims = [min(48, max(4, int(round(cardinality ** 0.25 * 4)))) for cardinality in cardinalities]
    target_mean = float(target_fit.mean())
    target_std = float(target_fit.std())

    models = []
    validation_logs = []
    test_logs = []
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
        validation_losses.append(validation_loss)
        validation_logs.append(predict_network(model, valid_numeric, valid_categories, target_mean, target_std, device))
        test_logs.append(predict_network(model, test_numeric, test_categories, target_mean, target_std, device))
        models.append({key: value.detach().cpu() for key, value in model.state_dict().items()})

    neural_valid = np.expm1(np.column_stack(validation_logs).mean(axis=1)) * area_valid.to_numpy()
    neural_test = np.expm1(np.column_stack(test_logs).mean(axis=1)) * area_test.to_numpy()
    tree_bundle = joblib.load(MODEL_DIR / "house_price_blend_compressed.pkl")
    tree_valid = global_blend_prediction(tree_bundle, X_valid, area_valid)
    tree_test = global_blend_prediction(tree_bundle, X_test, area_test)
    weights = np.linspace(0, 1, 101)
    tree_weight = min(
        weights,
        key=lambda weight: mean_squared_error(price_valid, weight * tree_valid + (1 - weight) * neural_valid),
    )
    prediction = np.clip(tree_weight * tree_test + (1 - tree_weight) * neural_test, 0, None)
    r2 = float(r2_score(price_test, prediction))
    metrics = {
        "model": "TreeAndEntityEmbeddingNeuralBlend",
        "device": str(device),
        "network_count": len(models),
        "tree_weight": float(tree_weight),
        "neural_weight": float(1 - tree_weight),
        "validation_losses": validation_losses,
        "r2": r2,
        "accuracy_percent_r2": r2 * 100,
        "mae": float(mean_absolute_error(price_test, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(price_test, prediction))),
        "training_seconds": float(time.perf_counter() - started),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dicts": models,
            "cardinalities": cardinalities,
            "embedding_dims": embedding_dims,
            "vocabularies": vocabularies,
            "numeric_metadata": numeric_metadata,
            "categorical_features": categorical_features,
            "target_mean": target_mean,
            "target_std": target_std,
            "tree_bundle": tree_bundle,
            "tree_weight": float(tree_weight),
            "neural_weight": float(1 - tree_weight),
            "prediction_target": "log1p(price_per_sqft)",
        },
        MODEL_DIR / "house_price_neural_blend.pt",
    )
    (MODEL_DIR / "model_metrics_neural_blend.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("FINAL_METRICS")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
