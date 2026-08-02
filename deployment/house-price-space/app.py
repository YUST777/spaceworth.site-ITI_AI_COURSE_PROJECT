from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from torch import nn


MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/house_price.pkl"))
API_KEY = os.getenv("API_KEY", "")
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]
RATE_LIMIT_REQUESTS = max(1, int(os.getenv("RATE_LIMIT_REQUESTS", "30")))
RATE_LIMIT_WINDOW_SECONDS = max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
request_windows: defaultdict[str, deque[float]] = defaultdict(deque)
rate_limit_lock = Lock()


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


class PropertyInput(BaseModel):
    area_sqft: float = Field(gt=99, le=25_000)
    area_type: Literal["carpet", "super"] = "super"
    location: str = Field(min_length=1, max_length=120)
    locality: str | None = Field(default=None, max_length=180)
    society: str | None = Field(default=None, max_length=180)
    bedrooms: float | None = Field(default=None, ge=0, le=20)
    bathroom: float | None = Field(default=None, ge=0, le=20)
    balcony: float | None = Field(default=None, ge=0, le=20)
    car_parking: float | None = Field(default=None, ge=0, le=20)
    floor_num: float | None = Field(default=None, ge=-2, le=250)
    total_floors: float | None = Field(default=None, ge=1, le=250)
    property_type: Literal["flat", "villa", "house", "builder_floor", "penthouse", "studio", "plot", "unknown"] = "flat"
    furnishing: Literal["unfurnished", "semi_furnished", "furnished", "unknown"] = "unknown"
    transaction: Literal["resale", "new_property", "other", "unknown"] = "unknown"
    ownership: Literal["freehold", "cooperative_society", "leasehold", "unknown"] = "unknown"
    facing: str | None = Field(default=None, max_length=60)
    overlooking: str | None = Field(default=None, max_length=100)

    @field_validator("location", "locality", "society", "facing", "overlooking", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


def normalise_category(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    text = " ".join(value.lower().replace("-", " ").replace("/", " ").split())
    return text.replace(" ", "_") or "unknown"


def locality_tail(locality: str, token_count: int) -> str:
    if not locality or locality == "unknown":
        return "unknown"
    return "_".join(locality.split("_")[-token_count:])


def verify_api_key(api_key: str | None = Security(api_key_header)) -> None:
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


def enforce_rate_limit(request: Request) -> None:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    client_key = forwarded_for.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with rate_limit_lock:
        timestamps = request_windows[client_key]
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0])))
            raise HTTPException(
                status_code=429,
                detail="Prediction rate limit reached. Try again shortly.",
                headers={"Retry-After": str(retry_after)},
            )
        timestamps.append(now)


class HousePriceService:
    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise RuntimeError(f"Model file not found: {model_path}")
        self.bundle = joblib.load(model_path)
        self.categorical_features: list[str] = self.bundle["categorical_features"]
        self.numeric_features: list[str] = self.bundle["numeric_features"]
        self.model_features: list[str] = self.bundle["model_features"]
        self.neural_models = []
        numeric_count = len(self.bundle["numeric_metadata"]["numeric_columns"])
        for state_dict in self.bundle["neural_state_dicts"]:
            model = EntityEmbeddingRegressor(
                self.bundle["cardinalities"], self.bundle["embedding_dims"], numeric_count
            )
            model.load_state_dict(state_dict)
            model.eval()
            self.neural_models.append(model)

    def feature_frame(self, request: PropertyInput) -> tuple[pd.DataFrame, float]:
        area = float(request.area_sqft)
        bedrooms = request.bedrooms
        bathroom = request.bathroom
        floor_ratio = (
            float(request.floor_num) / float(request.total_floors)
            if request.floor_num is not None and request.total_floors is not None and request.total_floors > 0
            else np.nan
        )
        locality = normalise_category(request.locality)
        society = normalise_category(request.society)
        location = normalise_category(request.location)
        row: dict[str, object] = {
            "area_sqft": area,
            "bedrooms": bedrooms,
            "bathroom": bathroom,
            "balcony": request.balcony,
            "car_parking": request.car_parking,
            "floor_num": request.floor_num,
            "total_floors": request.total_floors,
            "floor_ratio": floor_ratio,
            "log_area": np.log1p(area),
            "area_per_bedroom": area / bedrooms if bedrooms and bedrooms > 0 else np.nan,
            "bathroom_per_bedroom": bathroom / bedrooms if bedrooms and bathroom is not None and bedrooms > 0 else np.nan,
            "is_ground_floor": float(request.floor_num == 0) if request.floor_num is not None else 0.0,
            "is_top_floor": float(
                request.floor_num is not None
                and request.total_floors is not None
                and request.floor_num == request.total_floors
            ),
            "effective_area_sqft": area * (self.bundle["carpet_to_saleable_factor"] if request.area_type == "carpet" else 1.0),
            "location": location,
            "area_type": request.area_type,
            "property_type": request.property_type,
            "furnishing": request.furnishing,
            "transaction": request.transaction,
            "ownership": request.ownership,
            "facing": normalise_category(request.facing),
            "overlooking": normalise_category(request.overlooking),
            "society": society,
            "locality_hint": locality,
            "locality_tail_1": locality_tail(locality, 1),
            "locality_tail_2": locality_tail(locality, 2),
            "locality_tail_3": locality_tail(locality, 3),
        }
        row["log_effective_area"] = np.log1p(float(row["effective_area_sqft"]))
        row["location_bedrooms"] = f"{location}__{bedrooms if bedrooms is not None else -1.0}"
        row["society_bedrooms"] = f"{society}__{bedrooms if bedrooms is not None else -1.0}"
        row["locality_bedrooms"] = f"{locality}__{bedrooms if bedrooms is not None else -1.0}"
        for feature_name in self.bundle["target_encoding_features"]:
            source_column = feature_name.removesuffix("_target_encoding")
            mapping = self.bundle["target_encoding_maps"][source_column]
            row[feature_name] = mapping.get(str(row[source_column]), self.bundle["target_mean"])
        frame = pd.DataFrame([row])
        for column in self.numeric_features:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame, float(row["effective_area_sqft"])

    def predict(self, request: PropertyInput) -> float:
        frame, effective_area = self.feature_frame(request)
        lightgbm_frame = frame[self.model_features].copy()
        pandas_categories = self.bundle["lightgbm_model"].booster_.pandas_categorical
        for column, categories in zip(self.categorical_features, pandas_categories):
            lightgbm_frame[column] = pd.Categorical(lightgbm_frame[column].astype(str), categories=categories)
        lightgbm_log = self.bundle["lightgbm_model"].booster_.predict(lightgbm_frame)[0]

        catboost_frame = frame[self.model_features].copy()
        for column in self.categorical_features:
            catboost_frame[column] = catboost_frame[column].fillna("unknown").astype(str)
        catboost_log = self.bundle["catboost_model"].predict(catboost_frame)[0]

        metadata = self.bundle["numeric_metadata"]
        numeric = frame[self.numeric_features].copy()
        numeric = numeric.fillna(pd.Series(metadata["medians"]))
        for column in self.categorical_features:
            frequency_map = metadata["frequency_maps"][column]
            numeric[f"{column}_frequency"] = np.log1p(
                frame[column].fillna("unknown").astype(str).map(frequency_map).fillna(0)
            )
        numeric = numeric[metadata["numeric_columns"]]
        numeric = (numeric - pd.Series(metadata["means"])) / pd.Series(metadata["stds"])
        categorical = np.zeros((1, len(self.categorical_features)), dtype=np.int64)
        for index, column in enumerate(self.categorical_features):
            vocabulary = self.bundle["vocabularies"][column]
            categorical[0, index] = vocabulary.get(str(frame.iloc[0][column]), 0)
        numeric_tensor = torch.from_numpy(numeric.to_numpy(np.float32))
        categorical_tensor = torch.from_numpy(categorical)
        with torch.no_grad():
            neural_logs = [
                float(model(numeric_tensor, categorical_tensor).item())
                * self.bundle["target_std"]
                + self.bundle["target_mean"]
                for model in self.neural_models
            ]
        weights = self.bundle["blend_weights"]
        prediction = (
            weights["lightgbm"] * np.expm1(float(lightgbm_log)) * effective_area
            + weights["catboost"] * np.expm1(float(catboost_log)) * effective_area
            + weights["neural"] * np.expm1(float(np.mean(neural_logs))) * effective_area
        )
        return max(0.0, float(prediction))


service = HousePriceService(MODEL_PATH)
app = FastAPI(title="ITI House Price API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": "FullRefitHighAccuracyTreeAndNeuralEnsemble",
        "held_out_r2": 0.906449314077493,
    }


@app.post("/predict", dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])
def predict(request: PropertyInput) -> dict[str, float]:
    return {"predicted_price_inr": round(service.predict(request), 2)}
