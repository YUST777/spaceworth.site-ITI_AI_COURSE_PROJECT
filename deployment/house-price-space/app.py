from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from collections import defaultdict, deque
import secrets
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import uuid4

import joblib
import httpx
import numpy as np
import pandas as pd
import psycopg
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field, field_validator
from torch import nn


MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/house_price.pkl"))
API_KEY = os.getenv("API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]
RATE_LIMIT_REQUESTS = max(1, int(os.getenv("RATE_LIMIT_REQUESTS", "30")))
RATE_LIMIT_WINDOW_SECONDS = max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
PLAN_RATE_LIMIT_REQUESTS = max(1, int(os.getenv("PLAN_RATE_LIMIT_REQUESTS", "6")))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_PLAN_FILE_BYTES = max(1, int(os.getenv("MAX_PLAN_FILE_BYTES", str(12 * 1024 * 1024))))
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
request_windows: defaultdict[str, deque[float]] = defaultdict(deque)
plan_request_windows: defaultdict[str, deque[float]] = defaultdict(deque)
rate_limit_lock = Lock()
project_id_pattern = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
allowed_plan_types = {"image/png", "image/jpeg", "image/webp", "application/pdf"}


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


class ProjectPayload(BaseModel):
    project: dict[str, object]
    upload_metadata: dict[str, object] | None = None


class DetectedRoom(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    category: Literal[
        "bedroom",
        "bathroom",
        "living_room",
        "kitchen",
        "dining_room",
        "balcony",
        "parking",
        "utility",
        "hallway",
        "other",
    ]
    dimensions: str | None = Field(default=None, max_length=80)
    area_sqft: float | None = Field(default=None, ge=0, le=25_000)
    confidence: float = Field(ge=0, le=1)


class FloorPlanAnalysis(BaseModel):
    usable: bool
    summary: str = Field(min_length=1, max_length=500)
    total_area_sqft: float | None = Field(default=None, ge=100, le=25_000)
    area_source: Literal["printed_total", "calculated_from_dimensions", "not_available"]
    bedrooms: int | None = Field(default=None, ge=0, le=20)
    bathrooms: int | None = Field(default=None, ge=0, le=20)
    balconies: int | None = Field(default=None, ge=0, le=20)
    parking_spaces: int | None = Field(default=None, ge=0, le=20)
    property_type: Literal["flat", "villa", "house", "builder_floor", "penthouse", "studio", "plot", "unknown"]
    rooms: list[DetectedRoom] = Field(default_factory=list, max_length=40)
    warnings: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = Field(ge=0, le=1)


floor_plan_response_schema = {
    "type": "OBJECT",
    "properties": {
        "usable": {"type": "BOOLEAN"},
        "summary": {"type": "STRING"},
        "total_area_sqft": {"type": "NUMBER", "nullable": True},
        "area_source": {
            "type": "STRING",
            "enum": ["printed_total", "calculated_from_dimensions", "not_available"],
        },
        "bedrooms": {"type": "INTEGER", "nullable": True},
        "bathrooms": {"type": "INTEGER", "nullable": True},
        "balconies": {"type": "INTEGER", "nullable": True},
        "parking_spaces": {"type": "INTEGER", "nullable": True},
        "property_type": {
            "type": "STRING",
            "enum": ["flat", "villa", "house", "builder_floor", "penthouse", "studio", "plot", "unknown"],
        },
        "rooms": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {"type": "STRING"},
                    "category": {
                        "type": "STRING",
                        "enum": [
                            "bedroom",
                            "bathroom",
                            "living_room",
                            "kitchen",
                            "dining_room",
                            "balcony",
                            "parking",
                            "utility",
                            "hallway",
                            "other",
                        ],
                    },
                    "dimensions": {"type": "STRING", "nullable": True},
                    "area_sqft": {"type": "NUMBER", "nullable": True},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["label", "category", "dimensions", "area_sqft", "confidence"],
            },
        },
        "warnings": {"type": "ARRAY", "items": {"type": "STRING"}},
        "confidence": {"type": "NUMBER"},
    },
    "required": [
        "usable",
        "summary",
        "total_area_sqft",
        "area_source",
        "bedrooms",
        "bathrooms",
        "balconies",
        "parking_spaces",
        "property_type",
        "rooms",
        "warnings",
        "confidence",
    ],
}


def validated_project_id(project_id: str) -> str:
    if not project_id_pattern.fullmatch(project_id):
        raise HTTPException(status_code=400, detail="Invalid project identifier")
    return project_id


def database_connection() -> psycopg.Connection:
    if not DATABASE_URL:
        raise HTTPException(status_code=503, detail="Database is not configured")
    return psycopg.connect(DATABASE_URL, connect_timeout=10)


def initialise_database() -> None:
    if not DATABASE_URL:
        return
    with database_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spacemap_projects (
                project_id VARCHAR(128) PRIMARY KEY,
                project JSONB NOT NULL,
                upload_metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spacemap_predictions (
                id BIGSERIAL PRIMARY KEY,
                query_id UUID,
                project_id VARCHAR(128),
                request JSONB NOT NULL,
                predicted_price_inr NUMERIC(16, 2) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute("ALTER TABLE spacemap_predictions ADD COLUMN IF NOT EXISTS query_id UUID")
        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS spacemap_predictions_query_id_idx
            ON spacemap_predictions (query_id)
            WHERE query_id IS NOT NULL
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS spacemap_predictions_project_created_idx
            ON spacemap_predictions (project_id, created_at DESC)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spaceworth_plan_analyses (
                analysis_id UUID PRIMARY KEY,
                query_id UUID,
                project_id VARCHAR(128),
                file_name VARCHAR(255) NOT NULL,
                file_sha256 CHAR(64) NOT NULL,
                mime_type VARCHAR(100) NOT NULL,
                gemini_model VARCHAR(100) NOT NULL,
                analysis JSONB NOT NULL,
                prediction_request JSONB NOT NULL,
                predicted_price_inr NUMERIC(16, 2) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS spaceworth_plan_analyses_project_created_idx
            ON spaceworth_plan_analyses (project_id, created_at DESC)
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spaceworth_api_keys (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(120) NOT NULL DEFAULT 'API Key',
                key_prefix VARCHAR(12) NOT NULL,
                key_hash CHAR(64) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ,
                enabled BOOLEAN NOT NULL DEFAULT TRUE
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS spaceworth_api_keys_created_idx
            ON spaceworth_api_keys (created_at DESC)
            """
        )


def database_is_ready() -> bool:
    if not DATABASE_URL:
        return False
    try:
        with database_connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            return cursor.fetchone() == (1,)
    except (psycopg.Error, HTTPException):
        return False


def save_prediction(query_id: str, project_id: str | None, request: PropertyInput, price: float) -> None:
    if not DATABASE_URL:
        return
    safe_project_id = validated_project_id(project_id) if project_id else None
    with database_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO spacemap_predictions (query_id, project_id, request, predicted_price_inr)
            VALUES (%s, %s, %s, %s)
            """,
            (query_id, safe_project_id, Jsonb(request.model_dump(mode="json")), price),
        )


def save_plan_analysis(
    analysis_id: str,
    query_id: str,
    project_id: str | None,
    file_name: str,
    file_sha256: str,
    mime_type: str,
    analysis: FloorPlanAnalysis,
    prediction_request: PropertyInput,
    price: float,
) -> None:
    if not DATABASE_URL:
        return
    safe_project_id = validated_project_id(project_id) if project_id else None
    with database_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO spaceworth_plan_analyses (
                analysis_id,
                query_id,
                project_id,
                file_name,
                file_sha256,
                mime_type,
                gemini_model,
                analysis,
                prediction_request,
                predicted_price_inr
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                analysis_id,
                query_id,
                safe_project_id,
                file_name[:255],
                file_sha256,
                mime_type,
                GEMINI_MODEL,
                Jsonb(analysis.model_dump(mode="json")),
                Jsonb(prediction_request.model_dump(mode="json")),
                price,
            ),
        )


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


def enforce_plan_rate_limit(request: Request) -> None:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    client_key = forwarded_for.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with rate_limit_lock:
        timestamps = plan_request_windows[client_key]
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
        if len(timestamps) >= PLAN_RATE_LIMIT_REQUESTS:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0])))
            raise HTTPException(
                status_code=429,
                detail="CAD analysis rate limit reached. Try again shortly.",
                headers={"Retry-After": str(retry_after)},
            )
        timestamps.append(now)


def effective_property_from_analysis(property_input: PropertyInput, analysis: FloorPlanAnalysis) -> PropertyInput:
    updates: dict[str, object] = {}
    if analysis.total_area_sqft is not None:
        updates["area_sqft"] = analysis.total_area_sqft
    if analysis.bedrooms is not None and analysis.bedrooms > 0:
        updates["bedrooms"] = analysis.bedrooms
    if analysis.bathrooms is not None and analysis.bathrooms > 0:
        updates["bathroom"] = analysis.bathrooms
    if analysis.balconies is not None:
        updates["balcony"] = analysis.balconies
    if analysis.parking_spaces is not None:
        updates["car_parking"] = analysis.parking_spaces
    if analysis.property_type != "unknown":
        updates["property_type"] = analysis.property_type
    return property_input.model_copy(update=updates)


async def analyse_floor_plan_with_gemini(file_bytes: bytes, mime_type: str) -> FloorPlanAnalysis:
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="CAD image intelligence is not configured")

    prompt = """
You are an architectural floor-plan reader for an Indian property valuation application.
Inspect only what is visibly supported by the uploaded CAD drawing, blueprint, floor plan, or PDF.
Count bedrooms, bathrooms, balconies, parking spaces, and all recognizable rooms.
Read printed dimensions and printed total area carefully. Set total_area_sqft only when a total area is printed or when visible dimensions allow a defensible calculation. Never infer real square footage from pixel size.
Use null for unknown numeric values. Mark unusable files honestly. Keep warnings short and specific.
Room confidence and overall confidence must reflect visual evidence, not optimism.
Return only JSON matching the required schema.
""".strip()
    request_body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(file_bytes).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
            "responseSchema": floor_plan_response_schema,
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0)) as client:
            response = await client.post(
                url,
                headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                json=request_body,
            )
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail="The CAD intelligence provider could not be reached") from error

    if response.status_code == 429:
        raise HTTPException(status_code=429, detail="The CAD intelligence quota is temporarily exhausted")
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail="The CAD intelligence provider rejected the analysis request")
    try:
        response_payload = response.json()
        response_text = response_payload["candidates"][0]["content"]["parts"][0]["text"]
        return FloorPlanAnalysis.model_validate(json.loads(response_text))
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail="The CAD intelligence provider returned invalid structured data") from error


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
initialise_database()
app = FastAPI(title="SpaceWorth Property Intelligence API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Project-ID"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": "FullRefitHighAccuracyTreeAndNeuralEnsemble",
        "held_out_r2": 0.906449314077493,
        "database": "connected" if database_is_ready() else "disconnected",
        "cad_analysis": "configured" if GEMINI_API_KEY else "not_configured",
        "vision_model": GEMINI_MODEL if GEMINI_API_KEY else None,
    }


@app.get("/project/{project_id}", dependencies=[Depends(verify_api_key)])
def get_project(project_id: str) -> dict[str, object]:
    safe_project_id = validated_project_id(project_id)
    with database_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT project, upload_metadata, updated_at
            FROM spacemap_projects
            WHERE project_id = %s
            """,
            (safe_project_id,),
        )
        row = cursor.fetchone()
    if not row:
        return {"project": None, "upload_metadata": None, "updated_at": None}
    return {
        "project": row[0],
        "upload_metadata": row[1],
        "updated_at": row[2].isoformat(),
    }


@app.put("/project/{project_id}", dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])
def put_project(project_id: str, payload: ProjectPayload) -> dict[str, str]:
    safe_project_id = validated_project_id(project_id)
    with database_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO spacemap_projects (project_id, project, upload_metadata)
            VALUES (%s, %s, %s)
            ON CONFLICT (project_id) DO UPDATE SET
                project = EXCLUDED.project,
                upload_metadata = EXCLUDED.upload_metadata,
                updated_at = NOW()
            RETURNING updated_at
            """,
            (safe_project_id, Jsonb(payload.project), Jsonb(payload.upload_metadata) if payload.upload_metadata else None),
        )
        updated_at = cursor.fetchone()[0]
    return {"status": "saved", "updated_at": updated_at.isoformat()}


@app.get("/project/{project_id}/predictions", dependencies=[Depends(verify_api_key)])
def get_predictions(project_id: str, limit: int = 10) -> dict[str, list[dict[str, object]]]:
    safe_project_id = validated_project_id(project_id)
    safe_limit = min(50, max(1, limit))
    with database_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT query_id, request, predicted_price_inr, created_at
            FROM spacemap_predictions
            WHERE project_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (safe_project_id, safe_limit),
        )
        rows = cursor.fetchall()
    return {
        "predictions": [
            {
                "query_id": str(row[0]) if row[0] else None,
                "request": row[1],
                "predicted_price_inr": float(row[2]),
                "created_at": row[3].isoformat(),
            }
            for row in rows
        ]
    }


@app.post("/predict", dependencies=[Depends(verify_api_key), Depends(enforce_rate_limit)])
def predict(
    request: PropertyInput,
    x_project_id: str | None = Header(default=None, alias="X-Project-ID"),
) -> dict[str, float | str]:
    query_id = str(uuid4())
    price = round(service.predict(request), 2)
    save_prediction(query_id, x_project_id, request, price)
    return {"query_id": query_id, "predicted_price_inr": price}


@app.post(
    "/analyze",
    dependencies=[Depends(verify_api_key), Depends(enforce_plan_rate_limit)],
)
async def analyze_floor_plan(
    file: UploadFile = File(...),
    property: str = Form(...),
    project_id: str | None = Form(default=None),
) -> dict[str, object]:
    mime_type = (file.content_type or "").lower()
    if mime_type not in allowed_plan_types:
        raise HTTPException(status_code=415, detail="Use a PNG, JPG, WEBP, or PDF floor plan")
    file_bytes = await file.read(MAX_PLAN_FILE_BYTES + 1)
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded floor plan is empty")
    if len(file_bytes) > MAX_PLAN_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"The floor plan must be {MAX_PLAN_FILE_BYTES // 1024 // 1024} MB or smaller")
    try:
        property_input = PropertyInput.model_validate_json(property)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="The property context is invalid") from error

    analysis = await analyse_floor_plan_with_gemini(file_bytes, mime_type)
    if not analysis.usable:
        raise HTTPException(
            status_code=422,
            detail=analysis.warnings[0] if analysis.warnings else "The uploaded file is not a readable floor plan",
        )

    prediction_request = effective_property_from_analysis(property_input, analysis)
    predicted_price = round(service.predict(prediction_request), 2)
    query_id = str(uuid4())
    analysis_id = str(uuid4())
    save_prediction(query_id, project_id, prediction_request, predicted_price)
    save_plan_analysis(
        analysis_id=analysis_id,
        query_id=query_id,
        project_id=project_id,
        file_name=file.filename or "floor-plan",
        file_sha256=hashlib.sha256(file_bytes).hexdigest(),
        mime_type=mime_type,
        analysis=analysis,
        prediction_request=prediction_request,
        price=predicted_price,
    )
    return {
        "analysis_id": analysis_id,
        "query_id": query_id,
        "vision_model": GEMINI_MODEL,
        "analysis": analysis.model_dump(mode="json"),
        "prediction_request": prediction_request.model_dump(mode="json"),
        "predicted_price_inr": predicted_price,
    }


class CreateApiKeyRequest(BaseModel):
    name: str = Field(default="API Key", min_length=1, max_length=120)


@app.post("/api-keys", dependencies=[Depends(enforce_rate_limit)])
def create_api_key(payload: CreateApiKeyRequest) -> dict[str, object]:
    raw_key = f"sw_live_{secrets.token_hex(16)}"
    key_prefix = raw_key[-4:]
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    with database_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO spaceworth_api_keys (name, key_prefix, key_hash)
            VALUES (%s, %s, %s)
            RETURNING id, created_at
            """,
            (payload.name.strip(), key_prefix, key_hash),
        )
        row = cursor.fetchone()
    return {
        "id": str(row[0]),
        "name": payload.name.strip(),
        "key": raw_key,
        "key_prefix": key_prefix,
        "created_at": row[1].isoformat(),
        "expires": None,
        "enabled": True,
    }


@app.get("/api-keys")
def list_api_keys() -> dict[str, list[dict[str, object]]]:
    with database_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, name, key_prefix, created_at, expires_at, enabled
            FROM spaceworth_api_keys
            ORDER BY created_at DESC
            LIMIT 100
            """
        )
        rows = cursor.fetchall()
    return {
        "keys": [
            {
                "id": str(row[0]),
                "name": row[1],
                "key_prefix": row[2],
                "created_at": row[3].isoformat(),
                "expires": row[4].isoformat() if row[4] else None,
                "enabled": row[5],
            }
            for row in rows
        ]
    }


@app.patch("/api-keys/{key_id}", dependencies=[Depends(enforce_rate_limit)])
def toggle_api_key(key_id: str, enabled: bool = True) -> dict[str, object]:
    with database_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE spaceworth_api_keys SET enabled = %s WHERE id = %s
            RETURNING id, enabled
            """,
            (enabled, key_id),
        )
        row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"id": str(row[0]), "enabled": row[1]}


@app.delete("/api-keys/{key_id}", dependencies=[Depends(enforce_rate_limit)])
def delete_api_key(key_id: str) -> dict[str, str]:
    with database_connection() as connection, connection.cursor() as cursor:
        cursor.execute("DELETE FROM spaceworth_api_keys WHERE id = %s RETURNING id", (key_id,))
        row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "deleted", "id": str(row[0])}
