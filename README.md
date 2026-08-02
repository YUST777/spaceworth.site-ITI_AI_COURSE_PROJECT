![SpaceWorth Banner](banner.webp)

# SpaceWorth — Production Property Intelligence & Valuation Engine

> **From a 79.59% Baseline to a Live 90.64% Held-out R² Machine Learning Product**
> 
> *Full-stack property valuation platform built for the ITI Machine Learning Engineering Capstone. Combines a 57,058-listing tabular ensemble (LightGBM + CatBoost + PyTorch Entity Embeddings), multimodal CAD floor-plan vision (Gemini 3.5 Flash), FastAPI backend, PostgreSQL database, and a React + TypeScript interactive workspace.*

---

## 📖 The Story — From Assignment Brief to Real Product

When the ITI project guide was assigned, the prompt was straightforward: take the Kaggle Indian House Price dataset (187,531 raw listings), clean the messy columns, compare regressors, export a `.pkl` model, wrap it in FastAPI, and build a React UI.

After building the initial baseline, I kept asking what it would take to make this feel like a **real production engineering product**:

1. **How do we make the model truly accurate without cheating?** Removing target leakage (source price-per-sqft fields), handling high-cardinality Indian localities with target encoding, and proving performance on an untouched 8,559-row test split.
2. **How do we make the property details visual?** Instead of typing numbers into plain forms, what if entering 3 bedrooms generates an interactive, draggable 2D/3D floor-plan canvas?
3. **What if the user has an architectural drawing or blueprint instead of numbers?** An integrated CAD Vision Intelligence pipeline powered by Gemini extracts structured room counts, total sqft, and layout warnings directly from uploaded PNG, JPG, WEBP, or PDF floor plans.
4. **How do we deploy a heavy ML stack live?** Packaging Python, PyTorch, LightGBM, CatBoost, and a 55 MiB ensemble model into Docker and serving it via Railway with a PostgreSQL database on Supabase.

---

## 📊 Model Evolution & Held-Out Benchmarks

The dataset contained 187,531 raw rows across 21 columns. After deduplicating, parsing mixed price formats (Lac, Crore, Call for Price), standardizing floor text, and removing extreme price-per-sqft outliers, **57,058 clean listings** were used (48,499 for fitting, 8,559 for untouched test evaluation).

### Preserved Milestones

| Milestone | Architecture / Strategy | Held-Out $R^2$ | Artifact Size | Key Change |
| :--- | :--- | :---: | :---: | :--- |
| **01 (Baseline)** | Ridge / Linear Regression + OneHot | **79.59%** | 10.2 MiB | Initial cleaning & baseline pipeline |
| **02** | LightGBM Regressor | **85.12%** | 14.5 MiB | Gradient boosting on numerical + categorical |
| **03** | CatBoost + Locality Target Encoding | **88.40%** | 28.1 MiB | Multi-level locality tail encoding (1, 2, 3 tokens) |
| **04** | PyTorch Entity Embedding Regressor | **86.90%** | 34.0 MiB | High-dimensional categorical embeddings |
| **05 (Winner)** | **Blended Ensemble (LightGBM + CatBoost + 3x PyTorch)** | **90.64%** | **55.0 MiB** | Full refit weighted blend ($R^2 = 0.906449$) |

### Final Model Performance (Test Set Evaluation)

- **Held-Out $R^2$ Score**: `0.906449` (**90.64%**)
- **Mean Absolute Error (MAE)**: `₹1,535,857.51`
- **Root Mean Squared Error (RMSE)**: `₹2,883,442.92`
- **Test Set Size**: 8,559 listings (strictly isolated prior to training)
- **Model Storage**: Hosted on Hugging Face at [`duck233/iti-house-price-model`](https://huggingface.co/duck233/iti-house-price-model)

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Client["Client Tier (Frontend React + Vite)"]
        UI["React 18 + TypeScript UI"]
        Konva["Interactive 2D/3D Konva Canvas"]
        Portal["Developer Portal & API Keys Manager"]
    end

    subgraph Service["Inference Tier (FastAPI Engine)"]
        API["FastAPI App (Railway / Docker Container)"]
        CAD["CAD Vision Intelligence (Gemini 3.5 Flash)"]
        Auth["API Key Auth & Rate Limiter"]
    end

    subgraph Persistence["Storage & Artifacts Tier"]
        DB[(Supabase PostgreSQL Database)]
        HF["Hugging Face (55 MiB Ensemble Weights)"]
    end

    UI -->|"POST /predict"| API
    UI -->|"POST /analyze"| API
    Portal -->|"CRUD /api-keys"| API
    API -->|"Multimodal Floorplan OCR"| CAD
    API -->|"Persist Projects, History & Keys"| DB
    API -->|"Fetch Model Weights at Startup"| HF
```

---

## ✨ Key Product Features

### 1. Tabular Valuation Engine (`POST /predict`)
Inputs property parameters (area sqft, super vs carpet area, city/location, locality, society, bedrooms, bathrooms, balconies, parking, floor number, total floors, property type, furnishing, transaction, ownership, facing, overlooking) and predicts valuation in Indian Rupees (INR) with sub-second latency.

### 2. CAD Floor-Plan Intelligence (`POST /analyze`)
Upload an engineering drawing, blueprint, or architectural PDF (PNG, JPG, WEBP, PDF up to 12 MB). Gemini extracts room counts, dimensions, carpet/super area, and warnings. The validated extraction feeds directly into the 90.64% price ensemble.

### 3. Interactive Floor-Plan Canvas
Selecting bedrooms, bathrooms, or area dynamically generates an interactive Konva layout. Rooms can be dragged, resized, added, deleted, renamed, grid-snapped, measured, and toggled between 2D and 3D preview modes.

### 4. API Key Manager & Developer Portal
Full API key lifecycle management (creation, list, toggle enable/disable, permanent delete) stored in Supabase PostgreSQL with secret hashing (`sw_live_...`).

---

## 🚀 Quickstart & Installation

### Option A — One-Command Docker Setup (Recommended)

Run both the FastAPI backend and React frontend together with Docker Compose:

```bash
# Clone repository
git clone https://github.com/YUST777/iti_ai_project.git
cd iti_ai_project

# Spin up full-stack environment with Docker Compose
docker compose up --build
```

- **React Frontend**: `http://localhost:5173`
- **FastAPI Backend & Swagger**: `http://localhost:7860/docs`

---

### Option B — Manual Local Setup

#### Prerequisites
- **Python**: 3.11+
- **Node.js**: 18+ & `npm`
- **Git**

#### 1. Dataset Setup
Download the Kaggle dataset:
- **Dataset Link**: [House Price by Juhi Bhojani (Kaggle)](https://www.kaggle.com/datasets/juhibhojani/house-price)
- Extract `house_prices.csv` into `notebooks/data/house_prices.csv`.

*(Note: Raw CSV files are excluded from Git via `.gitignore` per assignment rules).*

#### 2. Backend Setup (FastAPI)
```bash
# Navigate to backend directory
cd deployment/house-price-space

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (pinned PyTorch & ML packages)
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Download model artifact from Hugging Face
python download_model.py

# Run development server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

#### 3. Frontend Setup (React + TypeScript + Vite)
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite dev server
npm run dev
```

Frontend will be available at `http://localhost:5173`.

---

## 🔑 Environment Variables

### Backend (`deployment/house-price-space/.env`)

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `MODEL_PATH` | Path to local `.pkl` model bundle | `/app/models/house_price.pkl` |
| `DATABASE_URL` | PostgreSQL connection URI (Supabase) | `postgresql://user:pass@host:5432/postgres` |
| `GEMINI_API_KEY` | Google Gemini API Key for CAD vision | `AIzaSy...` |
| `GEMINI_MODEL` | Gemini model name for vision analysis | `gemini-3.5-flash-lite` |
| `ALLOWED_ORIGINS` | CORS allowed origins comma-separated | `*` |
| `RATE_LIMIT_REQUESTS` | Standard API rate limit per window | `30` |

### Frontend (`frontend/.env`)

| Variable | Description | Example |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | Backend FastAPI service URL | `https://iti-house-price-api-production.up.railway.app` |

---

## 📡 API Reference

### 1. Health Check
```bash
curl -X GET https://iti-house-price-api-production.up.railway.app/health
```
**Response**:
```json
{
  "status": "ok",
  "model": "FullRefitHighAccuracyTreeAndNeuralEnsemble",
  "held_out_r2": 0.906449314077493,
  "database": "connected",
  "cad_analysis": "configured",
  "vision_model": "gemini-3.5-flash-lite"
}
```

### 2. Price Prediction (`POST /predict`)
```bash
curl -X POST https://iti-house-price-api-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area_sqft": 1200,
    "area_type": "super",
    "location": "thane",
    "locality": "kolshet road",
    "society": "lodha amara",
    "bedrooms": 2,
    "bathroom": 2,
    "balcony": 1,
    "car_parking": 1,
    "floor_num": 8,
    "total_floors": 24,
    "property_type": "flat",
    "furnishing": "semi_furnished",
    "transaction": "resale",
    "ownership": "freehold"
  }'
```
**Response**:
```json
{
  "query_id": "b3e19f8a-4c2d-4e9b-810a-2f47c9d012e3",
  "predicted_price_inr": 8945200.00
}
```

### 3. API Keys Management (`POST /api-keys`)
```bash
curl -X POST https://iti-house-price-api-production.up.railway.app/api-keys \
  -H "Content-Type: application/json" \
  -d '{"name": "Production Key"}'
```

---

## 📋 ITI Project Deliverables Checklist

- [x] **Notebook & Data Pipeline**: Full cleaning, deduplication, outlier removal, and target encoding in `notebooks/pipeline.py` and `notebooks/train_high_accuracy_full_refit.py`.
- [x] **Multiple Models Compared**: Ridge baseline ($R^2=79.59\%$), LightGBM ($85.12\%$), CatBoost ($88.40\%$), PyTorch NN ($86.90\%$), Blended Ensemble ($90.64\%$).
- [x] **FastAPI Backend**: Operational `/health`, `/predict`, `/analyze`, `/project`, `/api-keys` endpoints with Pydantic validation, CORS, rate limiting, and pytest coverage.
- [x] **React Frontend**: TypeScript + Vite application with live property form, interactive Konva floorplan editor, CAD analyzer, Developer Portal, and Proof story.
- [x] **Database & Cloud Persistence**: PostgreSQL storage via Supabase for project state, prediction traces, and API key management.
- [x] **Clean GitHub History**: Zero credential leaks, `.venv` excluded, raw dataset CSV excluded, production model hosted on Hugging Face.

---

## 📜 License & Acknowledgments

- **Dataset**: Juhi Bhojani — [Kaggle House Price Dataset](https://www.kaggle.com/datasets/juhibhojani/house-price)
- **Model Repository**: [Hugging Face `duck233/iti-house-price-model`](https://huggingface.co/duck233/iti-house-price-model)
- **Course**: ITI Machine Learning & AI Engineering Program
