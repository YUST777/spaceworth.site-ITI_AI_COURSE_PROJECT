FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG HF_MODEL_REPO=duck233/iti-house-price-model
ENV HF_MODEL_REPO=${HF_MODEL_REPO}

WORKDIR /app

COPY deployment/house-price-space/requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY deployment/house-price-space/app.py deployment/house-price-space/download_model.py ./
RUN python download_model.py && rm download_model.py

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
