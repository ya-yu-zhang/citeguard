FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data/manifest.json ./data/manifest.json
# data/processed and data/indexes are gitignored: they are NOT copied. At startup
# citeguard.bootstrap checks for them and either downloads them from the optional HF dataset
# CITEGUARD_INDEX_DATASET or exits with instructions (mount ./data or run scripts/ingest_docs.py).

RUN pip install --no-cache-dir . huggingface_hub

ENV PORT=8000
ENV CITEGUARD_DATA_DIR=/app/data
EXPOSE 8000

CMD ["sh", "-c", "python -m citeguard.bootstrap && exec uvicorn citeguard.api.app:app --host 0.0.0.0 --port ${PORT}"]
