FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data/manifest.json ./data/manifest.json
# optional prebuilt indexes — rebuild at runtime if missing
COPY data/processed ./data/processed
COPY data/indexes ./data/indexes

RUN pip install --no-cache-dir .

ENV PORT=8000
ENV QDRANT_URL=http://qdrant:6333
EXPOSE 8000

CMD ["sh", "-c", "uvicorn citeguard.api.app:app --host 0.0.0.0 --port ${PORT}"]
