FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_CONFIG_PATH=configs/base.json \
    RAG_HOST=0.0.0.0

WORKDIR /app

COPY . .

RUN if [ -d wheelhouse ] && ls wheelhouse/*.whl >/dev/null 2>&1; then \
        python -m pip install --no-index --find-links=wheelhouse -r requirements.txt; \
    else \
        python -m pip install -r requirements.txt; \
    fi

EXPOSE 8001 8002

# RAG_SERVICE=ingestion -> ingestion.app:app, default port 8001
# RAG_SERVICE=query     -> query.app:app, set RAG_PORT=8002
CMD ["sh", "-c", "python -m uvicorn ${RAG_SERVICE:-ingestion}.app:app --host ${RAG_HOST:-0.0.0.0} --port ${RAG_PORT:-8001}"]
