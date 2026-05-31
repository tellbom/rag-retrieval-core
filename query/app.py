"""
query/app.py

Query service — online hot path.
P1-01: config loading + skeleton.
P1-02: ServingRegistry warm-up gate.
P1-03: StorageProvisioner connection verification.
P1-11: QueryPreprocessor construction + /query/preprocess endpoint.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from core.config import AppConfig, ConfigLoadError, dump_effective_config, load_config
from core.query.preprocessor import QueryPreprocessor
from core.serving.health import ServiceNotReadyError
from core.serving.registry import ServingRegistry
from core.storage import StorageProvisioner, StorageSettings
from query.routers.preprocess import init_router as init_preprocess_router
from query.routers.preprocess import router as preprocess_router

_CONFIG_PATH_ENV  = "RAG_CONFIG_PATH"
_DEFAULT_CONFIG   = Path("configs/base.json")
_SKIP_WARMUP_ENV  = "RAG_SKIP_MODEL_WARMUP"
_SKIP_STORAGE_ENV = "RAG_SKIP_STORAGE_PROVISION"


def _resolve_config_path() -> Path:
    return Path(os.environ.get(_CONFIG_PATH_ENV, str(_DEFAULT_CONFIG)))


class _AppState:
    config: AppConfig
    serving: ServingRegistry
    storage: StorageProvisioner
    preprocessor: QueryPreprocessor


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    # Stage 1: config
    cfg_path = _resolve_config_path()
    try:
        _state.config = load_config(cfg_path)
    except ConfigLoadError as exc:
        print(f"[FATAL] Config load failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"[query] Config loaded (version={_state.config.version})")

    # Stage 2: model warm-up
    _state.serving = ServingRegistry.from_config(_state.config)
    if os.environ.get(_SKIP_WARMUP_ENV, "").strip() == "1":
        print("[query] Skipping model warm-up")
    else:
        try:
            _state.serving.wait_all_ready()
        except ServiceNotReadyError as exc:
            print(f"[FATAL] Model warm-up failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    # Stage 3: verify storage connections
    _state.storage = StorageProvisioner(StorageSettings.from_env(), _state.config)
    if os.environ.get(_SKIP_STORAGE_ENV, "").strip() == "1":
        print("[query] Skipping storage verification")
    else:
        try:
            _state.storage.verify_connections()
        except Exception as exc:
            print(f"[FATAL] Storage connection failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    # Stage 4: query preprocessor
    _state.preprocessor = QueryPreprocessor.from_config(_state.config)
    init_preprocess_router(_state.preprocessor)
    print("[query] QueryPreprocessor ready")

    print("[query] Ready.")
    yield


app = FastAPI(
    title="RAG Query Service",
    version="0.1.0",
    description=(
        "Online query pipeline — preprocessor → retrievers → fusion → rerank → context → LLM. "
        "Exposes /query/preprocess for adapter validation and debugging."
    ),
    lifespan=lifespan,
)

app.include_router(preprocess_router)


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "query",
        "config_version": _state.config.version,
    }


@app.get("/config/effective", tags=["ops"])
async def effective_config() -> JSONResponse:
    try:
        effective: dict[str, Any] = dump_effective_config(_state.config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(content=effective)
