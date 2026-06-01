"""FastAPI app for the online query service."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from core.config import AppConfig, ConfigLoadError, dump_effective_config, load_config
from core.pipeline.factory import PipelineFactory
from core.query.preprocessor import QueryPreprocessor
from core.serving.health import ServiceNotReadyError
from core.serving.registry import ServingRegistry
from core.storage import StorageProvisioner, StorageSettings
from query.routers.preprocess import init_router as init_preprocess_router
from query.routers.preprocess import router as preprocess_router
from query.routers.query import init_query_router
from query.routers.query import router as query_router

_CONFIG_PATH_ENV = "RAG_CONFIG_PATH"
_DEFAULT_CONFIG = Path("configs/base.json")
_SKIP_WARMUP_ENV = "RAG_SKIP_MODEL_WARMUP"
_SKIP_STORAGE_ENV = "RAG_SKIP_STORAGE_PROVISION"


class _AppState:
    config: AppConfig
    serving: ServingRegistry
    storage: StorageProvisioner


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    cfg_path = Path(os.environ.get(_CONFIG_PATH_ENV, str(_DEFAULT_CONFIG)))
    try:
        _state.config = load_config(cfg_path)
    except ConfigLoadError as exc:
        print(f"[FATAL] Config: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    _state.serving = ServingRegistry.from_config(_state.config)
    if os.environ.get(_SKIP_WARMUP_ENV, "").strip() != "1":
        try:
            _state.serving.wait_all_ready()
        except ServiceNotReadyError as exc:
            print(f"[FATAL] Model warm-up: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    _state.storage = StorageProvisioner(StorageSettings.from_env(), _state.config)
    if os.environ.get(_SKIP_STORAGE_ENV, "").strip() != "1":
        try:
            _state.storage.verify_connections()
        except Exception as exc:
            print(f"[FATAL] Storage: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    preprocessor = QueryPreprocessor.from_config(_state.config)
    query_pipeline = PipelineFactory.build_query_pipeline(
        _state.config,
        es=_state.storage.es_client.raw,
        qdrant=_state.storage.qdrant_client.raw,
        serving_registry=_state.serving,
        es_index=_state.storage.es_alias,
        qdrant_collection=_state.storage.qdrant_alias,
        component_overrides={"preprocessor": preprocessor},
    )

    init_preprocess_router(preprocessor)
    init_query_router(query_pipeline)
    yield


app = FastAPI(
    title="RAG Query Service",
    version="0.1.0",
    description=(
        "Online query pipeline: preprocess, retrieve, fuse, rerank, "
        "build context, and generate grounded answers."
    ),
    lifespan=lifespan,
)

app.include_router(query_router)
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
