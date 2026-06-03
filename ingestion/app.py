"""FastAPI app for the offline ingestion service."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from core.config import AppConfig, ConfigLoadError, dump_effective_config, load_config
from core.ingestion.cleaner import Cleaner
from core.ingestion.embedder import Embedder
from core.ingestion.enhancer import EnhancerFactory
from core.ingestion.routers.crud import init_router as init_crud_router
from core.ingestion.routers.crud import router as crud_router
from core.ingestion.routers.embedding import init_router as init_embedding_router
from core.ingestion.routers.embedding import router as embedding_router
from core.ingestion.routers.indexer import init_router as init_indexer_router
from core.ingestion.routers.indexer import router as indexer_router
from core.ingestion.routers.ingest import init_router as init_ingest_router
from core.ingestion.routers.ingest import router as ingest_router
from core.ingestion.semantic_chunker import SemanticChunker
from core.ingestion.structural_chunker import StructuralChunker
from core.pipeline.factory import PipelineFactory
from core.serving.health import ServiceNotReadyError
from core.serving.registry import ServingRegistry
from core.storage import (
    CrudService,
    FailedIndexStore,
    Indexer,
    OriginalTextStore,
    RebuildService,
    ReconciliationCommand,
    RetryCommand,
    StorageProvisioner,
    StorageSettings,
)
from core.storage.es.mapping import build_index_name
from core.storage.qdrant.provisioner import _collection_name

_CONFIG_PATH_ENV = "RAG_CONFIG_PATH"
_DEFAULT_CONFIG_PATH = Path("configs/base.json")
_SKIP_WARMUP_ENV = "RAG_SKIP_MODEL_WARMUP"
_SKIP_STORAGE_ENV = "RAG_SKIP_STORAGE_PROVISION"
_FAILED_INDEX_DB_ENV = "RAG_FAILED_INDEX_DB"
_DEFAULT_FAILED_DB_PATH = Path("data/failed_index_records.db")
_ORIGINAL_TEXT_DIR_ENV = "RAG_ORIGINAL_TEXT_DIR"
_DEFAULT_ORIGINAL_DIR = Path("data/originals")


class _AppState:
    config: AppConfig
    serving: ServingRegistry
    storage: StorageProvisioner
    fail_store: FailedIndexStore
    original_store: OriginalTextStore
    indexer: Indexer
    crud_svc: CrudService
    rebuild_svc: RebuildService
    retry_cmd: RetryCommand
    reconcile_cmd: ReconciliationCommand


_state = _AppState()


def _versioned_store_names(cfg: AppConfig, base_name: str) -> tuple[str, str]:
    primary_version = cfg.models.embeddings[0].version
    return (
        build_index_name(base_name, cfg.version, primary_version),
        _collection_name(base_name, cfg.version, primary_version),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    cfg_path = Path(os.environ.get(_CONFIG_PATH_ENV, str(_DEFAULT_CONFIG_PATH)))
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

    storage_settings = StorageSettings.from_config(_state.config)
    _state.storage = StorageProvisioner(storage_settings, _state.config)
    if os.environ.get(_SKIP_STORAGE_ENV, "").strip() == "1":
        current_es_index, current_qdrant_collection = _versioned_store_names(
            _state.config,
            storage_settings.base_name,
        )
    else:
        try:
            provision_result = _state.storage.provision()
            current_es_index = provision_result.es.index_name
            current_qdrant_collection = provision_result.qdrant.collection_name
        except Exception as exc:
            print(f"[FATAL] Storage: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    db_path = Path(os.environ.get(_FAILED_INDEX_DB_ENV, str(_DEFAULT_FAILED_DB_PATH)))
    original_dir = Path(os.environ.get(_ORIGINAL_TEXT_DIR_ENV, str(_DEFAULT_ORIGINAL_DIR)))
    _state.fail_store = FailedIndexStore(db_path)
    _state.original_store = OriginalTextStore(original_dir)

    cleaner = Cleaner()
    enhancer = EnhancerFactory.from_config(_state.config)
    structural_chunker = StructuralChunker(_state.config.chunking)
    semantic_chunker = SemanticChunker(_state.config.chunking)
    embedder = Embedder.from_config(_state.config, _state.serving)
    _state.indexer = Indexer(
        es=_state.storage.es_client.raw,
        qdrant=_state.storage.qdrant_client.raw,
        fail_store=_state.fail_store,
        es_index=_state.storage.es_alias,
        qdrant_collection=_state.storage.qdrant_alias,
    )

    ingestion_pipeline = PipelineFactory.build_ingestion_pipeline(
        _state.config,
        es=_state.storage.es_client.raw,
        qdrant=_state.storage.qdrant_client.raw,
        fail_store=_state.fail_store,
        serving_registry=_state.serving,
        es_index=_state.storage.es_alias,
        qdrant_collection=_state.storage.qdrant_alias,
        component_overrides={
            "cleaner": cleaner,
            "enhancer": enhancer,
            "structural_chunker": structural_chunker,
            "semantic_chunker": semantic_chunker,
            "embedder": embedder,
            "indexer": _state.indexer,
        },
    )

    _state.crud_svc = CrudService(
        original_store=_state.original_store,
        cleaner=cleaner,
        enhancer=enhancer,
        structural_chunker=structural_chunker,
        semantic_chunker=semantic_chunker,
        embedder=embedder,
        indexer=_state.indexer,
        config_version=_state.config.version,
    )
    _state.rebuild_svc = RebuildService(
        cfg=_state.config,
        es_client=_state.storage.es_client,
        qdrant_client=_state.storage.qdrant_client,
        fail_store=_state.fail_store,
        original_store=_state.original_store,
        cleaner=cleaner,
        enhancer=enhancer,
        structural_chunker=structural_chunker,
        semantic_chunker=semantic_chunker,
        embedder=embedder,
        current_es_index=current_es_index,
        current_qdrant_collection=current_qdrant_collection,
        base_name=storage_settings.base_name,
    )
    _state.retry_cmd = RetryCommand(_state.indexer, _state.fail_store)
    _state.reconcile_cmd = ReconciliationCommand(
        es=_state.storage.es_client.raw,
        qdrant=_state.storage.qdrant_client.raw,
        fail_store=_state.fail_store,
        es_index=_state.storage.es_alias,
        qdrant_collection=_state.storage.qdrant_alias,
    )

    init_embedding_router(embedder, _state.config)
    init_ingest_router(ingestion_pipeline, _state.original_store)
    init_indexer_router(
        _state.indexer,
        _state.fail_store,
        _state.retry_cmd,
        _state.reconcile_cmd,
    )
    init_crud_router(_state.crud_svc, _state.rebuild_svc, _state.original_store)

    yield
    _state.fail_store.close()


app = FastAPI(
    title="RAG Ingestion Service",
    version="0.1.0",
    description="Offline ingestion, CRUD, rebuild, and index operations.",
    lifespan=lifespan,
)

app.include_router(ingest_router)
app.include_router(embedding_router)
app.include_router(indexer_router)
app.include_router(crud_router)


@app.get("/health", tags=["ops"])
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "ingestion",
        "config_version": _state.config.version,
        "pending_failures": _state.fail_store.pending_count(),
        "original_docs": _state.original_store.count(),
    }


@app.get("/config/effective", tags=["ops"])
async def effective_config() -> JSONResponse:
    try:
        effective: dict[str, Any] = dump_effective_config(_state.config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(content=effective)
