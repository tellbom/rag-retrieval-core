"""
ingestion/routers/embedding.py

FastAPI router that exposes embedding configuration and a direct embed
endpoint.  Callers (business adapters, ops tools) use this to:

  GET  /embedding/models          — list configured embedding models
  GET  /embedding/models/{model_id} — single model detail
  POST /embedding/embed           — embed arbitrary texts (for testing / adapters)

Design principle: the retrieval core controls the embedding contract.
Callers follow the schema returned by GET /embedding/models — they must
not assume vector names, dimensions, or model IDs.  These come from config.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/embedding", tags=["embedding"])

# App state is injected at router registration time (set by lifespan)
_state: "_EmbeddingRouterState | None" = None


class _EmbeddingRouterState:
    """Holds references injected from the main app lifespan."""
    from core.ingestion.embedder import Embedder
    from core.config.models import AppConfig

    def __init__(self, embedder: "Embedder", cfg: "AppConfig") -> None:
        self.embedder = embedder
        self.cfg = cfg


def init_router(embedder: "Embedder", cfg: "AppConfig") -> None:  # noqa: F821
    """Called from app lifespan after embedder is constructed."""
    global _state
    _state = _EmbeddingRouterState(embedder, cfg)


def _require_state() -> _EmbeddingRouterState:
    if _state is None:
        raise HTTPException(status_code=503, detail="Embedding service not initialised")
    return _state


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class EmbeddingModelInfo(BaseModel):
    """Public description of one configured embedding model."""
    id: str = Field(description="Logical model ID — use this in API calls")
    name: str = Field(description="HuggingFace model name")
    version: str = Field(description="Model version. Changing this requires a full re-index.")
    vector_name: str = Field(description="Qdrant named-vector key for this model")
    dimension: int = Field(description="Vector dimension")
    max_seq_len: int = Field(description="Maximum input tokens for this model")
    normalize: bool = Field(description="Whether vectors are L2-normalised")


class EmbedRequest(BaseModel):
    """Request body for POST /embedding/embed."""
    texts: list[str] = Field(
        min_length=1,
        max_length=256,
        description="List of texts to embed. Each text is embedded by all configured models.",
    )
    model_id: str | None = Field(
        default=None,
        description=(
            "Optional: embed with this specific model only. "
            "If omitted, all configured models are used."
        ),
    )


class EmbedResponse(BaseModel):
    """Response from POST /embedding/embed."""
    count: int = Field(description="Number of texts embedded")
    vectors: dict[str, list[list[float]]] = Field(
        description=(
            "Map of vector_name → list of vectors (one per input text). "
            "Keys match the `vector_name` field from GET /embedding/models."
        )
    )
    model_versions: dict[str, str] = Field(
        description="Map of model_id → version used for this call."
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/models",
    response_model=list[EmbeddingModelInfo],
    summary="List all configured embedding models",
    description=(
        "Returns the ordered list of embedding models as declared in the config. "
        "Callers must use the `vector_name` values from this response when "
        "referencing vectors in search results — never hard-code them."
    ),
)
def list_embedding_models() -> list[EmbeddingModelInfo]:
    state = _require_state()
    return [
        EmbeddingModelInfo(
            id=m.id,
            name=m.name,
            version=m.version,
            vector_name=m.vector_name,
            dimension=m.dimension,
            max_seq_len=m.max_seq_len,
            normalize=m.normalize,
        )
        for m in state.cfg.models.embeddings
    ]


@router.get(
    "/models/{model_id}",
    response_model=EmbeddingModelInfo,
    summary="Get a single embedding model's configuration",
)
def get_embedding_model(model_id: str) -> EmbeddingModelInfo:
    state = _require_state()
    model_map = {m.id: m for m in state.cfg.models.embeddings}
    if model_id not in model_map:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. "
                   f"Available: {list(model_map)}",
        )
    m = model_map[model_id]
    return EmbeddingModelInfo(
        id=m.id,
        name=m.name,
        version=m.version,
        vector_name=m.vector_name,
        dimension=m.dimension,
        max_seq_len=m.max_seq_len,
        normalize=m.normalize,
    )


@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Embed texts using configured model(s)",
    description=(
        "Embeds a list of texts using the configured embedding model(s). "
        "Primarily for operator testing and adapter validation. "
        "Production ingestion goes through the /ingest endpoint. "
        "Maximum 256 texts per call."
    ),
)
def embed_texts(req: EmbedRequest) -> EmbedResponse:
    from core.serving.embed import EmbeddingError

    state = _require_state()
    embedder = state.embedder

    # Filter to one model if specified
    if req.model_id is not None:
        if req.model_id not in embedder.model_ids:
            raise HTTPException(
                status_code=400,
                detail=f"model_id '{req.model_id}' not configured. "
                       f"Available: {embedder.model_ids}",
            )
        clients_to_use = [
            (cfg, client)
            for (cfg, client) in embedder.iter_clients()
            if cfg.id == req.model_id
        ]
    else:
        clients_to_use = embedder.iter_clients()

    vectors_out: dict[str, list[list[float]]] = {}
    versions_out: dict[str, str] = {}

    try:
        for emb_cfg, client in clients_to_use:
            vecs = client.embed(req.texts)
            vectors_out[emb_cfg.vector_name] = vecs
            versions_out[emb_cfg.id] = emb_cfg.version
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return EmbedResponse(
        count=len(req.texts),
        vectors=vectors_out,
        model_versions=versions_out,
    )
