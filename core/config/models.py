"""
core/config/models.py

Pydantic v2 typed representation of the resolved config.
Used after JSON Schema validation passes; gives the rest of the codebase
typed access with IDE completion and attribute-level error messages.

Design rules:
- One Pydantic model per JSON Schema object.
- `model_config = ConfigDict(extra="forbid")` on every model —
  unknown keys are rejected at the model layer too, not only by JSON Schema.
- No business logic here; this is a pure typed DTO.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# standard_fields
# ---------------------------------------------------------------------------

FieldType = Literal["keyword", "text", "date", "integer", "float", "boolean"]
AnalyzerType = Literal["ik_max_word", "ik_smart", "keyword", "standard"]


class FieldDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: FieldType
    filterable: bool = False
    highlightable: bool = False
    analyzer: AnalyzerType | None = None
    required: bool = False


class StandardFieldsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: list[FieldDefinition] = Field(min_length=1)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class EmbeddingModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    version: str
    endpoint: str
    vector_name: str
    dimension: int = Field(ge=1)
    batch_size: int = Field(default=32, ge=1)
    max_seq_len: int = Field(default=512, ge=1)
    normalize: bool = True


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    endpoint: str
    max_concurrency: int = Field(default=4, ge=1)
    timeout_seconds: float = Field(default=30.0, ge=1.0)


class EnhancementLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str
    model: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=60.0, ge=1.0)
    max_tokens: int = Field(default=1024, ge=1)


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embeddings: list[EmbeddingModelConfig] = Field(min_length=1)
    reranker: RerankerConfig
    enhancement_llm: EnhancementLLMConfig | None = None


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------

StructuralLevel = Literal["heading", "clause", "paragraph", "list_item", "table", "step"]
SemanticStrategy = Literal["similarity_drop", "late_chunking"]


class LengthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(ge=64, le=8192)
    overlap_tokens: int = Field(ge=0)
    context_preservation_tokens: int = Field(ge=0)


class StructuralConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    levels: list[StructuralLevel] = Field(
        default_factory=lambda: ["heading", "paragraph"]
    )


class SemanticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    min_trigger_tokens: int = Field(default=256, ge=64)
    strategy: SemanticStrategy = "late_chunking"


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    length: LengthConfig
    structural: StructuralConfig = Field(default_factory=StructuralConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

RetrieverType = Literal["lexical", "dense"]
RetrieverEngine = Literal["elasticsearch", "qdrant"]
FusionMethod = Literal["rrf", "weighted_rrf"]


class RetrieverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: RetrieverType
    engine: RetrieverEngine
    model_id: str | None = None
    vector_name: str | None = None
    top_k: int = Field(ge=1, le=1000)
    weight: float = Field(ge=0.0, le=10.0)

    @model_validator(mode="after")
    def dense_requires_model_and_vector(self) -> "RetrieverConfig":
        if self.type == "dense":
            if not self.model_id:
                raise ValueError("Dense retriever requires model_id")
            if not self.vector_name:
                raise ValueError("Dense retriever requires vector_name")
        return self


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: FusionMethod
    k: int = Field(default=60, ge=1)
    pool_top_k: int = Field(ge=1)


class RerankConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    top_k: int | None = None
    context_top_k: int | None = None

    @model_validator(mode="after")
    def enabled_requires_top_k(self) -> "RerankConfig":
        if self.enabled:
            if self.top_k is None:
                raise ValueError("rerank.top_k is required when rerank.enabled=true")
            if self.context_top_k is None:
                raise ValueError("rerank.context_top_k is required when rerank.enabled=true")
        return self


class TopKLadder(BaseModel):
    """Four-stage top-k ladder. Values should be non-increasing."""

    model_config = ConfigDict(extra="forbid")

    recall_top_k: int = Field(ge=1)
    rrf_pool_k: int = Field(ge=1)
    rerank_top_k: int = Field(ge=1)
    context_top_k: int = Field(ge=1)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrievers: list[RetrieverConfig] = Field(min_length=1)
    fusion: FusionConfig
    rerank: RerankConfig
    top_k_ladder: TopKLadder


# ---------------------------------------------------------------------------
# enhancement
# ---------------------------------------------------------------------------

DegradationPolicy = Literal["rules_only_and_flag", "fail_fast"]


class DerivedFieldsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: bool = False
    keywords: bool = False
    entities: bool = False
    potential_questions: bool = False
    context_padding: bool = False


class EnhancementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    mutate_source: Literal[False] = False  # non-overridable; LLM never alters canonical text
    derived_fields: DerivedFieldsConfig = Field(default_factory=DerivedFieldsConfig)
    degradation_policy: DegradationPolicy = "rules_only_and_flag"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    """
    Fully resolved, validated application config.

    Cross-entity validation rules enforced here (post JSON Schema):
    - Every dense retriever's model_id must reference a models.embeddings entry.
    - Every dense retriever's vector_name must match that embedding's vector_name.
    """

    model_config = ConfigDict(extra="forbid")

    version: str
    standard_fields: StandardFieldsConfig
    models: ModelsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    enhancement: EnhancementConfig

    @model_validator(mode="after")
    def validate_retriever_model_references(self) -> "AppConfig":
        embedding_index: dict[str, EmbeddingModelConfig] = {
            e.id: e for e in self.models.embeddings
        }
        for retriever in self.retrieval.retrievers:
            if retriever.type != "dense":
                continue
            model_id = retriever.model_id  # already guaranteed non-None by RetrieverConfig
            if model_id not in embedding_index:
                raise ValueError(
                    f"Retriever '{retriever.id}': model_id '{model_id}' not found in models.embeddings"
                )
            expected_vector = embedding_index[model_id].vector_name
            if retriever.vector_name != expected_vector:
                raise ValueError(
                    f"Retriever '{retriever.id}': vector_name '{retriever.vector_name}' "
                    f"does not match embedding '{model_id}' vector_name '{expected_vector}'"
                )
        return self
