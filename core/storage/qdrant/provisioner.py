"""
core/storage/qdrant/provisioner.py

Idempotent Qdrant collection provisioning for the RAG retrieval core.

Responsibilities
----------------
1. Create a collection with ONE named-vector space per configured embedding model.
   One chunk = one point; each point carries N named vectors (one per model).
   Deleting a point removes all its vectors atomically — clean CRUD.

2. Create payload indexes on all filterable fields declared in standard_fields,
   plus the fixed system filter fields.  Filters must be pushed INTO the Qdrant
   query (not post-fusion) — payload indexes are what make that fast.

3. Alias support: Qdrant uses collection aliases for zero-downtime rebuild
   (used by P1-10).  An alias "rag_chunks" → "rag_chunks_0_1_0_1_5_0" lets
   retrievers always read from the alias while rebuild creates a new collection
   and switches atomically.

Named vector config per model
------------------------------
    vector_name  = EmbeddingModelConfig.vector_name   (e.g. "bge_base")
    size         = EmbeddingModelConfig.dimension      (e.g. 768)
    distance     = Cosine (normalised vectors; dot-product equivalent)

Idempotency guarantee
---------------------
provision() is safe to call multiple times.  If the collection and indexes
already exist with the correct config, it is a no-op.

Payload field → Qdrant schema type mapping
-------------------------------------------
    keyword  → keyword (payload index type: Keyword)
    text     → text    (payload index: not indexed — full-text is in ES)
    date     → float   (stored as Unix timestamp ms; indexed as Float)
    integer  → integer (Integer)
    float    → float   (Float)
    boolean  → boolean (Bool)

Only filterable fields get a payload index; the rest are stored but not indexed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from core.config.models import AppConfig, FieldDefinition
from core.storage.qdrant.client import QdrantClientWrapper

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Fixed system payload fields always indexed in Qdrant
# (in addition to whatever standard_fields declares as filterable)
# ------------------------------------------------------------------
_SYSTEM_FILTER_FIELDS: list[tuple[str, str]] = [
    # (field_name, config_type)
    ("chunk_id",      "keyword"),
    ("doc_id",        "keyword"),
    ("parent_id",     "keyword"),
    ("business_type", "keyword"),
    ("category",      "keyword"),
    ("hierarchy_level", "integer"),
    ("config_version", "keyword"),
]

_DEFAULT_COLLECTION_BASE = "rag_chunks"


@dataclass
class QdrantProvisionResult:
    collection_name: str
    alias_name: str
    created: bool


def _qdrant_payload_schema(config_type: str) -> qmodels.PayloadSchemaType:
    """Map a config field type to a Qdrant PayloadSchemaType for payload index."""
    mapping = {
        "keyword": qmodels.PayloadSchemaType.KEYWORD,
        "integer": qmodels.PayloadSchemaType.INTEGER,
        "float":   qmodels.PayloadSchemaType.FLOAT,
        "date":    qmodels.PayloadSchemaType.FLOAT,   # stored as epoch-ms float
        "boolean": qmodels.PayloadSchemaType.BOOL,
        # text: not payload-indexed (full-text is in ES)
    }
    return mapping.get(config_type, qmodels.PayloadSchemaType.KEYWORD)


def _collection_name(base: str, config_version: str, model_version: str) -> str:
    """
    Versioned collection name.  Dots replaced with underscores.
    E.g.: rag_chunks_0_1_0_1_5_0
    """
    cv = config_version.replace(".", "_")
    mv = model_version.replace(".", "_")
    return f"{base}_{cv}_{mv}"


class QdrantProvisioner:
    """
    Handles idempotent Qdrant collection + payload index creation.

    Usage
    -----
        provisioner = QdrantProvisioner(qdrant_client_wrapper, cfg)
        result = provisioner.provision()
    """

    def __init__(
        self,
        client: QdrantClientWrapper,
        cfg: AppConfig,
        *,
        base_name: str = _DEFAULT_COLLECTION_BASE,
    ) -> None:
        self._client: QdrantClient = client.raw
        self._cfg = cfg
        self._base_name = base_name

        primary_model_version = cfg.models.embeddings[0].version
        self._collection_name = _collection_name(
            base_name, cfg.version, primary_model_version
        )
        self._alias_name = base_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def alias_name(self) -> str:
        return self._alias_name

    def provision(self) -> QdrantProvisionResult:
        """
        Idempotent: create collection + payload indexes + alias.
        Returns QdrantProvisionResult.
        """
        created = self._ensure_collection()
        self._ensure_payload_indexes()
        self._ensure_alias()
        return QdrantProvisionResult(
            collection_name=self._collection_name,
            alias_name=self._alias_name,
            created=created,
        )

    def alias_switch(self, old_collection: str, new_collection: str) -> None:
        """
        Atomically move the alias from old to new collection.
        Used by P1-10 rebuild.
        """
        operations: list[qmodels.AliasOperations] = []
        current_target = self.get_alias_target()
        if current_target is not None:
            operations.append(
                qmodels.DeleteAliasOperation(
                    delete_alias=qmodels.DeleteAlias(alias_name=self._alias_name)
                )
            )
        operations.append(
            qmodels.CreateAliasOperation(
                create_alias=qmodels.CreateAlias(
                    collection_name=new_collection,
                    alias_name=self._alias_name,
                )
            )
        )

        self._client.update_collection_aliases(
            change_aliases_operations=operations
        )
        logger.info(
            "Qdrant alias '%s' switched: %s → %s",
            self._alias_name, old_collection, new_collection,
        )

    def delete_collection(self, collection_name: str) -> None:
        """Delete a versioned collection after a successful alias switch."""
        try:
            self._client.delete_collection(collection_name)
            logger.info("Deleted old Qdrant collection: %s", collection_name)
        except Exception as exc:
            logger.warning("Could not delete collection %s: %s", collection_name, exc)

    def get_alias_target(self) -> str | None:
        """Return the collection currently pointed to by the alias, or None."""
        try:
            aliases = self._client.get_collection_aliases()
            for alias in aliases.aliases:
                if alias.alias_name == self._alias_name:
                    return alias.collection_name
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_named_vectors_config(self) -> dict[str, qmodels.VectorParams]:
        """
        Build the named_vectors dict: one entry per configured embedding model.
        Key   = EmbeddingModelConfig.vector_name  (e.g. "bge_base")
        Value = VectorParams(size=dimension, distance=Cosine)
        """
        named_vectors: dict[str, qmodels.VectorParams] = {}
        for emb in self._cfg.models.embeddings:
            named_vectors[emb.vector_name] = qmodels.VectorParams(
                size=emb.dimension,
                distance=qmodels.Distance.COSINE,
                # on_disk=True can be set later when corpus is large
            )
        return named_vectors

    def _ensure_collection(self) -> bool:
        """Create the collection if it does not exist. Returns True if created."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection_name in existing:
            logger.info(
                "Qdrant collection already exists: %s", self._collection_name
            )
            return False

        named_vectors = self._build_named_vectors_config()
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=named_vectors,
            # HNSW index config — defaults are fine for CPU intranet;
            # tune m/ef_construct if recall/latency tradeoff needs adjusting
            hnsw_config=qmodels.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000,
            ),
            optimizers_config=qmodels.OptimizersConfigDiff(
                # Disable indexing during bulk ingestion; re-enable after
                indexing_threshold=20000,
            ),
        )
        logger.info(
            "Created Qdrant collection '%s' with %d named vector(s): %s",
            self._collection_name,
            len(named_vectors),
            list(named_vectors.keys()),
        )
        return True

    def _ensure_payload_indexes(self) -> None:
        """
        Create payload indexes on all filterable fields.
        Idempotent: Qdrant ignores create_payload_index if it already exists.
        """
        fields_to_index: list[tuple[str, str]] = []

        # From standard_fields config
        for field_def in self._cfg.standard_fields.fields:
            if field_def.filterable and field_def.type != "text":
                fields_to_index.append((field_def.name, field_def.type))

        # System fields (always indexed regardless of config)
        existing_names = {f[0] for f in fields_to_index}
        for fname, ftype in _SYSTEM_FILTER_FIELDS:
            if fname not in existing_names:
                fields_to_index.append((fname, ftype))

        for field_name, field_type in fields_to_index:
            schema_type = _qdrant_payload_schema(field_type)
            try:
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.debug(
                    "Payload index ensured: %s.%s (%s)",
                    self._collection_name, field_name, schema_type,
                )
            except Exception as exc:
                # Qdrant returns an error if the index already exists with the
                # same type; log and continue rather than aborting.
                logger.debug(
                    "Payload index %s.%s may already exist: %s",
                    self._collection_name, field_name, exc,
                )

        logger.info(
            "Payload indexes ensured on '%s': %d field(s)",
            self._collection_name, len(fields_to_index),
        )

    def _ensure_alias(self) -> None:
        """Create alias if it doesn't already point to this collection."""
        current_target = self.get_alias_target()
        if current_target == self._collection_name:
            logger.info(
                "Qdrant alias '%s' already points to '%s'",
                self._alias_name, self._collection_name,
            )
            return
        if current_target is not None:
            logger.warning(
                "Qdrant alias '%s' exists but points to '%s', not '%s'. "
                "Use alias_switch() to update it.",
                self._alias_name, current_target, self._collection_name,
            )
            return

        self._client.update_collection_aliases(
            change_aliases_operations=[
                qmodels.CreateAliasOperation(
                    create_alias=qmodels.CreateAlias(
                        collection_name=self._collection_name,
                        alias_name=self._alias_name,
                    )
                )
            ]
        )
        logger.info(
            "Created Qdrant alias '%s' → '%s'",
            self._alias_name, self._collection_name,
        )
