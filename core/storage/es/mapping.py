"""
core/storage/es/mapping.py

Derives an Elasticsearch 7.x index mapping from AppConfig.standard_fields.

Design rules
------------
- Field type mapping is explicit, not inferred.
- `keyword` config type -> ES `keyword` (non-analyzed, exact match).
  This is the critical rule: equipment fault codes, document numbers, IDs
  must never go through IK tokenizer or exact-match recall breaks.
- `text` config type -> ES `text` with the declared analyzer (ik_max_word
  or ik_smart). If no analyzer declared, defaults to ik_max_word.
- `date` config type -> ES `date` with standard ISO-8601 format.
- `integer`/`float` -> ES `integer`/`float`.
- `boolean` -> ES `boolean`.
- `filterable: true` -> the field is included in the mapping (always is),
  and for text fields a `.keyword` sub-field is added for exact filtering.
- `highlightable: true` -> stored with term_vector for faster highlighting.
  ES 7.x highlight works without term_vector too, but it is faster with it.

Fixed system fields added regardless of config
------------------------------------------------
These fields are always present and have fixed types that must not drift:
  _chunk_id_raw   - keyword copy of chunk_id for fast ID lookup
  _enhanced       - boolean flag: was LLM enhancement applied?
  _reranked       - boolean: was the result reranked (query-time annotation)

Public API
----------
build_mapping(cfg: AppConfig) -> dict
    Returns the full "mappings" + "settings" dict ready to PUT to ES.
"""

from __future__ import annotations

from typing import Any

from core.config.models import AppConfig, FieldDefinition

# ---------------------------------------------------------------------------
# Analyzer names (must match the analysis block we define in settings)
# ---------------------------------------------------------------------------
_DEFAULT_TEXT_ANALYZER = "ik_max_word"
_SEARCH_ANALYZER = "ik_smart"   # lighter at query time


def _field_to_es_property(field: FieldDefinition) -> dict[str, Any]:
    """Convert one FieldDefinition to its ES property dict."""
    ftype = field.type

    if ftype == "keyword":
        return {"type": "keyword", "ignore_above": 512}

    if ftype == "text":
        analyzer = field.analyzer or _DEFAULT_TEXT_ANALYZER
        prop: dict[str, Any] = {
            "type": "text",
            "analyzer": analyzer,
            "search_analyzer": _SEARCH_ANALYZER,
        }
        if field.highlightable:
            prop["term_vector"] = "with_positions_offsets"
        # Add a .keyword sub-field for exact-match filtering on text fields.
        if field.filterable:
            prop["fields"] = {
                "keyword": {"type": "keyword", "ignore_above": 256}
            }
        return prop

    if ftype == "date":
        return {
            "type": "date",
            "format": "strict_date_optional_time||epoch_millis",
        }

    if ftype == "integer":
        return {"type": "integer"}

    if ftype == "float":
        return {"type": "float"}

    if ftype == "boolean":
        return {"type": "boolean"}

    # Should never reach here; schema.json enums guard this.
    raise ValueError(f"Unknown field type in config: {ftype!r}")


def _system_properties() -> dict[str, Any]:
    """
    Fixed internal fields that every index carries.
    These are NOT declared in standard_fields (they are pipeline-internal).
    """
    return {
        "_enhanced": {"type": "boolean"},
        "_config_version": {"type": "keyword"},
        "_embedding_model_versions": {"type": "keyword"},
    }


def build_mapping(cfg: AppConfig) -> dict[str, Any]:
    """
    Build the full ES index body (settings + mappings) from AppConfig.

    Returns a dict suitable for:
        es.indices.create(index=name, body=build_mapping(cfg))
    or as a component template body.
    """
    # --- Properties from standard_fields ---
    properties: dict[str, Any] = {}
    for field_def in cfg.standard_fields.fields:
        properties[field_def.name] = _field_to_es_property(field_def)

    # --- System fields ---
    properties.update(_system_properties())

    # --- Settings (IK analyzers, performance) ---
    settings: dict[str, Any] = {
        "number_of_shards": 1,          # single-node intranet default; override per env
        "number_of_replicas": 0,        # intranet: set to 1 when multi-node
        "refresh_interval": "30s",      # ingest-optimised; tighten if near-realtime needed
        "max_result_window": 10000,
        # Accurate hit counting is enabled per search request via track_total_hits.
        "analysis": {
            "analyzer": {
                # ik_max_word: fine-grained tokenisation for index
                "ik_max_word_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_max_word",
                    "filter": ["lowercase"],
                },
                # ik_smart: coarser tokenisation for query.
                "ik_smart_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_smart",
                    "filter": ["lowercase"],
                },
            }
        },
    }

    return {
        "settings": settings,
        "mappings": {
            # ES 7.x: _doc is the only type; explicit declaration avoids warnings.
            "dynamic": "strict",
            "properties": properties,
        },
    }


def build_index_name(base_name: str, config_version: str, model_version: str) -> str:
    """
    Build a versioned index name for zero-downtime rebuild.
    Pattern: {base}_{config_version}_{model_version}
    Dots in versions are replaced with underscores for ES compatibility.

    Example: chunks_0_1_0_1_5_0
    """
    cv = config_version.replace(".", "_")
    mv = model_version.replace(".", "_")
    return f"{base_name}_{cv}_{mv}"
