"""
tests/unit/core/config/test_loader.py

Acceptance criteria for P1-01:
  ✓ Valid config loads without error and exposes a resolved view.
  ✓ Invalid config aborts with a precise, human-readable error message.
  ✓ JSON Schema rejects unknown keys.
  ✓ JSON Schema rejects out-of-range values.
  ✓ Pydantic cross-entity validation: dense retriever referencing unknown model_id fails.
  ✓ Pydantic cross-entity validation: dense retriever with mismatched vector_name fails.
  ✓ mutate_source can only be false (never true).
  ✓ rerank.enabled=true requires top_k and context_top_k.
  ✓ dump_effective_config returns a JSON-serialisable dict with all defaults materialised.
  ✓ Missing required keys fail with a precise path.
  ✓ Non-JSON file fails with a clear error.
  ✓ Missing file fails with a clear error.
"""

from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from core.config.loader import ConfigLoadError, dump_effective_config, load_config
from core.config.models import AppConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_VALID_BASE = _FIXTURES_DIR / "valid_base.json"


def _load_valid() -> dict:
    with _VALID_BASE.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write_tmp(tmp_path: Path, data: dict, filename: str = "cfg.json") -> Path:
    p = tmp_path / filename
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestValidConfig:
    def test_load_returns_app_config(self):
        cfg = load_config(_VALID_BASE)
        assert isinstance(cfg, AppConfig)

    def test_version_parsed(self):
        cfg = load_config(_VALID_BASE)
        assert cfg.version == "0.1.0"

    def test_embedding_model_accessible(self):
        cfg = load_config(_VALID_BASE)
        assert cfg.models.embeddings[0].id == "bge_base"
        assert cfg.models.embeddings[0].dimension == 768

    def test_reranker_accessible(self):
        cfg = load_config(_VALID_BASE)
        assert "bge-reranker" in cfg.models.reranker.name
        assert cfg.models.reranker.max_batch_size == 8
        assert cfg.retrieval.rerank.min_score is None

    def test_retrievers_count(self):
        cfg = load_config(_VALID_BASE)
        assert len(cfg.retrieval.retrievers) == 2

    def test_lexical_retriever_has_no_model_id(self):
        cfg = load_config(_VALID_BASE)
        lex = next(r for r in cfg.retrieval.retrievers if r.type == "lexical")
        assert lex.model_id is None

    def test_dense_retriever_has_model_id(self):
        cfg = load_config(_VALID_BASE)
        dense = next(r for r in cfg.retrieval.retrievers if r.type == "dense")
        assert dense.model_id == "bge_base"

    def test_enhancement_disabled(self):
        cfg = load_config(_VALID_BASE)
        assert cfg.enhancement.enabled is False

    def test_mutate_source_always_false(self):
        cfg = load_config(_VALID_BASE)
        assert cfg.enhancement.mutate_source is False

    def test_dump_effective_config_is_dict(self):
        cfg = load_config(_VALID_BASE)
        effective = dump_effective_config(cfg)
        assert isinstance(effective, dict)
        assert "version" in effective
        assert "retrieval" in effective

    def test_dump_effective_config_serialisable(self):
        cfg = load_config(_VALID_BASE)
        effective = dump_effective_config(cfg)
        # must not raise
        json.dumps(effective)

    def test_dump_effective_config_materialises_defaults(self):
        """Defaults like batch_size and normalize must appear even if not in input."""
        cfg = load_config(_VALID_BASE)
        effective = dump_effective_config(cfg)
        emb = effective["models"]["embeddings"][0]
        assert "batch_size" in emb
        assert "normalize" in emb

    def test_base_json_is_also_valid(self):
        """The canonical configs/base.json must be valid end-to-end."""
        base = Path("configs/base.json")
        if not base.exists():
            pytest.skip("configs/base.json not found from cwd")
        cfg = load_config(base)
        assert cfg.version is not None


# ---------------------------------------------------------------------------
# File-level errors
# ---------------------------------------------------------------------------

class TestFileErrors:
    def test_missing_file_raises(self):
        with pytest.raises(ConfigLoadError, match="not found"):
            load_config(Path("/nonexistent/path/config.json"))

    def test_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not: valid json", encoding="utf-8")
        with pytest.raises(ConfigLoadError, match="not valid JSON"):
            load_config(bad)


# ---------------------------------------------------------------------------
# JSON Schema — structural rejections
# ---------------------------------------------------------------------------

class TestJsonSchemaValidation:
    def test_unknown_top_level_key_rejected(self, tmp_path):
        data = _load_valid()
        data["unknown_key"] = "oops"
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError):
            load_config(p)

    def test_missing_version_rejected(self, tmp_path):
        data = _load_valid()
        del data["version"]
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError, match="version"):
            load_config(p)

    def test_invalid_version_format_rejected(self, tmp_path):
        data = _load_valid()
        data["version"] = "not-semver"
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError):
            load_config(p)

    def test_max_tokens_below_minimum_rejected(self, tmp_path):
        data = _load_valid()
        data["chunking"]["length"]["max_tokens"] = 10  # min is 64
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError):
            load_config(p)

    def test_max_tokens_above_maximum_rejected(self, tmp_path):
        data = _load_valid()
        data["chunking"]["length"]["max_tokens"] = 99999  # max is 8192
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError):
            load_config(p)

    def test_invalid_fusion_method_rejected(self, tmp_path):
        data = _load_valid()
        data["retrieval"]["fusion"]["method"] = "linear_combination"
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError):
            load_config(p)

    def test_retriever_weight_out_of_range_rejected(self, tmp_path):
        data = _load_valid()
        data["retrieval"]["retrievers"][0]["weight"] = 99.0  # max is 10.0
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError):
            load_config(p)

    def test_invalid_field_type_rejected(self, tmp_path):
        data = _load_valid()
        data["standard_fields"]["fields"][0]["type"] = "blob"
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError):
            load_config(p)

    def test_mutate_source_true_rejected(self, tmp_path):
        data = _load_valid()
        data["enhancement"]["mutate_source"] = True
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError):
            load_config(p)

    def test_empty_retrievers_list_rejected(self, tmp_path):
        data = _load_valid()
        data["retrieval"]["retrievers"] = []
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError):
            load_config(p)

    def test_empty_embeddings_list_rejected(self, tmp_path):
        data = _load_valid()
        data["models"]["embeddings"] = []
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError):
            load_config(p)


# ---------------------------------------------------------------------------
# Pydantic — cross-entity validations
# ---------------------------------------------------------------------------

class TestPydanticCrossEntityValidation:
    def test_dense_retriever_unknown_model_id_rejected(self, tmp_path):
        data = _load_valid()
        for r in data["retrieval"]["retrievers"]:
            if r["type"] == "dense":
                r["model_id"] = "nonexistent_model"
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError, match="model_id"):
            load_config(p)

    def test_dense_retriever_mismatched_vector_name_rejected(self, tmp_path):
        data = _load_valid()
        for r in data["retrieval"]["retrievers"]:
            if r["type"] == "dense":
                r["vector_name"] = "wrong_vector"
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError, match="vector_name"):
            load_config(p)

    def test_rerank_enabled_without_top_k_rejected(self, tmp_path):
        data = _load_valid()
        data["retrieval"]["rerank"] = {"enabled": True}  # missing top_k, context_top_k
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError):
            load_config(p)

    def test_rerank_disabled_no_top_k_required(self, tmp_path):
        data = _load_valid()
        data["retrieval"]["rerank"] = {"enabled": False}
        p = _write_tmp(tmp_path, data)
        cfg = load_config(p)
        assert cfg.retrieval.rerank.enabled is False

    def test_dense_retriever_missing_model_id_rejected(self, tmp_path):
        data = _load_valid()
        for r in data["retrieval"]["retrievers"]:
            if r["type"] == "dense":
                del r["model_id"]
        p = _write_tmp(tmp_path, data)
        with pytest.raises(ConfigLoadError):
            load_config(p)


# ---------------------------------------------------------------------------
# Error message quality
# ---------------------------------------------------------------------------

class TestErrorMessageQuality:
    def test_schema_error_contains_field_path(self, tmp_path):
        data = _load_valid()
        data["chunking"]["length"]["max_tokens"] = 1  # too small
        p = _write_tmp(tmp_path, data)
        try:
            load_config(p)
            pytest.fail("Expected ConfigLoadError")
        except ConfigLoadError as exc:
            # Message should mention the path to the bad field
            assert "max_tokens" in str(exc) or "chunking" in str(exc)

    def test_pydantic_error_contains_field_path(self, tmp_path):
        data = _load_valid()
        for r in data["retrieval"]["retrievers"]:
            if r["type"] == "dense":
                r["model_id"] = "bad_model"
        p = _write_tmp(tmp_path, data)
        try:
            load_config(p)
            pytest.fail("Expected ConfigLoadError")
        except ConfigLoadError as exc:
            assert "model_id" in str(exc) or "bad_model" in str(exc)
