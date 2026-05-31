"""
core/config/loader.py

Two-stage config validation:
  Stage 1 — JSON Schema (jsonschema):  structural, type, enum, range checks.
  Stage 2 — Pydantic (AppConfig):      cross-entity references, business rules.

Both stages must pass; the first error encountered aborts with a precise message.

Public API:
    load_config(path) -> AppConfig
    dump_effective_config(cfg)  -> dict  (JSON-serialisable resolved view)

CLI (python -m core.config.loader <path>):
    Validates and pretty-prints the effective config. Exit 1 on any error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import ValidationError

from core.config.models import AppConfig

# Schema is co-located with this package; resolved at import time.
_SCHEMA_PATH = Path(__file__).parent / "schema.json"


def _load_schema() -> dict[str, Any]:
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise ConfigLoadError(f"Config file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ConfigLoadError(f"Config file is not valid JSON: {exc}")


def _validate_schema(raw: dict[str, Any]) -> None:
    """Stage 1: JSON Schema validation. Raises ConfigLoadError with the first violation."""
    schema = _load_schema()
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(raw), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        path_str = " → ".join(str(p) for p in first.absolute_path) or "<root>"
        raise ConfigLoadError(
            f"Config validation failed at [{path_str}]: {first.message}"
        )


def _build_pydantic(raw: dict[str, Any]) -> AppConfig:
    """Stage 2: Pydantic validation. Raises ConfigLoadError with all field errors."""
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        lines = ["Config Pydantic validation failed:"]
        for err in exc.errors():
            loc = " → ".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"  [{loc}]: {err['msg']}")
        raise ConfigLoadError("\n".join(lines)) from exc


def load_config(path: str | Path) -> AppConfig:
    """
    Load, validate (JSON Schema then Pydantic), and return the resolved AppConfig.
    Raises ConfigLoadError on any violation.
    """
    raw = _load_json(Path(path))
    _validate_schema(raw)
    return _build_pydantic(raw)


def dump_effective_config(cfg: AppConfig) -> dict[str, Any]:
    """
    Return a JSON-serialisable dict of the fully resolved config.
    Useful for the /config/effective endpoint and for ops debugging.
    All defaults are materialised; no None values for optional fields that have defaults.
    """
    return cfg.model_dump(mode="json", exclude_none=False)


# ---------------------------------------------------------------------------
# ConfigLoadError
# ---------------------------------------------------------------------------

class ConfigLoadError(Exception):
    """Raised when config loading or validation fails. Message is human-readable."""


# ---------------------------------------------------------------------------
# CLI entry point: python -m core.config.loader <config_path>
# ---------------------------------------------------------------------------

def _cli() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m core.config.loader <config_file.json>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    try:
        cfg = load_config(config_path)
    except ConfigLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    effective = dump_effective_config(cfg)
    print(json.dumps(effective, indent=2, ensure_ascii=False))
    print(f"\n✓ Config '{config_path}' is valid (version={cfg.version})", file=sys.stderr)


if __name__ == "__main__":
    _cli()
