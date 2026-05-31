"""
core/storage/es/provisioner.py

Idempotent Elasticsearch provisioning for the RAG retrieval core.

Responsibilities
----------------
1. Verify the IK analysis plugin is installed and version-matched to ES build.
2. Create a versioned index with the derived mapping (if it does not exist).
3. Create or update an alias pointing to the versioned index.
4. Provide alias-switch for zero-downtime rebuild (used by P1-10).

Idempotency guarantee
---------------------
Every operation checks existence before acting.  Running provision() twice
on the same cluster/config is safe and produces no side-effects.

Alias pattern
-------------
  alias: {base_name}                          e.g.  rag_chunks
  index: {base_name}_{config_v}_{model_v}     e.g.  rag_chunks_0_1_0_1_5_0

The alias is what retrievers and indexers use.  The versioned index is what
rebuild creates and switches to atomically.

IK plugin check
---------------
ES 7.x: `cat/plugins` must list `analysis-ik`.  Version must match ES build
exactly (e.g. both 7.17.x).  A version mismatch causes tokeniser failures at
index or query time — we surface this as a hard error at startup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from elasticsearch import NotFoundError

from core.config.models import AppConfig
from core.storage.es.client import ESClient
from core.storage.es.mapping import build_index_name, build_mapping

logger = logging.getLogger(__name__)

# The alias name used by all retrievers and indexers
_DEFAULT_BASE_NAME = "rag_chunks"


@dataclass
class ProvisionResult:
    index_name: str
    alias_name: str
    created: bool       # True = freshly created; False = already existed


class ESProvisioner:
    """
    Handles idempotent ES index + alias creation.

    Usage
    -----
        provisioner = ESProvisioner(es_client, cfg)
        result = provisioner.provision()
        # result.alias_name is used by all other ES operations
    """

    def __init__(
        self,
        client: ESClient,
        cfg: AppConfig,
        *,
        base_name: str = _DEFAULT_BASE_NAME,
    ) -> None:
        self._es = client.raw
        self._cfg = cfg
        self._base_name = base_name

        # Derive the primary embedding model version for index naming.
        # Convention: use the first configured embedding model version.
        primary_model_version = cfg.models.embeddings[0].version
        self._index_name = build_index_name(
            base_name, cfg.version, primary_model_version
        )
        self._alias_name = base_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def index_name(self) -> str:
        return self._index_name

    @property
    def alias_name(self) -> str:
        return self._alias_name

    def provision(self) -> ProvisionResult:
        """
        Idempotent: create index + alias if they do not exist.
        Returns a ProvisionResult describing what happened.
        Raises on IK plugin mismatch or ES connection failure.
        """
        self._verify_ik_plugin()
        created = self._ensure_index()
        self._ensure_alias()
        return ProvisionResult(
            index_name=self._index_name,
            alias_name=self._alias_name,
            created=created,
        )

    def alias_switch(self, old_index: str, new_index: str) -> None:
        """
        Atomically move the alias from old_index to new_index.
        Used by P1-10 rebuild for zero-downtime cutover.

        Both operations happen in a single _update_aliases call — ES
        guarantees atomicity at the cluster level.
        """
        body: dict[str, Any] = {
            "actions": [
                {"remove": {"index": old_index, "alias": self._alias_name}},
                {"add":    {"index": new_index, "alias": self._alias_name}},
            ]
        }
        self._es.indices.update_aliases(body=body)
        logger.info(
            "Alias '%s' switched: %s → %s",
            self._alias_name, old_index, new_index,
        )

    def delete_index(self, index_name: str) -> None:
        """Delete a versioned index. Called after a successful alias switch."""
        if self._es.indices.exists(index=index_name):
            self._es.indices.delete(index=index_name)
            logger.info("Deleted old index: %s", index_name)

    def get_alias_target(self) -> str | None:
        """
        Return the index name currently pointed to by the alias,
        or None if the alias does not exist.
        """
        try:
            result = self._es.indices.get_alias(name=self._alias_name)
            # result is {index_name: {aliases: {alias_name: {}}}}
            indices = list(result.keys())
            return indices[0] if indices else None
        except NotFoundError:
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _verify_ik_plugin(self) -> None:
        """
        Verify that analysis-ik is installed and that its version matches
        the ES cluster version.  Hard error on mismatch — a version skew
        silently corrupts tokenisation.
        """
        try:
            plugins_resp = self._es.cat.plugins(format="json")
        except Exception as exc:
            # Some ES setups restrict _cat/plugins; treat as warning not error
            logger.warning(
                "Could not verify IK plugin via _cat/plugins (%s). "
                "Ensure analysis-ik is installed and version-matched.",
                exc,
            )
            return

        installed = {p.get("component", ""): p.get("version", "") for p in plugins_resp}

        if "analysis-ik" not in installed:
            raise RuntimeError(
                "Elasticsearch plugin 'analysis-ik' is not installed. "
                "Install it and restart ES before provisioning. "
                "See deploy/docker/README.md for the intranet mirror procedure."
            )

        # Version-match check: IK version should equal ES version
        es_version = self._es.info().get("version", {}).get("number", "")
        ik_version = installed["analysis-ik"]

        if es_version and ik_version and es_version != ik_version:
            raise RuntimeError(
                f"IK plugin version mismatch: ES={es_version}, IK={ik_version}. "
                "Install the IK version that matches your ES build exactly. "
                "A version skew causes silent tokenisation failures."
            )

        logger.info(
            "IK plugin verified: analysis-ik %s (ES %s)", ik_version, es_version
        )

    def _ensure_index(self) -> bool:
        """Create the versioned index if it does not exist. Returns True if created."""
        if self._es.indices.exists(index=self._index_name):
            logger.info("ES index already exists: %s", self._index_name)
            return False

        body = build_mapping(self._cfg)
        self._es.indices.create(index=self._index_name, body=body)
        logger.info("Created ES index: %s", self._index_name)

        # Force a settings update to enable track_total_hits (belt-and-suspenders
        # in case the index-level setting needs explicit activation on 7.x)
        self._es.indices.put_settings(
            index=self._index_name,
            body={"index": {"max_result_window": 10000}},
        )
        return True

    def _ensure_alias(self) -> None:
        """
        Create alias → index if not already pointing there.
        If the alias exists but points elsewhere, log a warning (do not clobber
        a live alias; use alias_switch() explicitly for that).
        """
        try:
            existing = self._es.indices.get_alias(name=self._alias_name)
            current_targets = list(existing.keys())
            if self._index_name in current_targets:
                logger.info(
                    "Alias '%s' already points to '%s'",
                    self._alias_name, self._index_name,
                )
                return
            logger.warning(
                "Alias '%s' exists but points to %s, not '%s'. "
                "Use alias_switch() to update it.",
                self._alias_name, current_targets, self._index_name,
            )
        except NotFoundError:
            # Alias does not exist; create it
            self._es.indices.put_alias(
                index=self._index_name, name=self._alias_name
            )
            logger.info(
                "Created alias '%s' → '%s'",
                self._alias_name, self._index_name,
            )
