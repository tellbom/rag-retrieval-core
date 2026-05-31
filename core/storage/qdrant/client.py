"""
core/storage/qdrant/client.py

Thin wrapper around the qdrant-client.

Responsibilities
----------------
- Construction from config (url, api_key optional for intranet).
- Connection ping / verification.
- Expose raw QdrantClient for provisioner and pipeline use.

Design rule: this class owns connection only; no collection logic here.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantClientWrapper:
    """
    Wraps a Qdrant connection.

    Parameters
    ----------
    url:
        Qdrant REST/gRPC URL, e.g. "http://localhost:6333".
    api_key:
        Optional API key (not needed for intranet default installs).
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        timeout: int = 30,
        prefer_grpc: bool = False,
    ) -> None:
        self._url = url
        self._client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
            prefer_grpc=prefer_grpc,
        )

    @property
    def raw(self) -> QdrantClient:
        """Direct access to the underlying QdrantClient."""
        return self._client

    def verify_connection(self) -> None:
        """
        Raise on connection failure.  Called at startup before provisioning.
        """
        try:
            info = self._client.get_collections()
            count = len(info.collections)
            logger.info(
                "Connected to Qdrant at %s (%d collection(s) present)",
                self._url, count,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Cannot reach Qdrant at {self._url}: {exc}. "
                "Verify Qdrant is running."
            ) from exc

    @classmethod
    def from_settings(
        cls,
        url: str,
        api_key: str | None = None,
        timeout: int = 30,
    ) -> "QdrantClientWrapper":
        return cls(url, api_key=api_key, timeout=timeout)
