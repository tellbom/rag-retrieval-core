"""
core/serving/health.py

Warm-up gate: poll a TEI /health endpoint until the model reports ready,
then allow the caller to proceed. Traffic must not be routed before this
gate opens — a TEI container that has started but not yet loaded its model
will return HTTP 503, not 200.

Public API
----------
wait_until_ready(endpoint, timeout, interval) -> None
    Block until endpoint /health returns 200, or raise ServiceNotReadyError.

is_healthy(endpoint) -> bool
    Single non-blocking probe. Used by lifespan and monitoring loops.
"""

from __future__ import annotations

import time
import logging

import httpx

logger = logging.getLogger(__name__)

_HEALTH_PATH = "/health"


class ServiceNotReadyError(Exception):
    """Raised when a model service does not become ready within the timeout."""


def is_healthy(endpoint: str, timeout_seconds: float = 5.0) -> bool:
    """
    Single health probe. Returns True only when the service responds HTTP 200.
    Does not raise — returns False on any network or non-200 error.
    """
    url = endpoint.rstrip("/") + _HEALTH_PATH
    try:
        resp = httpx.get(url, timeout=timeout_seconds)
        return resp.status_code == 200
    except Exception as exc:  # network error, timeout, etc.
        logger.debug("Health probe failed for %s: %s", url, exc)
        return False


def wait_until_ready(
    endpoint: str,
    *,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 5.0,
    service_name: str = "",
) -> None:
    """
    Block until the TEI service at `endpoint` becomes healthy, or raise
    ServiceNotReadyError if `timeout_seconds` is exceeded.

    This is the warm-up gate: callers (lifespan hooks, pipeline init) must
    call this before issuing any embed or rerank requests.

    Parameters
    ----------
    endpoint:
        Base URL of the TEI service, e.g. "http://localhost:8080".
    timeout_seconds:
        Maximum time to wait. Default 300 s — CPU model load can be slow.
    poll_interval_seconds:
        Seconds between probes.
    service_name:
        Optional label for log messages.
    """
    label = service_name or endpoint
    deadline = time.monotonic() + timeout_seconds
    attempt = 0

    logger.info("Waiting for model service to become ready: %s", label)

    while time.monotonic() < deadline:
        attempt += 1
        if is_healthy(endpoint):
            logger.info("Model service ready after %d probe(s): %s", attempt, label)
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sleep = min(poll_interval_seconds, remaining)
        logger.debug(
            "Model service not ready (attempt %d), retrying in %.1fs: %s",
            attempt, sleep, label,
        )
        time.sleep(sleep)

    raise ServiceNotReadyError(
        f"Model service '{label}' did not become ready within "
        f"{timeout_seconds:.0f}s (after {attempt} probe(s)). "
        "Check that the TEI container is running and the model weights are mounted."
    )
