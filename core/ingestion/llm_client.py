"""
core/ingestion/llm_client.py

Minimal OpenAI-compatible HTTP client for the intranet LLM.

Scope: used exclusively by the Enhancer for batch offline calls.
This is NOT the query-time answer generator (P1-16 will reuse or extend it).

Design rules
------------
- Synchronous (ingestion is offline/batch).
- Timeout and error handling are the caller's concern: this client raises
  LLMCallError on any failure; the Enhancer decides whether to degrade.
- No retry logic here — retry belongs at the pipeline level (per-document).
- JSON-mode request: system prompt instructs the model to respond with
  valid JSON only; caller parses the response.
- No streaming — enhancement is a batch operation; we want the complete
  structured response.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMCallError(Exception):
    """Raised when the LLM call fails for any reason (network, timeout, bad response)."""


class LLMClient:
    """
    Minimal OpenAI-compatible client for the intranet LLM.

    Parameters
    ----------
    endpoint:
        Base URL of the OpenAI-compatible API, e.g. "http://llm.intranet/v1".
        The client appends "/chat/completions".
    model:
        Model name to use in the API request.
    timeout_seconds:
        Request timeout. Enhancement is offline/batch; 60 s default is generous.
    max_tokens:
        Maximum tokens in the response.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        model: str = "intranet-llm",
        timeout_seconds: float = 60.0,
        max_tokens: int = 1024,
        api_key: str = "not-used",   # intranet LLMs often require a placeholder key
    ) -> None:
        self._url = endpoint.rstrip("/") + "/chat/completions"
        self._model = model
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=timeout_seconds,
            write=10.0,
            pool=5.0,
        )
        self._max_tokens = max_tokens
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        history: list[dict] | None = None,
    ) -> str:
        """
        Send a chat completion request and return the assistant message content.

        Parameters
        ----------
        system_prompt:
            Instructs the model on output format (JSON-only for enhancement).
        user_prompt:
            The document text or enhancement request.
        temperature:
            0.0 by default — enhancement fields should be deterministic.
        history:
            Optional prior conversation turns for multi-step calls.
            Each entry must be {"role": "user"|"assistant", "content": "..."}.

        Returns
        -------
        str
            Raw assistant message content.

        Raises
        ------
        LLMCallError
            On network failure, timeout, non-200 status, or missing content.
        """
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._url, json=payload, headers=self._headers)
        except httpx.TimeoutException as exc:
            raise LLMCallError(
                f"LLM request timed out after {self._timeout.read}s"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMCallError(f"LLM request failed: {exc}") from exc

        if resp.status_code != 200:
            raise LLMCallError(
                f"LLM returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        try:
            data: dict[str, Any] = resp.json()
            content: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMCallError(
                f"Unexpected LLM response shape: {exc}. "
                f"Response: {resp.text[:300]}"
            ) from exc

        logger.debug(
            "LLM call OK (model=%s, prompt_len=%d, response_len=%d)",
            self._model, len(user_prompt), len(content),
        )
        return content

    @classmethod
    def from_config(cls, cfg_llm: Any) -> "LLMClient":
        """
        Build from an EnhancementLLMConfig (or any object with .endpoint,
        .model, .api_key_env, .timeout_seconds, .max_tokens attributes).
        """
        api_key = "not-used"
        api_key_env = getattr(cfg_llm, "api_key_env", None)
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise LLMCallError(
                    f"LLM api_key_env '{api_key_env}' is configured but the "
                    "environment variable is not set"
                )

        return cls(
            endpoint=cfg_llm.endpoint,
            model=cfg_llm.model or "intranet-llm",
            timeout_seconds=cfg_llm.timeout_seconds,
            max_tokens=cfg_llm.max_tokens,
            api_key=api_key,
        )
