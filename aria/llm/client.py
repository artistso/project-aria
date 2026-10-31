"""Minimal client for OpenAI-compatible local inference servers.

Works with llama.cpp's ``llama-server``, LM Studio, Ollama's ``/v1`` shim, vLLM
and anything else exposing ``/v1/chat/completions``.  Standard library only: the
point of a local math channel is that nothing leaves the device.

The one non-standard field is ``grammar`` (GBNF), which llama.cpp uses for
constrained decoding.  Servers that do not understand it ignore it, so the same
client works everywhere.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

__all__ = ["Message", "ChatClient", "OpenAICompatClient", "StubClient", "ClientError"]


class ClientError(RuntimeError):
    """Raised when the inference server cannot be reached or returns nonsense."""


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatClient(Protocol):
    """Anything that can complete a chat conversation."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        grammar: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 384,
    ) -> str:  # pragma: no cover - protocol
        ...


class OpenAICompatClient:
    """HTTP client for a local OpenAI-compatible endpoint.

    Parameters
    ----------
    endpoint:
        Full URL of the chat completions route, e.g.
        ``http://127.0.0.1:8080/v1/chat/completions``.
    model:
        Model name the server expects; local servers usually ignore it but
        require the field.
    transport:
        Injection point for tests: a callable ``(url, payload, headers, timeout)
        -> str``.  Defaults to :func:`_http_post`.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8080/v1/chat/completions",
        model: str = "local",
        api_key: Optional[str] = None,
        timeout: float = 180.0,
        retries: int = 1,
        transport=None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self._transport = transport or _http_post
        self.calls: List[Dict[str, Any]] = []

    # -- request building --------------------------------------------------

    def payload(
        self,
        messages: Sequence[Message],
        grammar: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [message.as_dict() for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if grammar:
            # llama.cpp extension: constrain sampling to the grammar.
            body["grammar"] = grammar
        return body

    def headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # -- public API --------------------------------------------------------

    def complete(
        self,
        messages: Sequence[Message],
        *,
        grammar: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 384,
    ) -> str:
        body = self.payload(messages, grammar, temperature, max_tokens)
        self.calls.append(body)
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                raw = self._transport(self.endpoint, body, self.headers(), self.timeout)
                return _extract_content(raw)
            except (urllib.error.URLError, TimeoutError, ClientError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
        raise ClientError(f"inference request failed: {last_error}")


class StubClient:
    """Deterministic client for tests and for ``--dry-run`` demonstrations.

    ``responses`` may be a list of strings (consumed in order) or a callable
    taking the message list and returning a string.
    """

    def __init__(self, responses: Sequence[str] | Any) -> None:
        self._responses = responses
        self.calls: List[Dict[str, Any]] = []
        self._index = 0

    def complete(
        self,
        messages: Sequence[Message],
        *,
        grammar: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 384,
    ) -> str:
        self.calls.append({"messages": [m.as_dict() for m in messages], "grammar": bool(grammar)})
        if callable(self._responses):
            return str(self._responses(messages))
        if self._index >= len(self._responses):
            raise ClientError("StubClient ran out of scripted responses")
        response = self._responses[self._index]
        self._index += 1
        return response


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _http_post(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:  # pragma: no cover - network dependent
        detail = error.read().decode("utf-8", errors="replace")[:400]
        raise ClientError(f"HTTP {error.code}: {detail}") from error


def _extract_content(raw: str) -> str:
    """Pull the assistant text out of an OpenAI-shaped response."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ClientError(f"response was not JSON: {raw[:200]!r}") from error

    if isinstance(data, dict) and "choices" in data:
        choices = data["choices"]
        if not choices:
            raise ClientError("response contained no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            content = choices[0].get("text", "")
        return str(content)
    # llama.cpp also exposes /completion with a "content" field.
    if isinstance(data, dict) and "content" in data:
        return str(data["content"])
    raise ClientError(f"unrecognised response shape: {list(data)[:6]}")
