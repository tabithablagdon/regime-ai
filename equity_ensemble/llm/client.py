"""Single LLM interface every agent and the Meta-Agent call through.

Nothing else in the codebase should import an OpenRouter/Anthropic SDK
directly — that's the point of this module (PRD §5.1). Swapping providers,
or falling back to a direct Anthropic call, is a change confined to this
file.
"""

from __future__ import annotations

import json
import os
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMError(RuntimeError):
    """Raised when the LLM gateway fails or returns output that cannot be
    coerced into the requested schema after retries handled by the caller."""


class LLMClient(Protocol):
    """Interface every agent codes against.

    `complete_structured` must return an instance of `response_model` or
    raise `LLMError` — callers (agents/base.py) own the retry-on-validation-
    failure policy from PRD §8.1, not this client.
    """

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
    ) -> ModelT: ...


class OpenRouterClient:
    """Real implementation, routed through OpenRouter (PRD §5.1).

    Uses OpenRouter's JSON-schema-constrained response format so structured
    output is enforced by the provider, not just requested in the prompt.
    Falls back to `OPENROUTER_FALLBACK_MODEL` (if set) on a request failure
    from the primary model.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self._model = model or os.environ["OPENROUTER_MODEL"]
        self._fallback_model = fallback_model or os.environ.get("OPENROUTER_FALLBACK_MODEL") or None
        self._client = client or httpx.AsyncClient(base_url=OPENROUTER_BASE_URL, timeout=60.0)

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
    ) -> ModelT:
        models_to_try = [self._model] + ([self._fallback_model] if self._fallback_model else [])
        last_error: Exception | None = None

        for model in models_to_try:
            try:
                return await self._call(model, system_prompt, user_prompt, response_model)
            except (httpx.HTTPError, LLMError) as exc:
                last_error = exc
                continue

        raise LLMError(f"all models failed: {models_to_try}") from last_error

    async def _call(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
    ) -> ModelT:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
        }
        resp = await self._client.post(
            "/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=payload,
        )
        resp.raise_for_status()
        body = resp.json()

        try:
            content = body["choices"][0]["message"]["content"]
            return response_model.model_validate(json.loads(content))
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"could not parse {response_model.__name__} from model output") from exc


class FakeLLMClient:
    """Test double. Returns a pre-scripted response for each call, in order,
    or a single fixed response if `responses` has length 1.

    Used by every agent's unit tests so they never hit a live gateway.
    """

    def __init__(self, responses: list[BaseModel]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, type[BaseModel]]] = []

    async def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
    ) -> ModelT:
        self.calls.append((system_prompt, user_prompt, response_model))
        next_response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if not isinstance(next_response, response_model):
            raise LLMError(
                f"FakeLLMClient scripted response is {type(next_response).__name__}, "
                f"expected {response_model.__name__}"
            )
        return next_response
