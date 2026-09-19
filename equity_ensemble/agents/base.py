"""Shared agent scaffolding — owned by Track C (Meta-Agent + Synthesis Logic).

Extracts the "validate -> on failure, retry once with the validation error
appended to context -> on second failure, mark unavailable" pattern (PRD
§8.1) so Track A, Track B, and the Meta-Agent all use one implementation
instead of three copies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar

from pydantic import BaseModel

from equity_ensemble.llm.client import LLMClient, LLMError

ClaimT = TypeVar("ClaimT", bound=BaseModel)


class AgentUnavailable(Exception):
    """Raised when an agent fails schema validation twice in a row (PRD
    §8.1). The Meta-Agent catches this and proceeds in single-agent mode
    with `escalate_to_analyst` forced true."""


class BaseSpecialistAgent(ABC, Generic[ClaimT]):
    """Common shape for the Technicals and Fundamentals/Sentiment agents."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @abstractmethod
    async def run(self, ticker: str, horizon_days: int) -> ClaimT:
        """Produce a schema-valid claim for `ticker`. Implementations should
        call `run_with_validation_retry` around their LLM call rather than
        handling retries themselves."""
        raise NotImplementedError


async def run_with_validation_retry(
    *,
    llm: LLMClient,
    system_prompt: str,
    build_user_prompt: Callable[[str | None], str],
    response_model: type[ClaimT],
) -> ClaimT:
    """PRD §8.1 retry policy: one retry with the validation error appended
    to context; a second failure raises `AgentUnavailable`.

    `build_user_prompt` is `(previous_error: str | None) -> str` so the
    retry attempt can include what went wrong the first time.
    """
    try:
        return await llm.complete_structured(
            system_prompt=system_prompt,
            user_prompt=build_user_prompt(None),
            response_model=response_model,
        )
    except LLMError as first_error:
        try:
            return await llm.complete_structured(
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(str(first_error)),
                response_model=response_model,
            )
        except LLMError as second_error:
            raise AgentUnavailable(
                f"{response_model.__name__} failed schema validation twice"
            ) from second_error
