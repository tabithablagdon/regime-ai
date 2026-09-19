from __future__ import annotations

from datetime import date

import pytest

from equity_ensemble.agents.base import AgentUnavailable, run_with_validation_retry
from equity_ensemble.llm.client import FakeLLMClient
from equity_ensemble.schemas.models import AgentClaim, Evidence


def _claim(**overrides) -> AgentClaim:
    defaults = dict(
        agent="technicals",
        ticker="AAPL",
        direction="bullish",
        magnitude_bps=100,
        confidence=0.6,
        evidence=[Evidence(source="test", date=date(2026, 1, 1), snippet="...")],
        falsifiers=["something"],
    )
    defaults.update(overrides)
    return AgentClaim(**defaults)


class _WrongModel:
    """Not a Pydantic model — used to force FakeLLMClient's isinstance check
    to fail, simulating a schema-validation failure from the gateway."""


async def test_succeeds_on_first_attempt():
    claim = _claim()
    llm = FakeLLMClient(responses=[claim])

    result = await run_with_validation_retry(
        llm=llm,
        system_prompt="sys",
        build_user_prompt=lambda err: "prompt",
        response_model=AgentClaim,
    )

    assert result is claim
    assert len(llm.calls) == 1


async def test_retries_once_then_succeeds():
    # Force a genuine failure on the first call by handing back a non-AgentClaim,
    # then a valid one on the retry.
    llm = FakeLLMClient(responses=[_WrongModel(), _claim()])

    result = await run_with_validation_retry(
        llm=llm,
        system_prompt="sys",
        build_user_prompt=lambda err: f"prompt (prev_error={err})",
        response_model=AgentClaim,
    )

    assert isinstance(result, AgentClaim)
    assert len(llm.calls) == 2
    # second prompt should have been built with the first error included
    assert "prev_error=None" not in llm.calls[1][1]


async def test_raises_agent_unavailable_after_two_failures():
    llm = FakeLLMClient(responses=[_WrongModel()])

    with pytest.raises(AgentUnavailable):
        await run_with_validation_retry(
            llm=llm,
            system_prompt="sys",
            build_user_prompt=lambda err: "prompt",
            response_model=AgentClaim,
        )

    assert len(llm.calls) == 2
