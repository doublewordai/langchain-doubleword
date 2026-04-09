"""Integration tests for ChatDoubleword and ChatDoublewordBatch.

Run only when ``DOUBLEWORD_API_KEY`` is set in the environment, and the chosen
model is reachable on ``api.doubleword.ai``. Override the model with the
``DOUBLEWORD_TEST_MODEL`` env var.

The batch tests additionally exercise tool calling end-to-end, which is the
one piece of inherited ``BaseChatOpenAI`` behaviour we cannot validate without
hitting the real API: it depends on (a) ``autobatcher.BatchOpenAI`` forwarding
``tools=[...]`` and ``tool_choice`` kwargs through to the batch payload, and
(b) Doubleword's batch endpoint accepting them.
"""

import asyncio
import os
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_doubleword import ChatDoubleword, ChatDoublewordBatch

pytestmark = pytest.mark.skipif(
    "DOUBLEWORD_API_KEY" not in os.environ,
    reason="DOUBLEWORD_API_KEY not set; skipping integration tests.",
)

TEST_MODEL = os.environ.get("DOUBLEWORD_TEST_MODEL", "doubleword-default")
TEST_BATCH_MODEL = os.environ.get("DOUBLEWORD_TEST_BATCH_MODEL", TEST_MODEL)


class TestChatDoublewordIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatDoubleword]:
        return ChatDoubleword

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {"model": TEST_MODEL}


# ---------------------------------------------------------------------------
# Tool definitions used by the tool-calling tests below.
# ---------------------------------------------------------------------------


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"It is sunny in {city}."


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


# ---------------------------------------------------------------------------
# ChatDoubleword (real-time) tool calling — sanity check the inherited
# BaseChatOpenAI bind_tools / tool_calls parsing path against Doubleword.
# ---------------------------------------------------------------------------


def test_chat_doubleword_tool_calling_sync() -> None:
    """Real-time chat model should bind tools and emit tool_calls."""
    llm = ChatDoubleword(model=TEST_MODEL, temperature=0).bind_tools([get_weather])
    response = llm.invoke(
        [HumanMessage(content="What's the weather in Paris? Use the tool.")]
    )
    assert isinstance(response, AIMessage)
    assert response.tool_calls, (
        f"expected at least one tool call, got: {response!r}"
    )
    call = response.tool_calls[0]
    assert call["name"] == "get_weather"
    assert "city" in call["args"]
    assert "paris" in call["args"]["city"].lower()


# ---------------------------------------------------------------------------
# ChatDoublewordBatch tool calling — the load-bearing test. Verifies that
# tool definitions survive the autobatcher → Doubleword batch endpoint round
# trip and that responses are parsed back into AIMessage.tool_calls.
# ---------------------------------------------------------------------------


async def test_chat_doubleword_batch_tool_calling() -> None:
    """Tool calling should work end-to-end through the batch endpoint."""
    llm = ChatDoublewordBatch(model=TEST_BATCH_MODEL, temperature=0).bind_tools(
        [get_weather]
    )
    response = await llm.ainvoke(
        [HumanMessage(content="What's the weather in Tokyo? Use the tool.")]
    )
    assert isinstance(response, AIMessage)
    assert response.tool_calls, (
        f"expected at least one tool call from batch path, got: {response!r}"
    )
    call = response.tool_calls[0]
    assert call["name"] == "get_weather"
    assert "city" in call["args"]
    assert "tokyo" in call["args"]["city"].lower()


async def test_chat_doubleword_batch_concurrent_collation() -> None:
    """Concurrent ainvoke calls should be collected into a single batch.

    There's no easy assertion that "this was one batch and not N", but we can
    at least confirm that ``asyncio.gather`` over many parallel calls returns
    valid responses for every input — which is the access pattern that makes
    autobatcher pay off in the first place.
    """
    llm = ChatDoublewordBatch(model=TEST_BATCH_MODEL, temperature=0)
    cities = ["Paris", "Tokyo", "Berlin", "Lisbon", "Cairo"]

    responses = await asyncio.gather(
        *(
            llm.ainvoke([HumanMessage(content=f"Name one landmark in {city}.")])
            for city in cities
        )
    )

    assert len(responses) == len(cities)
    for resp, city in zip(responses, cities, strict=True):
        assert isinstance(resp, AIMessage)
        assert isinstance(resp.content, str)
        assert resp.content.strip(), f"empty response for {city}"


async def test_chat_doubleword_batch_tool_calling_concurrent() -> None:
    """Tool calling under concurrent fan-out — the LangGraph use case.

    Combines the two prior tests: parallel ``ainvoke`` calls, each binding a
    tool. Ensures every parallel branch parses tool_calls correctly.
    """
    llm = ChatDoublewordBatch(model=TEST_BATCH_MODEL, temperature=0).bind_tools(
        [add]
    )
    pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

    responses = await asyncio.gather(
        *(
            llm.ainvoke(
                [HumanMessage(content=f"Use the tool to add {a} and {b}.")]
            )
            for a, b in pairs
        )
    )

    assert len(responses) == len(pairs)
    for resp, (a, b) in zip(responses, pairs, strict=True):
        assert isinstance(resp, AIMessage)
        assert resp.tool_calls, (
            f"missing tool_calls for ({a}, {b}): {resp!r}"
        )
        call = resp.tool_calls[0]
        assert call["name"] == "add"
        assert int(call["args"]["a"]) == a
        assert int(call["args"]["b"]) == b
