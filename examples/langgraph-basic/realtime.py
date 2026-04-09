"""Run a basic LangGraph workflow with ChatDoubleword (sync, real-time).

A trivial agent — model + calculator tool + conditional edge — invoked
synchronously through Doubleword's real-time `/v1/chat/completions` endpoint.
Runs a single query and a sequential batch of five, printing wall time for
each.

Requires DOUBLEWORD_API_KEY in the environment (or ~/.dw/credentials.toml).
"""

import os
import time
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langchain_doubleword import ChatDoubleword

MODEL = "Qwen/Qwen3-14B-FP8"

QUERIES = [
    "What is 137 * 49?",
    "What is 100 + 250?",
    "What is 81 / 9?",
    "What is 2 ** 10?",
    "What is 1000 - 333?",
]


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Supports +, -, *, /, **, parentheses, and integer/float literals.
    """
    if not all(c in "0123456789+-*/.()** " for c in expression):
        return f"error: invalid characters in {expression!r}"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as e:
        return f"error: {e}"


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(llm: ChatDoubleword):
    """Build a model + tool agent graph parameterised by the LLM."""
    bound = llm.bind_tools([calculator])

    def model_node(state: State) -> dict:
        return {"messages": [bound.invoke(state["messages"])]}

    def tools_node(state: State) -> dict:
        last = state["messages"][-1]
        tool_messages = []
        for tc in last.tool_calls:
            if tc["name"] == "calculator":
                result = calculator.invoke(tc["args"])
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tc["id"])
                )
        return {"messages": tool_messages}

    def should_continue(state: State) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(State)
    workflow.add_node("model", model_node)
    workflow.add_node("tools", tools_node)
    workflow.set_entry_point("model")
    workflow.add_conditional_edges(
        "model", should_continue, {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "model")
    return workflow.compile()


def run_one(graph, query: str) -> str:
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": 50},
        )
        return result["messages"][-1].content
    except Exception as e:
        return f"<error: {type(e).__name__}: {e}>"


def main() -> None:
    if not os.environ.get("DOUBLEWORD_API_KEY") and not (
        os.path.exists(os.path.expanduser("~/.dw/credentials.toml"))
        and os.path.exists(os.path.expanduser("~/.dw/config.toml"))
    ):
        raise SystemExit(
            "DOUBLEWORD_API_KEY not set and no ~/.dw/credentials.toml found."
        )

    llm = ChatDoubleword(model=MODEL, temperature=0)
    graph = build_graph(llm)

    print("=" * 60)
    print(f"ChatDoubleword (real-time, sync)")
    print(f"Model: {MODEL}")
    print("=" * 60)
    print()

    # Single query
    print(f"--- single query ---")
    start = time.monotonic()
    answer = run_one(graph, QUERIES[0])
    elapsed = time.monotonic() - start
    print(f"  wall time: {elapsed:5.1f}s")
    print(f"  Q: {QUERIES[0]}")
    print(f"  A: {answer[:120]}")
    print()

    # Sequential batch of N queries — sync invoke runs them one after the
    # other, so total time ≈ sum of individual latencies.
    print(f"--- {len(QUERIES)} queries (sequential) ---")
    start = time.monotonic()
    answers = [run_one(graph, q) for q in QUERIES]
    elapsed = time.monotonic() - start
    print(f"  wall time: {elapsed:5.1f}s")
    for q, a in zip(QUERIES, answers):
        print(f"  Q: {q}")
        print(f"  A: {a[:120]}")
    print()


if __name__ == "__main__":
    main()
