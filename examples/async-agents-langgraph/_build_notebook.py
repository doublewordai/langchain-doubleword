"""Build notebook.ipynb from inline cell sources.

Run via `uv run --no-project python _build_notebook.py` to (re)generate
`notebook.ipynb`. Kept committed for transparency about how the notebook
was authored — feel free to delete it if you prefer to edit the .ipynb
directly.
"""

import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.rstrip("\n").split("\n")],
    }


CELLS = [
    # ------------------------------------------------------------------
    md(
        """
# Recursive Research Agents — LangGraph Edition

This notebook is a LangGraph adaptation of the
[Doubleword async-agents workbook](https://docs.doubleword.ai/inference-api/async-agents).
A single root agent breaks a topic into sub-queries, spawns parallel sub-agents,
each of which can recursively spawn its own sub-agents, and finally synthesises
a report. The original workbook hand-rolls every piece of the orchestrator
(JSONL batch construction, file upload, polling, downloading, agent state
machine, parent-resolution logic, ~1200 lines). This notebook deletes all of
that and leans on two pieces of infrastructure:

1. **`ChatDoublewordBatch`** from `langchain-doubleword`. Every concurrent
   `ainvoke` call is collected by `autobatcher` and submitted as a single batch
   through Doubleword's batch endpoint. There is no JSONL on disk, no batch ID
   to track, no polling loop. We just `await llm.ainvoke(...)` like any other
   chat model.
2. **LangGraph**. The agent loop is a tiny `StateGraph` with a `model` node and
   a `tools` node. Recursive sub-agent spawning is just `agent_graph.ainvoke`
   called from inside the `tools` node, fanned out via `asyncio.gather`. All
   in-flight calls — root, sub, sub-sub — hit the same `BatchOpenAI` instance
   and get collated into the same autobatcher windows.

The bulk of what's in this notebook is the *application* logic — tools, prompts,
search-first wiring, the cross-agent reference registry. The orchestrator
itself is about forty lines.
        """
    ),
    md(
        """
## Setup

You'll need:

```bash
export DOUBLEWORD_API_KEY="sk-..."   # or use ~/.dw/credentials.toml
export SERPER_API_KEY="..."           # https://serper.dev
```

Doubleword's [1-hour batch SLA](https://docs.doubleword.ai/inference-api/batches)
is what makes this practical: a recursive research run that touches dozens of
agents over half a dozen batch rounds completes in a day at batch pricing,
instead of weeks at the standard 24-hour SLA — see the cost comparison in
the [original workbook](https://docs.doubleword.ai/inference-api/async-agents).
        """
    ),
    code(
        """
import asyncio
import json
import os
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langchain_doubleword import ChatDoublewordBatch

# Local supporting modules — see the files alongside this notebook.
from prompts import ROOT_AGENT_SYSTEM, SUB_AGENT_SYSTEM
from tools.search import search as serper_search, format_results_for_context
from tools.scrape import fetch_urls

assert os.environ.get("SERPER_API_KEY"), "SERPER_API_KEY must be set"
        """
    ),
    # ------------------------------------------------------------------
    md(
        """
## Configuration

Edit these to taste. The 235B model is the workbook's flagship; the 30B model
is faster and cheaper for iterating on the workflow itself. `MAX_DEPTH=3` lets
the root spawn sub-agents, and those sub-agents spawn sub-sub-agents — three
levels deep is a reasonable balance for most topics. `OUTPUT_DIR` is where
`agent-tree.json`, `summary.json`, and the final report end up after the run.
        """
    ),
    code(
        """
TOPIC = "quantum computing error correction"
MODEL = "Qwen/Qwen3-14B-FP8"
MAX_DEPTH = 3
MAX_ITERATIONS = 8  # per agent
OUTPUT_DIR = Path("results") / TOPIC.lower().replace(" ", "-")[:50]
        """
    ),
    # ------------------------------------------------------------------
    md(
        """
## The model

`ChatDoublewordBatch` is the chat model that wraps Doubleword's batch
endpoint via `autobatcher`. We pass `completion_window="1h"` to opt into the
1-hour SLA, and tighten `batch_window_seconds` from the default `10.0` so the
collator doesn't sit around waiting after the last in-flight call returns.
        """
    ),
    code(
        """
llm = ChatDoublewordBatch(
    model=MODEL,
    temperature=0,
    max_tokens=16384,
    completion_window="1h",
    batch_window_seconds=2.5,
)
        """
    ),
    # ------------------------------------------------------------------
    md(
        """
## Tools

The five tools from the original workbook. Two are *immediate* — they execute
locally inside the `tools` node — and three are *deferred* control-flow
markers: `spawn_agents` triggers recursive sub-agent invocation,
`reference_findings` looks up another agent's completed research from the
session registry, and `write_report` signals completion.

The deferred tools have `raise NotImplementedError` bodies because their
schemas are what we need (for `bind_tools` to expose them to the model), but
their execution is custom and lives in the `tools_node` below — they should
never be `.invoke()`d directly.
        """
    ),
    code(
        '''
@tool
def search(query: str, max_results: int = 5) -> str:
    """Search the web for a specific angle or follow-up query.

    Your topic was already searched when you were created and results are in
    your context — use this only to explore DIFFERENT angles or follow-up
    questions, not to repeat your initial search. Prefer spawning sub-agents
    over calling search multiple times — each sub-agent gets its own
    automatic search and works in parallel.
    """
    result = serper_search(query, max_results=max_results)
    return json.dumps(result)


@tool
def read_pages(urls: list[str]) -> str:
    """Read one or more web pages in parallel.

    Returns each page's content as markdown text (truncated to 15000 chars
    each). Pass ALL the URLs you want to read in a single call — they are
    fetched simultaneously.
    """
    if not urls:
        return json.dumps({"error": "No URLs provided"})
    fetched = fetch_urls(urls)
    pages = []
    for url in urls:
        content = fetched.get(url)
        if content:
            pages.append({"url": url, "content": content[:15000]})
        else:
            pages.append({"url": url, "error": f"Failed to fetch {url}"})
    return json.dumps({"pages": pages})


@tool
def spawn_agents(queries: list[str]) -> str:
    """Spawn parallel sub-agents to research different topics independently.

    Each sub-agent automatically gets web search results for its topic and can
    then read pages, search for new angles, or spawn its own sub-agents.
    Returns their combined findings when all complete. Prefer this over
    calling search multiple times — sub-agents work in parallel.
    """
    raise NotImplementedError(
        "spawn_agents is handled by tools_node — never invoke directly."
    )


@tool
def write_report(report: str) -> str:
    """Write the final research report.

    Call this when you have gathered all findings from your sub-agents and
    any additional research, and are ready to produce the final output.
    """
    raise NotImplementedError(
        "write_report is handled by tools_node — never invoke directly."
    )


@tool
def reference_findings(agent_id: str) -> str:
    """Reference the findings of another agent that has already researched a
    similar or related topic.

    Use this instead of re-searching a topic that another agent has already
    covered. Check the `Other agents in this research session` block in your
    context to see what topics are available and which have completed.
    """
    raise NotImplementedError(
        "reference_findings is handled by tools_node — never invoke directly."
    )
        '''
    ),
    # ------------------------------------------------------------------
    md(
        """
## Search-first messages

Each agent starts life with a web search already executed for its topic and
the results injected as a system message. This is the original workbook's
single most important optimisation: it means an agent's *first* model call
already has data to act on, instead of wasting a batch round on a search.

Wide trees beat deep trees when batching is the bottleneck.
        """
    ),
    code(
        '''
def build_initial_messages(topic: str, is_root: bool) -> list[BaseMessage]:
    """Build the initial message list for a fresh agent, with pre-searched results."""
    system_prompt = ROOT_AGENT_SYSTEM if is_root else SUB_AGENT_SYSTEM
    user_text = (
        f"Research the following topic and produce a comprehensive report: {topic}"
        if is_root
        else f"Research the following topic thoroughly: {topic}"
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ]
    try:
        results = serper_search(topic)
        formatted = format_results_for_context(topic, results)
        messages.append(
            SystemMessage(content=f"Initial search results for your topic:\\n\\n{formatted}")
        )
    except Exception as e:
        messages.append(
            SystemMessage(content=f"Initial search failed ({e}). Use the search tool instead.")
        )
    return messages
        '''
    ),
    # ------------------------------------------------------------------
    md(
        """
## Session registry

Every agent in a research session — root, sub, sub-sub — registers itself in
a session-wide dict held in a `ContextVar`. This is what makes
`reference_findings` work: a sub-agent can look up another agent by ID and
read its completed findings without that information being plumbed through
the LangGraph state.

Why a `ContextVar` and not a module global? Two reasons:

- **Async-safe**. Inside a single `asyncio.gather` fan-out, every spawned
  task gets the same registry by reference (mutations propagate to siblings),
  but separate notebook runs in the same kernel get fresh registries.
- **No global pollution**. Re-running the run cell starts a fresh registry
  rather than accumulating stale entries from previous runs.

The registry also feeds the **per-event log** during the run and the **final
tree print** + **`agent-tree.json` / `summary.json` files** afterwards.
        """
    ),
    code(
        '''
SESSION_REGISTRY: ContextVar[dict] = ContextVar("session_registry")
SESSION_START: ContextVar[float] = ContextVar("session_start")


def _registry() -> dict:
    return SESSION_REGISTRY.get()


def _next_agent_id(prefix: str) -> str:
    """Generate sequential agent IDs from a counter stored in the registry."""
    registry = _registry()
    n = registry.get("_id_counter", 0)
    registry["_id_counter"] = n + 1
    return f"{prefix}-{n}"


def register_agent(
    agent_id: str,
    parent_id: str | None,
    depth: int,
    is_root: bool,
    topic: str,
) -> None:
    _registry()[agent_id] = {
        "agent_id": agent_id,
        "parent_id": parent_id,
        "depth": depth,
        "is_root": is_root,
        "topic": topic,
        "status": "in_progress",
        "findings": "",
        "sources": [],
        "iterations": 0,
        "started_at": time.monotonic() - SESSION_START.get(),
        "completed_at": None,
    }


def update_agent(agent_id: str, **fields) -> None:
    registry = _registry()
    if agent_id in registry:
        registry[agent_id].update(fields)


def build_session_context(for_agent_id: str) -> str:
    """Build the 'Other agents in this session' block injected into model calls."""
    registry = _registry()
    lines = ["Other agents in this research session:"]
    for aid, entry in registry.items():
        if aid.startswith("_") or aid == for_agent_id:
            continue
        has_findings = "yes" if entry.get("findings") else "no"
        topic = entry.get("topic", "")[:80]
        lines.append(
            f"  - {aid} [{entry.get('status', '?')}] (findings: {has_findings}): {topic}"
        )
    if len(lines) == 1:
        return ""
    lines.append("")
    lines.append(
        "Use reference_findings(agent_id) to reuse another agent's "
        "research instead of re-searching the same topic."
    )
    return "\\n".join(lines)


def log_event(agent_id: str, msg: str) -> None:
    """Print a one-line log entry, indented by the agent's depth."""
    registry = _registry()
    entry = registry.get(agent_id, {})
    depth = entry.get("depth", 0)
    elapsed = time.monotonic() - SESSION_START.get()
    indent = "  " * depth
    print(f"[{elapsed:6.1f}s] {indent}{agent_id:14s} {msg}", flush=True)
        '''
    ),
    # ------------------------------------------------------------------
    md(
        """
## State

The agent state is one `TypedDict` per agent. Each call to `agent_graph.ainvoke`
creates a fresh state, so recursive sub-agent calls don't share anything via
state — they communicate purely through the compiled findings the parent's
`tools_node` writes back as a `ToolMessage`, and through the session registry
above.

The `messages` field uses LangGraph's built-in `add_messages` reducer so
appended messages accumulate naturally across nodes.
        """
    ),
    code(
        '''
class AgentState(TypedDict):
    agent_id: str
    parent_id: str | None
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    is_root: bool
    depth: int
    max_depth: int
    max_iterations: int
    iteration: int
    findings: str
    report: str | None
    sources: list[dict]
    children_findings: list[dict]


def build_initial_state(
    topic: str,
    is_root: bool,
    parent_id: str | None = None,
    depth: int = 0,
    max_depth: int = MAX_DEPTH,
    max_iterations: int = MAX_ITERATIONS,
) -> AgentState:
    agent_id = _next_agent_id("root" if is_root else "sub")
    register_agent(agent_id, parent_id, depth, is_root, topic)
    return {
        "agent_id": agent_id,
        "parent_id": parent_id,
        "messages": build_initial_messages(topic, is_root),
        "topic": topic,
        "is_root": is_root,
        "depth": depth,
        "max_depth": max_depth,
        "max_iterations": max_iterations,
        "iteration": 0,
        "findings": "",
        "report": None,
        "sources": [],
        "children_findings": [],
    }
        '''
    ),
    # ------------------------------------------------------------------
    md(
        """
## The agent subgraph

Two nodes:

- **`model_node`**: bind the right tools (root gets `write_report`, sub-agents
  don't), inject the `Other agents in this session` block as a system message,
  and call `await llm.ainvoke(...)`. Logs entry and exit. Marks the agent as
  completed in the registry when the model returns no tool calls.
- **`tools_node`**: dispatch each tool call. Immediate tools run inline. The
  deferred tools each have their own short branch — `spawn_agents` is the
  one that recursively calls `agent_graph.ainvoke` for each child topic and
  fans them out via `asyncio.gather`. Every one of those in-flight ainvoke
  calls hits the same `ChatDoublewordBatch` instance, so autobatcher
  collates them into the same batch.

The conditional edge from `model` ends the loop when the response has no
tool calls, when the report has been written, or when `max_iterations` is
hit. Otherwise it goes to `tools` and back to `model`.

`agent_graph` is referenced by `tools_node` *by name* — Python resolves it
at call time, so the recursion works fine even though the variable doesn't
exist yet when the function is defined.
        """
    ),
    code(
        '''
async def model_node(state: AgentState) -> dict:
    """Bind tools, call the LLM, record findings."""
    log_event(state["agent_id"], "→ model")

    # Inject the session context block as a one-shot system message before the
    # LLM call. We don't write it back into state["messages"] so it stays
    # fresh on every iteration (rather than accumulating).
    context = build_session_context(state["agent_id"])
    call_messages: list[BaseMessage] = list(state["messages"])
    if context:
        call_messages.append(SystemMessage(content=context))

    if state["is_root"]:
        bound = llm.bind_tools(
            [search, read_pages, spawn_agents, reference_findings, write_report]
        )
    else:
        bound = llm.bind_tools(
            [search, read_pages, spawn_agents, reference_findings]
        )

    response = await bound.ainvoke(call_messages)

    if response.tool_calls:
        names = [tc["name"] for tc in response.tool_calls]
        log_event(state["agent_id"], f"← tools: {', '.join(names)}")
    else:
        log_event(state["agent_id"], "← stop")

    update: dict = {"messages": [response]}
    if not response.tool_calls:
        update["findings"] = response.content or ""
        update_agent(
            state["agent_id"],
            status="completed",
            findings=response.content or "",
            completed_at=time.monotonic() - SESSION_START.get(),
        )
    return update
        '''
    ),
    code(
        '''
async def tools_node(state: AgentState) -> dict:
    """Execute pending tool calls — immediate inline, deferred via custom logic."""
    last = state["messages"][-1]
    assert isinstance(last, AIMessage), "tools_node should only run after model_node"

    tool_messages: list[ToolMessage] = []
    new_sources = list(state.get("sources", []))
    new_children = list(state.get("children_findings", []))
    report = state.get("report")

    for tc in last.tool_calls:
        name = tc["name"]
        args = tc["args"]
        tcid = tc["id"]

        if name == "search":
            try:
                result = await asyncio.to_thread(
                    serper_search, args["query"], args.get("max_results", 5)
                )
                tool_messages.append(
                    ToolMessage(content=json.dumps(result), tool_call_id=tcid)
                )
            except Exception as e:
                tool_messages.append(
                    ToolMessage(content=json.dumps({"error": str(e)}), tool_call_id=tcid)
                )

        elif name == "read_pages":
            urls = args.get("urls", [])
            if not urls:
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps({"error": "No URLs provided"}),
                        tool_call_id=tcid,
                    )
                )
                continue
            fetched = await asyncio.to_thread(fetch_urls, urls)
            pages = []
            for url in urls:
                content = fetched.get(url)
                if content:
                    pages.append({"url": url, "content": content[:15000]})
                    title = content[:100].split("\\n")[0]
                    new_sources.append({"url": url, "title": title})
                else:
                    pages.append({"url": url, "error": f"Failed to fetch {url}"})
            tool_messages.append(
                ToolMessage(content=json.dumps({"pages": pages}), tool_call_id=tcid)
            )

        elif name == "spawn_agents":
            depth = state["depth"]
            max_depth = state["max_depth"]
            if depth >= max_depth:
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "error": (
                                    f"Maximum depth ({max_depth}) reached. "
                                    "Research this topic directly using "
                                    "search and read_pages instead."
                                )
                            }
                        ),
                        tool_call_id=tcid,
                    )
                )
                continue

            queries = args.get("queries", [])
            if not queries:
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps({"error": "No queries provided"}),
                        tool_call_id=tcid,
                    )
                )
                continue

            log_event(
                state["agent_id"],
                f"⤴ spawn {len(queries)} children: {[q[:30] for q in queries]}",
            )

            # The recursion. Each child invokes the same compiled graph.
            # asyncio.gather fans them out concurrently — autobatcher then
            # collects every in-flight ainvoke call (this agent's siblings,
            # this agent's children, grandchildren — anything in flight at
            # the same time) into the same batch window.
            child_states = [
                build_initial_state(
                    topic=q,
                    is_root=False,
                    parent_id=state["agent_id"],
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_iterations=state["max_iterations"],
                )
                for q in queries
            ]
            child_results = await asyncio.gather(
                *(agent_graph.ainvoke(cs) for cs in child_states)
            )

            new_children.extend(child_results)
            for child in child_results:
                new_sources.extend(child.get("sources", []))

            compiled = [
                {
                    "agent_id": child.get("agent_id"),
                    "topic": child["topic"],
                    "findings": child.get("findings") or "(no findings)",
                    "verified_sources": child.get("sources", []),
                }
                for child in child_results
            ]
            tool_messages.append(
                ToolMessage(
                    content=json.dumps({"sub_agent_results": compiled}),
                    tool_call_id=tcid,
                )
            )

        elif name == "reference_findings":
            ref_id = args.get("agent_id", "")
            ref_entry = _registry().get(ref_id)
            if ref_entry and ref_entry.get("findings"):
                log_event(state["agent_id"], f"↪ reference_findings({ref_id}): hit")
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "agent_id": ref_id,
                                "status": ref_entry.get("status"),
                                "findings": ref_entry["findings"],
                            }
                        ),
                        tool_call_id=tcid,
                    )
                )
            else:
                log_event(state["agent_id"], f"↪ reference_findings({ref_id}): miss")
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {
                                "error": f"Agent {ref_id} not found or has no findings yet."
                            }
                        ),
                        tool_call_id=tcid,
                    )
                )

        elif name == "write_report":
            report = args.get("report", "")
            update_agent(
                state["agent_id"],
                status="completed",
                findings=report,
                completed_at=time.monotonic() - SESSION_START.get(),
            )
            log_event(state["agent_id"], "✓ write_report")
            tool_messages.append(
                ToolMessage(
                    content=json.dumps({"status": "Report saved"}),
                    tool_call_id=tcid,
                )
            )

        else:
            tool_messages.append(
                ToolMessage(
                    content=json.dumps({"error": f"Unknown tool: {name}"}),
                    tool_call_id=tcid,
                )
            )

    new_iteration = state.get("iteration", 0) + 1
    update_agent(state["agent_id"], sources=new_sources, iterations=new_iteration)

    return {
        "messages": tool_messages,
        "sources": new_sources,
        "children_findings": new_children,
        "report": report,
        "iteration": new_iteration,
    }


def should_continue(state: AgentState) -> str:
    if state.get("report"):
        return END
    if state.get("iteration", 0) >= state["max_iterations"]:
        return END
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


workflow = StateGraph(AgentState)
workflow.add_node("model", model_node)
workflow.add_node("tools", tools_node)
workflow.set_entry_point("model")
workflow.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "model")
agent_graph = workflow.compile()
        '''
    ),
    # ------------------------------------------------------------------
    md(
        """
## Force-complete + synthesis fallback

Two situations leave the root agent without a final report:

1. **The root hits `max_iterations` without calling `write_report`.** Its
   loop ends because `should_continue` returns `END`, but no report exists.
2. **The root spent its budget on tool calls and never reached a stop.**
   Same outcome — no report.

In both cases we want to do what the original workbook does: take all the
findings collected so far, give the model one more shot at synthesising a
report, but with **no tools bound** so the response content *is* the report.

`run_research` wraps `agent_graph.ainvoke` to handle this. It also force-marks
any agents still in `in_progress` as `incomplete` so the registry reflects
reality before we print the tree.
        """
    ),
    code(
        '''
SYNTHESIS_PROMPT = """\\
All research is now complete. Based on all the findings above, write a \\
comprehensive, well-structured research report in markdown. Include an \\
executive summary, thematic sections with source citations, areas where \\
sources disagree, and areas for further research.

CITATION RULES:
- Only cite URLs from the verified sources list below.
- Do not cite URLs from search snippets or invent URLs.
- If a finding has no verified URL, state it without a link.

Output ONLY the report — no preamble or commentary."""


async def run_research(topic: str) -> dict:
    """Run a full research session with force-complete + synthesis fallback."""
    SESSION_REGISTRY.set({})
    SESSION_START.set(time.monotonic())

    initial = build_initial_state(topic=topic, is_root=True)
    print(f"Starting research: {topic}")
    print(f"Root: {initial['agent_id']}")
    print()

    result = await agent_graph.ainvoke(initial, config={"recursion_limit": 200})

    # Force-complete: any agent still in_progress hit max_iterations.
    registry = _registry()
    for aid, entry in registry.items():
        if aid.startswith("_"):
            continue
        if entry.get("status") == "in_progress":
            entry["status"] = "incomplete"
            if not entry.get("findings"):
                entry["findings"] = "Max iterations reached before completion."
            entry["completed_at"] = time.monotonic() - SESSION_START.get()

    # If the root has no report, do one final tools-removed synthesis round.
    if not result.get("report"):
        print()
        print("Root did not call write_report; running synthesis fallback...")
        # De-dupe sources from the entire tree.
        all_sources = list({s["url"]: s for s in result.get("sources", [])}.values())
        sources_block = ""
        if all_sources:
            source_lines = [f"- [{s['title']}]({s['url']})" for s in all_sources]
            sources_block = (
                "\\n\\nVERIFIED SOURCES — these URLs were actually fetched and "
                "read during research. Use ONLY these for citations:\\n"
                + "\\n".join(source_lines)
            )

        synthesis_msg = HumanMessage(content=SYNTHESIS_PROMPT + sources_block)
        synthesis_response = await llm.ainvoke(result["messages"] + [synthesis_msg])
        result["report"] = synthesis_response.content or ""
        update_agent(
            result["agent_id"],
            status="completed",
            findings=result["report"],
            completed_at=time.monotonic() - SESSION_START.get(),
        )

    return result
        '''
    ),
    # ------------------------------------------------------------------
    md(
        """
## Run

`run_research` does everything: resets the registry, runs the recursive
graph, applies the force-complete + synthesis fallback if needed, and
returns the result. The live event log prints as agents transition.
        """
    ),
    code(
        """
result = await run_research(TOPIC)
        """
    ),
    # ------------------------------------------------------------------
    md(
        """
## Results

After the run, we have a fully populated registry. We can:

1. Print a Unicode tree of the agent hierarchy with statuses.
2. Dump the registry as `agent-tree.json` and aggregate stats as
   `summary.json` next to the notebook.
3. Print (and save) the final report.
        """
    ),
    code(
        '''
def print_tree() -> None:
    """Walk the registry and print the full agent tree."""
    registry = _registry()
    children_by_parent: dict[str | None, list[str]] = {}
    for aid, entry in registry.items():
        if aid.startswith("_"):
            continue
        children_by_parent.setdefault(entry.get("parent_id"), []).append(aid)

    STATUS_ICON = {
        "in_progress": "◉",
        "completed": "●",
        "failed": "✗",
        "incomplete": "○",
    }

    def _walk(aid: str, prefix: str, is_last: bool) -> None:
        entry = registry[aid]
        connector = "└─ " if is_last else "├─ "
        icon = STATUS_ICON.get(entry.get("status", "?"), "?")
        topic = entry.get("topic", "")[:60]
        elapsed = entry.get("completed_at") or 0
        print(f"  {prefix}{connector}{icon} {aid} ({elapsed:.1f}s) {topic}")
        children = children_by_parent.get(aid, [])
        new_prefix = prefix + ("   " if is_last else "│  ")
        for i, cid in enumerate(children):
            _walk(cid, new_prefix, i == len(children) - 1)

    roots = children_by_parent.get(None, [])
    for rid in roots:
        _walk(rid, "", True)


def write_session_files(out_dir: Path) -> None:
    """Dump agent-tree.json + summary.json + report.md into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = _registry()
    agents = [
        entry for aid, entry in registry.items() if not aid.startswith("_")
    ]

    with open(out_dir / "agent-tree.json", "w") as f:
        json.dump(agents, f, indent=2)

    counts: dict[str, int] = {}
    for entry in agents:
        status = entry.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    summary = {
        "topic": TOPIC,
        "model": MODEL,
        "total_agents": len(agents),
        "by_status": counts,
        "max_depth": max((a.get("depth", 0) for a in agents), default=0),
        "elapsed_seconds": time.monotonic() - SESSION_START.get(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    report = result.get("report") or result.get("findings") or "(no output)"
    with open(out_dir / "report.md", "w") as f:
        f.write(report)


print()
print("=" * 60)
print(f"Topic: {TOPIC}")
print(f"Total agents: {len([k for k in _registry() if not k.startswith('_')])}")
print(f"Sources collected: {len(result.get('sources', []))}")
print("=" * 60)
print()
print_tree()

write_session_files(OUTPUT_DIR)
print(f"\\nWrote {OUTPUT_DIR}/report.md, agent-tree.json, summary.json")

print()
print("=" * 60)
print("REPORT")
print("=" * 60)
print(result.get("report") or result.get("findings") or "(no output)")
        '''
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).parent / "notebook.ipynb"
    out.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
