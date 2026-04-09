# Async Agents — LangGraph Edition

A LangGraph + `langchain-doubleword` adaptation of the
[Doubleword async-agents workbook](https://docs.doubleword.ai/inference-api/async-agents).

The original (~1200 lines) hand-rolls every piece of the orchestrator:
JSONL batch construction, file upload, batch creation, polling, downloading
results, parsing finish reasons, dispatching tools, tracking agent state,
resolving waiting parents. This adaptation deletes essentially all of it
by leaning on two pieces of infrastructure:

1. **`ChatDoublewordBatch`** — every concurrent `ainvoke` call is collected
   by `autobatcher` and submitted as a single batch through Doubleword's
   batch API. The `batch.py` plumbing in the original (~150 lines) is
   completely gone.
2. **LangGraph** — the orchestrator loop is a `StateGraph` with model and
   tools nodes. Recursive sub-agent spawning is just a recursive
   `agent_graph.ainvoke(...)` call from inside the tools node, fanned out
   via `asyncio.gather`. All the in-flight calls — root, sub, sub-sub — hit
   the same `BatchOpenAI` instance and get collated into the same batches.

The result is a single notebook (`notebook.ipynb`) that fits in your head.

## What's the same

- Search-first agent creation (search runs at agent creation time, results
  injected into the first prompt — minimizes batch rounds per agent)
- Same five tools: `search`, `read_pages`, `spawn_agents`,
  `reference_findings`, `write_report`
- Same prompts (`prompts.py` is copied verbatim)
- Same Serper / Jina HTTP wrappers (`tools/search.py` and `tools/scrape.py`
  copied verbatim)
- Same Qwen3-VL XML tool-call fallback (some Qwen models emit
  `<tool_call>...</tool_call>` in content rather than the structured
  `tool_calls` field)

## What's different

- No CLI, no click — open the notebook, edit the topic, run all cells.
- No JSONL files on disk, no batch IDs to track, no manual polling.
- No `Agent` dataclass, no `AgentRegistry`, no `process_responses`,
  `execute_pending_tools`, `resolve_waiting_parents` — all replaced by
  the natural recursion of the LangGraph subgraph.
- The `1h` Doubleword batch completion window is set explicitly via
  `ChatDoublewordBatch(completion_window="1h")`.

## Running

You'll need:

```bash
export DOUBLEWORD_API_KEY="sk-..."  # or use ~/.dw/credentials.toml
export SERPER_API_KEY="..."          # https://serper.dev
```

Then:

```bash
cd examples/async-agents-langgraph
uv sync
uv run jupyter lab notebook.ipynb
```

Edit the `TOPIC` and `MODEL` constants in the configuration cell, then
run all cells.

## Layout

```
async-agents-langgraph/
├── pyproject.toml
├── README.md             ← you are here
├── notebook.ipynb        ← the workbook
├── prompts.py            ← copied verbatim from the original
└── tools/
    ├── __init__.py
    ├── search.py         ← Serper wrapper, copied verbatim
    └── scrape.py         ← Jina Reader wrapper, copied verbatim
```
