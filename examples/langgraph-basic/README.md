# langgraph-basic

A minimal sanity-check example. Two scripts run the same trivial LangGraph
workflow (model + calculator tool + conditional edge) against the two
`langchain-doubleword` chat models, printing wall time for each.

| Script        | Model                | Path                    | Concurrency      |
|---------------|----------------------|-------------------------|------------------|
| `realtime.py` | `ChatDoubleword`     | `/v1/chat/completions`  | sync, sequential |
| `batched.py`  | `ChatDoublewordBatch`| Doubleword batch API via `autobatcher` | async, `asyncio.gather` |

Each script runs:

1. **A single query** — `ChatDoubleword` is faster here because the
   `batch_window_seconds` of `ChatDoublewordBatch` is overhead with nothing
   to collate.
2. **Five queries** — sequential for `realtime.py`, concurrent via
   `asyncio.gather` for `batched.py`. The five concurrent calls to
   `ChatDoublewordBatch` get collated into a single autobatcher window and
   submitted as one batch.

The point isn't to benchmark one against the other — it's to confirm that
both wire into LangGraph identically (same `bind_tools`, same conditional
edge, same state machine).

## Running

```bash
export DOUBLEWORD_API_KEY="sk-..."   # or use ~/.dw/credentials.toml

cd examples/langgraph-basic
uv sync
uv run python realtime.py
uv run python batched.py
```

Edit `MODEL` at the top of either script to point at whichever model you
have access to. The default is the 30B Qwen model since this example
doesn't benefit from the larger 235B.
