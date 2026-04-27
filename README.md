# langchain-doubleword

A LangChain integration package for [Doubleword](https://doubleword.ai).

This package wires Doubleword's OpenAI-compatible inference API
(`https://api.doubleword.ai/v1`) into LangChain and LangGraph as both **real-time**
chat / embedding models and **transparently-batched** variants powered by
[`autobatcher`](https://pypi.org/project/autobatcher/).

The batched variants are required to access models that Doubleword exposes
**only via the batch API**, and they cut cost on workloads that fan out
many concurrent calls — typically the case in LangGraph agents.

## Installation

```bash
pip install langchain-doubleword
```

## Authentication

Three resolution paths, in precedence order:

1. **Explicit constructor argument**:
   ```python
   ChatDoubleword(model="...", api_key="sk-...")
   ```
2. **Environment variable**:
   ```bash
   export DOUBLEWORD_API_KEY=sk-...
   ```
3. **`~/.dw/credentials.toml`** — the same file written by Doubleword's CLI
   tooling. The active account is selected by `~/.dw/config.toml`'s
   `active_account` field, and `inference_key` from that account is used.

   ```toml
   # ~/.dw/config.toml
   active_account = "work"
   ```
   ```toml
   # ~/.dw/credentials.toml
   [accounts.work]
   inference_key = "sk-..."
   ```

   To use a non-active account from your credentials file, set
   `DOUBLEWORD_API_KEY` directly to that account's `inference_key` — there
   is no `account=` selector on the model itself.

## Chat models

### `ChatDoubleword` (real-time)

Drop-in chat model. Use this in any LangChain or LangGraph workflow that
expects a `BaseChatModel`.

```python
from langchain_doubleword import ChatDoubleword

llm = ChatDoubleword(model="your-model-name")

response = llm.invoke("Explain bismuth in three sentences.")
print(response.content)
```

### `ChatDoublewordBatch` (transparently batched)

Same interface, but every concurrent `.ainvoke()` call is collected by
`autobatcher` and submitted via Doubleword's batch endpoint. **Async-only** —
sync `.invoke()` raises.

Use this when:

- The model you want is **batch-only** (some Doubleword-hosted models do not
  expose a real-time chat endpoint).
- You're running a LangGraph workflow with parallel branches and want
  ~50% cost savings via batch pricing.

```python
import asyncio
from langchain_doubleword import ChatDoublewordBatch

llm = ChatDoublewordBatch(model="batch-only-model")

async def main():
    # Concurrent calls collected into a single batch under the hood.
    results = await asyncio.gather(*[
        llm.ainvoke(f"Summarize chapter {i}") for i in range(50)
    ])
    for r in results:
        print(r.content)

asyncio.run(main())
```

#### Tuning autobatcher

Four `autobatcher.BatchOpenAI` knobs are exposed as constructor arguments:

| Argument                | Default | Purpose                                                              |
|-------------------------|---------|----------------------------------------------------------------------|
| `batch_size`            | `1000`  | Submit a batch when this many requests are queued.                   |
| `batch_window_seconds`  | `10.0`  | Submit a batch after this many seconds even if the size cap is not reached. |
| `poll_interval_seconds` | `5.0`   | How often autobatcher polls for batch completion.                    |
| `completion_window`     | `"24h"` | Doubleword batch completion window. `"1h"` is more expensive but faster. |

```python
llm = ChatDoublewordBatch(
    model="your-model",
    batch_size=250,           # smaller batches for fast-turnaround LangGraph nodes
    batch_window_seconds=2.5, # don't make latency-sensitive calls wait 10s
    completion_window="1h",   # pay more, finish quicker
)
```

The same arguments are available on `DoublewordEmbeddingsBatch`.

### `ChatDoublewordAsync` (1-hour flex tier)

A thin subclass of `ChatDoublewordBatch` pinned to Doubleword's **flex
(1-hour)** completion window. Backed by `autobatcher.AsyncOpenAI` rather
than `BatchOpenAI`. Use this when 24-hour batch turnaround is too slow but
realtime cost is too high — typical for fan-out workflows that need results
within minutes-to-an-hour.

```python
import asyncio
from langchain_doubleword import ChatDoublewordAsync

llm = ChatDoublewordAsync(model="your-model")  # completion_window="1h" by default

async def main():
    results = await asyncio.gather(*[
        llm.ainvoke(f"Summarize chapter {i}") for i in range(50)
    ])
    for r in results:
        print(r.content)

asyncio.run(main())
```

All the autobatcher tuning knobs above apply unchanged. The only difference
from `ChatDoublewordBatch` is the default `completion_window` (`"1h"` vs
`"24h"`); the same `DoublewordEmbeddingsAsync` exists on the embeddings side.

## Embeddings

```python
from langchain_doubleword import (
    DoublewordEmbeddings,
    DoublewordEmbeddingsAsync,
    DoublewordEmbeddingsBatch,
)

embed = DoublewordEmbeddings(model="your-embedding-model")
vec = embed.embed_query("hello world")

# Or, transparently batched (24h tier):
batch_embed = DoublewordEmbeddingsBatch(model="your-embedding-model")
# vecs = await batch_embed.aembed_documents([...])

# Or on the 1h flex tier:
async_embed = DoublewordEmbeddingsAsync(model="your-embedding-model")
# vecs = await async_embed.aembed_documents([...])
```

## Use with LangGraph

`ChatDoubleword`, `ChatDoublewordBatch`, and `ChatDoublewordAsync` are all standard `BaseChatModel`
implementations, so they slot into any LangGraph node:

```python
from langgraph.graph import StateGraph, END
from langchain_doubleword import ChatDoublewordBatch

llm = ChatDoublewordBatch(model="your-model")

async def call_model(state):
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(dict)
graph.add_node("model", call_model)
graph.set_entry_point("model")
graph.add_edge("model", END)
app = graph.compile()
```

When several `model` nodes execute in parallel (e.g. via `Send` or fan-out
edges), `autobatcher` collects their requests into a single batch.

## Configuration

| Argument    | Env var              | Default                          |
|-------------|----------------------|----------------------------------|
| `api_key`   | `DOUBLEWORD_API_KEY` | _required_                       |
| `base_url`  | `DOUBLEWORD_API_BASE`| `https://api.doubleword.ai/v1`   |
| `model`     | —                    | _required_                       |

All other arguments accepted by `langchain_openai.ChatOpenAI` are forwarded
unchanged (`temperature`, `max_tokens`, `model_kwargs`, `timeout`, etc.).

## License

MIT
