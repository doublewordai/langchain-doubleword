# LangSmith

[LangSmith](https://smith.langchain.com) is LangChain's cloud platform for tracing and evaluating
LLM apps. The `langchain-doubleword` chat models are standard LangChain models, so Doubleword slots
straight into LangSmith tracing and evals — and the batched variants let you run large evals at
Doubleword's batch price.

This guide connects a Doubleword model to LangSmith cloud and traces a call. It takes a few minutes.

## Step 1 — Sign up for LangSmith

Create an account at [smith.langchain.com](https://smith.langchain.com). Pick a data region (US, EU,
or APAC — this can't be changed later), then sign up with Google, GitHub, or email.

![LangSmith sign-up](images/01_signup.jpeg)

## Step 2 — Choose the code-first experience

LangSmith offers a code-first mode and a no-code mode (Fleet). For SDK tracing and evals with
`langchain-doubleword`, choose **LangSmith**.

![Choose the LangSmith code-first experience](images/02_choose_langsmith_mode.jpeg)

## Step 3 — Create an API key

Go to **Settings → API Keys** and click **+ API Key**. A Personal Access Token is fine for local
use (choose a Service Key for CI). Name it, set an expiry, and click **Create API Key**.

![Create an API key](images/03_create_api_key.jpeg)

The key is shown only once — copy it now. It starts with `lsv2_`.

![API key created](images/04_api_key_created.jpeg)

## Step 4 — Install

```bash
pip install langchain-doubleword langsmith
```

## Step 5 — Authenticate

```bash
export DOUBLEWORD_API_KEY="sk-..."          # app.doubleword.ai → API Keys
export LANGSMITH_API_KEY="lsv2_..."         # the key from Step 3
export LANGSMITH_TRACING="true"
export LANGSMITH_PROJECT="doubleword-langsmith"
# Regional endpoint, if you picked EU/APAC in Step 1:
# export LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com"
```

## Step 6 — Trace a Doubleword model

With tracing on, every call is recorded in LangSmith under `LANGSMITH_PROJECT`:

```python
from langchain_doubleword import ChatDoubleword

llm = ChatDoubleword(model="Qwen/Qwen3.5-9B")
print(llm.invoke("Explain bismuth in three sentences.").content)
```

Open the **Tracing** tab in LangSmith and you'll see the run, with inputs, outputs, latency, and
token counts. That's the integration — any `langchain-doubleword` model now traces to LangSmith.

## Run evals at scale

Once tracing works, the same models drop into LangSmith's `evaluate` / `aevaluate` runs. To keep
evals cheap, use the batched chat model (`ChatDoublewordBatch`) and run with concurrency so the
calls collate into batches at Doubleword's batch price.

The most common eval is the one you run every day: an LLM-as-judge scoring your app's outputs so you
catch prompt or model regressions before they ship. A complete runnable example — judge an app
against reference answers on the batch tier, then re-run after a change to see the score move — lives
in [`examples/langsmith`](https://github.com/doublewordai/langchain-doubleword/tree/master/examples/langsmith).

## Cost visibility

LangSmith shows traces, tokens, and feedback scores. For the authoritative batch spend, use
Doubleword: the console at [app.doubleword.ai/batches](https://app.doubleword.ai/batches), or
`dw batches analytics`.

## Further reading

- Doubleword inference API: https://docs.doubleword.ai/inference-api/intro-to-doubleword-inference
- `langchain-doubleword`: https://github.com/doublewordai/langchain-doubleword
- LangSmith tracing: https://docs.langchain.com/langsmith/observability
- LangSmith evaluation: https://docs.langchain.com/langsmith/evaluation
