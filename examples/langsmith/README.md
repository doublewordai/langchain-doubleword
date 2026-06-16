# langsmith

Run an  regression evaluation with [LLM-as-judge] in [LangSmith](https://smith.langchain.com). One app answers a
set of questions, a stronger Doubleword model grades every answer, and you re-run on each prompt or
model change to catch quality regressions before they ship — all on the batch tier.

`eval.py` answers [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) questions with
the app model, scores each answer against the dataset's reference answer on three axes (relevance,
truthfulness, tone) with an LLM-as-judge, and records those scores as LangSmith feedback on one
experiment. Run it again after a change and compare the experiments — if the scores drop, you've
caught a regression.

To show that in action the example ships two variants: `baseline` (a healthy prompt) and `regressed`
(a deliberately worse one). The regressed run scores lower — the kind of regression you'd want to
catch before it reaches production.

| Step | Model | How |
|------|-------|-----|
| Answer | `APP_MODEL` (default Qwen3.5-9B) | `ChatDoublewordBatch` → one experiment |
| Judge | `JUDGE_MODEL` (default DeepSeek-V4-Pro) | score relevance / truthfulness / tone vs reference |
| Record | LangSmith `aevaluate(experiment, ...)` | four feedback scores per example |

The app answers via `aevaluate(max_concurrency=N)`, so autobatcher collates the calls into a batch.
The judge then scores every answer in one batched pass, and the LangSmith evaluator is a pure lookup
of those verdicts — so the judge batches too, with no per-evaluator model calls.

## Running

```bash
export DOUBLEWORD_API_KEY="sk-..."        # or ~/.dw/credentials.toml
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_TRACING="true"
# (or copy .env.example to .env and fill it in)

cd examples/langsmith
uv sync
uv run python eval.py --variant baseline -n 50
uv run python eval.py --variant regressed -n 50
```

Each run prints the average scores and writes an experiment to LangSmith with per-example
relevance / truthfulness / tone / overall feedback. Open the two experiments side by side: the
`regressed` one scores lower across the board — the regression, caught before production. The batches
show up in the Doubleword console at https://app.doubleword.ai/batches, and `dw batches analytics`
reports the authoritative spend.

For your own app, replace the `--variant` prompts with your real prompt and model, point the dataset
at your own cases, and re-run on every change. LangSmith tracks the scores across experiments, so a
drop is obvious.

Notes:

- Set `APP_MODEL` and `JUDGE_MODEL` to grade whatever you like.
- The batch tier trades latency for cost — even a small run waits for batch completion (usually
  minutes). Swap `ChatDoublewordBatch` for `ChatDoublewordAsync` (1-hour flex tier) for a snappier
  run, or `ChatDoubleword` for real-time.
- `trace.py` is a one-call tracing smoke test.
