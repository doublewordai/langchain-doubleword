# Catch Prompt and Model Regressions Before They Ship with LangSmith and Doubleword

Every prompt tweak or model swap can make your app worse without you realising until unexpected outputs appear in production. 

One 'cheap' insurance is an LLM-as-judge eval run on every change, but running a frontier judge
over thousands of traces in real time is slow, rate-limited, and pricey. Doubleword's batch tier takes that excuse away by enabling you to grade every trace offline for cents, offering confidence in your application, at scale, and cost effectively. 

We run **LLM-as-judge** regression evals in [LangSmith](https://smith.langchain.com). One app answers a
set of questions, and then a stronger model grades every answer, and you re-run on each prompt or
model change to catch quality regressions before they ship — all on Doubleword's async or batch tier.

`eval.py` answers [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) questions with
the app model, scores each answer against the dataset's reference answer on three axes (relevance,
truthfulness, tone) with an LLM-as-judge, and records those scores as LangSmith feedback on one
experiment. Run it again after a change and compare the experiments — if the scores drop, you've
caught a regression.

To show that in action the example ships two variants: `baseline` (a healthy prompt) and `regressed`
(a deliberately worse one). The regressed run scores lower and the two prompts are the same but differ by one instruction set:

- **baseline** — "Answer the question truthfully and concisely. If you are unsure, say so rather than guessing."
- **regressed** — "You are a confident, entertaining assistant. Always give a definitive, elaborate answer… Never admit uncertainty and never refuse."

On a 50-example run over the same questions:

| Prompt | relevance | truthfulness | tone | overall pass |
|------|------|------|------|------|
| baseline | 0.97 | 0.75 | 0.92 | 76% |
| regressed | 0.87 | 0.38 | 0.55 | 34% |

The degraded prompt nearly halves truthfulness (0.75 → 0.38) and drops the pass rate from 76% to
34%. This test can catch regression before it ships.

| Step | Model | How |
|------|-------|-----|
| Answer | `APP_MODEL` (default gpt-oss-20b) | the app under test answers each question |
| Judge | `JUDGE_MODEL` (default DeepSeek-V4-Pro) | score relevance / truthfulness / tone vs reference |
| Record | LangSmith `aevaluate` | answer + four feedback scores per example, in one experiment |

Generation and judging each run as a single `asyncio.gather` pass, so autobatcher collates the calls
into one batch per stage. A final `aevaluate` records each answer and its scores — its target and
evaluator are pure lookups of that work, so it adds no model calls.

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

![A LangSmith experiment showing the four judge scores averaged across the run](images/experiment_summary.png)

*A 500-example baseline run in LangSmith: the four judge scores (overall, relevance, tone,
truthfulness) averaged across the dataset, with every per-example trace one click away. Re-run after a
change and the bars move.*

For your own app, replace the `--variant` prompts with your real prompt and model, point the dataset
at your own cases, and re-run on every change. LangSmith tracks the scores across experiments, so a
drop is obvious.

Notes:

- Set `APP_MODEL` and `JUDGE_MODEL` to grade whatever you like.
- Pick the tier with `--tier`: `batch` (24h, cheapest, the default), `async` (1-hour flex), or
  `realtime`. The same eval runs on all three; batch and async wait for completion (usually minutes),
  realtime returns immediately.
- `MAX_TOKENS` (default 2048) caps each call so a reasoning model can't run away.
- `trace.py` is a one-call tracing smoke test.

## What it costs

Measured on the 500-example baseline run pictured above — `gpt-oss-20b` answering, `DeepSeek-V4-Pro`
judging — read straight from `dw batches analytics`:

| Stage | Model | Input tokens | Output tokens | Cost |
|------|------|------|------|------|
| Generate | gpt-oss-20b | 55,548 | 297,513 | $0.061 |
| Judge | DeepSeek-V4-Pro | 155,954 | 63,832 | $0.380 |
| **Total (500 evals)** | | 211,502 | 361,345 | **$0.441** |

In production the answers already exist, so the eval you actually re-run on every change is the judge.
Grading one trace costs about **$0.00076**. Scaled up, and priced against the same token volume run on
a frontier model:

**Judge only** — the recurring regression-eval cost:

| Traces judged | Doubleword | GPT-5.5 | Claude Opus 4.8 |
|------|------|------|------|
| 1,000 | $0.76 | $5.39 | $4.75 |
| 100,000 | $76 | $539 | $475 |
| 1,000,000 | $760 | $5,389 (7.1×) | $4,751 (6.3×) |

**Whole run** — generate plus judge:

| Evals | Doubleword | GPT-5.5 | Claude Opus 4.8 |
|------|------|------|------|
| 1,000 | $0.88 | $23.80 | $20.18 |
| 100,000 | $88 | $2,380 | $2,018 |
| 1,000,000 | $882 | $23,796 (27×) | $20,182 (23×) |

The frontier columns price the identical input/output token volume at published rates — GPT-5.5 at
$5/$30 per million tokens, Claude Opus 4.8 at $5/$25. These figures are from the 1-hour async tier;
the 24-hour batch tier is cheaper still.

## Where the numbers come from

Two systems hold the results, and they answer different questions.

**Eval quality — LangSmith.** Each run creates an experiment. Open it for:

- Per-example traces: the question, the app's answer, and the judge's relevance / truthfulness / tone
  / overall scores, with the judge's rationale stored as the feedback comment.
- Summary statistics: the average of each feedback key across the dataset, shown at the top of the
  experiment and charted across experiments — so a regression shows up as a dropped average from one
  run to the next.

The script also prints those averages and the overall pass count when it finishes, as a quick check.

**Cost and tokens — Doubleword.** LangSmith shows token counts per trace, but the authoritative spend
lives with Doubleword:

- The console at https://app.doubleword.ai/batches lists every batch with its status, token counts,
  and cost.
- `dw batches analytics <batch_id>` reports input/output tokens and the exact cost for a batch. A run
  produces two kinds of batch — generation (the app model) and judging (the judge model) — so sum
  them for the run's total.

The script prints generation and judge token totals at the end too, but treat the Doubleword console
/ `dw batches analytics` as the source of truth for cost.
