"""Minimal LangSmith tracing smoke test with a Doubleword model.

With LANGSMITH_TRACING=true and LANGSMITH_API_KEY set, this single call shows up as
a trace in LangSmith. Requires DOUBLEWORD_API_KEY too (see .env.example).

    uv run python trace.py
"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_doubleword import ChatDoubleword

MODEL = os.environ.get("APP_MODEL", "Qwen/Qwen3.5-9B")

llm = ChatDoubleword(model=MODEL)
print(llm.invoke("Explain bismuth in three sentences.").content)
print(f"\nTraced to LangSmith (project: {os.environ.get('LANGSMITH_PROJECT', 'default')}).")
