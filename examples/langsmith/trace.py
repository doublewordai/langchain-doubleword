"""Minimal LangSmith tracing smoke test with a Doubleword model.

With LANGSMITH_TRACING=true and LANGSMITH_API_KEY set, this single call shows up as
a trace in LangSmith. Requires DOUBLEWORD_API_KEY too (see .env.example).

    uv run python trace.py
"""
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_doubleword import ChatDoubleword
from langsmith import Client
from langsmith.utils import LangSmithError, tracing_is_enabled

MODEL = os.environ.get("APP_MODEL", "openai/gpt-oss-20b")

llm = ChatDoubleword(model=MODEL)
print(llm.invoke("Explain bismuth in three sentences.").content)
project = os.environ.get("LANGSMITH_PROJECT", "default")
if not tracing_is_enabled():
    print("\nLangSmith tracing is off. Set LANGSMITH_TRACING=true.")
else:
    wait_for_all_tracers()
    try:
        next(Client().list_projects(limit=1), None)
        print(f"\nTraced to LangSmith (project: {project}).")
    except LangSmithError as e:
        print(f"\nLangSmith export failed: {e}")
