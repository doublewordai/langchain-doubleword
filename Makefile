.PHONY: help install test tests integration_test integration_tests \
        lint format typecheck check clean

help:
	@echo "Available targets:"
	@echo "  install         - install package + all dependency groups"
	@echo "  test            - run unit tests (no network)"
	@echo "  integration_test- run integration tests (requires DOUBLEWORD_API_KEY)"
	@echo "  lint            - run ruff lint + format check"
	@echo "  format          - run ruff format + autofix"
	@echo "  typecheck       - run mypy"
	@echo "  check           - run lint + typecheck + unit tests"
	@echo "  clean           - remove build / cache artifacts"

install:
	uv sync --all-groups

test tests:
	uv run --group test pytest --disable-socket --allow-unix-socket tests/unit_tests/

integration_test integration_tests:
	uv run --group test --group test_integration pytest -n auto tests/integration_tests/

lint:
	uv run --group lint ruff check langchain_doubleword tests
	uv run --group lint ruff format --check langchain_doubleword tests

format:
	uv run --group lint ruff format langchain_doubleword tests
	uv run --group lint ruff check --fix langchain_doubleword tests

typecheck:
	uv run --group typing mypy langchain_doubleword

check: lint typecheck test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
