default: run

run:
    uv run src/main.py

test:
    uv run pytest

test-cov:
    uv run pytest --cov=src --cov-report=term-missing

sync:
    uv sync --all-groups
