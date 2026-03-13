default: run

run:
    PYTHONPATH=. uv run src/main.py --backend pyannote

test:
    uv run pytest

test-cov:
    uv run pytest --cov=src --cov-report=term-missing

sync:
    uv sync --all-groups
