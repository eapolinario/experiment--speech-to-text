default: run

run backend="pyannote":
    PYTHONPATH=. uv run src/main.py --backend {{backend}}

test:
    uv run pytest

test-cov:
    uv run pytest --cov=src --cov-report=term-missing

sync:
    uv sync --all-groups
