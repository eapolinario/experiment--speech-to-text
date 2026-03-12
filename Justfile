default: run

run backend="pyannote":
    #!/usr/bin/env bash
    if [ "{{backend}}" = "whisperx" ]; then
        nix develop .#whisperx --command env PYTHONPATH=. uv run --project backends/whisperx src/main.py --backend whisperx
    else
        PYTHONPATH=. uv run src/main.py --backend {{backend}}
    fi

test:
    uv run pytest

test-cov:
    uv run pytest --cov=src --cov-report=term-missing

sync:
    uv sync --all-groups

sync-whisperx:
    uv sync --project backends/whisperx

install extra:
    uv sync --extra {{extra}}
