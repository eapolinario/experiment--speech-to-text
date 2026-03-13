"""Backend registry — maps CLI names to backend classes."""

from src.diarizers.base import DiarizationBackend

BACKENDS: dict[str, tuple[str, str]] = {
    "pyannote": (
        "src.diarizers.pyannote_backend",
        "PyannoteBackend",
    ),
}


def get_backend(name: str) -> DiarizationBackend:
    """Lazily import and instantiate a backend by name."""
    if name not in BACKENDS:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown diarization backend '{name}'. Choose from: {available}")

    module_path, class_name = BACKENDS[name]
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    return cls()
