"""Diarization backends for speaker identification."""

from src.diarizers.base import DiarizationBackend, DiarizationSegment
from src.diarizers.registry import BACKENDS, get_backend

__all__ = ["DiarizationBackend", "DiarizationSegment", "BACKENDS", "get_backend"]
