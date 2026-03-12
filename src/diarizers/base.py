"""Base class and types for diarization backends."""

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class DiarizationSegment:
    """A single speaker turn with start/end times."""

    speaker: str
    start: float
    end: float


class DiarizationBackend(ABC):
    """Interface that all diarization backends implement."""

    @abstractmethod
    def load(self, device: str, hf_token: str | None = None) -> None:
        """Download weights and initialise the pipeline on *device*."""

    @abstractmethod
    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[DiarizationSegment]:
        """Return a list of speaker segments for the given audio."""

    def transcribe_with_speakers(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[tuple[str, str]]:
        """Return (speaker, text) pairs for backends that combine ASR + diarization.

        The default implementation raises :exc:`NotImplementedError`. Override in
        backends (e.g. WhisperX) that handle transcription internally.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support transcribe_with_speakers(). "
            "Use diarize() instead."
        )
