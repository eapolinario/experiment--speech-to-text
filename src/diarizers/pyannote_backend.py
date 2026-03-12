"""Pyannote speaker-diarization-3.1 backend (default)."""

import numpy as np
import torch
from pyannote.audio import Pipeline

from src.diarizers.base import DiarizationBackend, DiarizationSegment

MODEL_ID = "pyannote/speaker-diarization-3.1"


class PyannoteBackend(DiarizationBackend):
    """Wraps pyannote/speaker-diarization-3.1."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None

    def load(self, device: str, hf_token: str | None = None) -> None:
        self._pipeline = Pipeline.from_pretrained(MODEL_ID, token=hf_token)
        self._pipeline.to(torch.device(device))

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[DiarizationSegment]:
        if self._pipeline is None:
            raise RuntimeError("Call load() first")
        waveform = torch.tensor(audio).unsqueeze(0)
        result = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})

        segments = []
        for turn, _, speaker in result.speaker_diarization.itertracks(yield_label=True):
            segments.append(DiarizationSegment(speaker=speaker, start=turn.start, end=turn.end))
        return segments
