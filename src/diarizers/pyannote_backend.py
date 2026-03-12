"""Pyannote speaker-diarization-3.1 backend (default)."""

import numpy as np
import torch
from pyannote.audio import Pipeline

from src.diarizers.base import DiarizationBackend, DiarizationSegment

MODEL_ID = "pyannote/speaker-diarization-3.1"
CHUNK_DURATION = 30  # seconds per chunk to stay within GPU memory


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

        chunk_samples = CHUNK_DURATION * sample_rate
        segments = []

        for chunk_start in range(0, len(audio), chunk_samples):
            chunk = audio[chunk_start : chunk_start + chunk_samples]
            waveform = torch.tensor(chunk).unsqueeze(0)
            result = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})

            offset = chunk_start / sample_rate
            for turn, _, speaker in result.speaker_diarization.itertracks(yield_label=True):
                segments.append(
                    DiarizationSegment(speaker=speaker, start=turn.start + offset, end=turn.end + offset)
                )

        return segments
