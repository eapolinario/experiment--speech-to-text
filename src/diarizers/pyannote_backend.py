"""Pyannote speaker-diarization-community-1 backend (default)."""

import numpy as np
import torch
from pyannote.audio import Pipeline

from src.diarizers.base import DiarizationBackend, DiarizationSegment

MODEL_ID = "pyannote/speaker-diarization-community-1"


class PyannoteBackend(DiarizationBackend):
    """Wraps pyannote/speaker-diarization-community-1.

    The full audio is passed to the pipeline in one call so that pyannote's
    internal global clustering produces coherent speaker labels. Manual
    chunking is intentionally avoided: splitting audio before the clustering
    step forces independent per-chunk label assignment and undermines speaker
    consistency across the recording.

    For very long recordings that cause GPU OOM, pass device="cpu" rather than
    re-introducing chunking.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None

    def load(self, device: str, hf_token: str | None = None) -> None:
        self._pipeline = Pipeline.from_pretrained(MODEL_ID, token=hf_token)
        self._pipeline.to(torch.device(device))

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[DiarizationSegment]:
        if self._pipeline is None:
            raise RuntimeError("Call load() first")

        audio_np = np.ascontiguousarray(audio, dtype=np.float32)
        waveform = torch.from_numpy(audio_np).unsqueeze(0)
        result = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})

        return [
            DiarizationSegment(speaker=speaker, start=turn.start, end=turn.end)
            for turn, _, speaker in result.speaker_diarization.itertracks(yield_label=True)
        ]
