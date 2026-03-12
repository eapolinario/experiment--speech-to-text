"""DiariZen (BUT-FIT) diarization backend.

Uses the DiariZen pipeline from BUT-FIT which combines WavLM-based EEND
with clustering. Especially strong with 5+ speakers.

Model: https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2
Requires: pip install diarizen
"""

import numpy as np

from src.diarizers.base import DiarizationBackend, DiarizationSegment

MODEL_ID = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"


class DiariZenBackend(DiarizationBackend):
    """Wraps the DiariZen pipeline from BUT-FIT."""

    def __init__(self) -> None:
        self._pipeline = None

    def load(self, device: str, hf_token: str | None = None) -> None:
        from diarizen.pipelines.inference import DiariZenPipeline

        self._pipeline = DiariZenPipeline.from_pretrained(MODEL_ID)
        self._pipeline.to(device)

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[DiarizationSegment]:
        assert self._pipeline is not None, "Call load() first"
        import tempfile
        import soundfile as sf

        # DiariZen expects a file path, so write a temp WAV file.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            result = self._pipeline(f.name)

        segments = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            segments.append(DiarizationSegment(speaker=speaker, start=turn.start, end=turn.end))
        return segments
