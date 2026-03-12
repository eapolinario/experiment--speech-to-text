"""NVIDIA NeMo Sortformer diarization backend.

Uses the Sortformer model which sorts speakers by arrival time.
Supports both offline and streaming variants.

Model: https://huggingface.co/nvidia/diar_sortformer_4spk-v1
Requires: pip install nemo_toolkit[asr]
"""

import numpy as np

from src.diarizers.base import DiarizationBackend, DiarizationSegment

OFFLINE_MODEL = "nvidia/diar_sortformer_4spk-v1"
STREAMING_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"


class NeMoSortformerBackend(DiarizationBackend):
    """Wraps NVIDIA NeMo Sortformer for offline diarization."""

    def __init__(self, streaming: bool = False) -> None:
        self._model = None
        self._streaming = streaming

    def load(self, device: str, hf_token: str | None = None) -> None:
        from nemo.collections.asr.models import SortformerEncLabelModel

        model_id = STREAMING_MODEL if self._streaming else OFFLINE_MODEL
        self._model = SortformerEncLabelModel.from_pretrained(model_id)
        if device == "cuda":
            self._model = self._model.cuda()

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[DiarizationSegment]:
        if self._model is None:
            raise RuntimeError("Call load() first")
        import tempfile
        import os
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            sf.write(tmp_path, audio, sample_rate)

            # Sortformer `diarize` already returns speaker turn segments; we just
            # wrap them into `DiarizationSegment` objects.
            preds = self._model.diarize(audio=[tmp_path], batch_size=1)
        finally:
            os.unlink(tmp_path)

        segments = []
        if preds and len(preds) > 0:
            for pred in preds:
                for turn in pred:
                    segments.append(
                        DiarizationSegment(
                            speaker=turn.speaker,
                            start=turn.start,
                            end=turn.end,
                        )
                    )

        return segments
