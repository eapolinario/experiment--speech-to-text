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
        assert self._model is not None, "Call load() first"
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            temp_path = f.name

        # Sortformer returns per-frame speaker probabilities; we threshold and
        # convert to segments.
        import torch

        preds = self._model.diarize(audio=[temp_path], batch_size=1)

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
