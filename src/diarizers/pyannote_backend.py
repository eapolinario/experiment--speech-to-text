"""Pyannote speaker-diarization-3.1 backend (default)."""

import numpy as np
import torch
from pyannote.audio import Pipeline

from src.diarizers.base import DiarizationBackend, DiarizationSegment
from src.diarizers.speaker_matching import match_speakers

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
        global_registry: list[tuple[str, np.ndarray]] = []
        segments = []

        for chunk_start in range(0, len(audio), chunk_samples):
            chunk = audio[chunk_start : chunk_start + chunk_samples]
            # Sliced chunks may be non-contiguous; ensure contiguous float32 for zero-copy transfer.
            chunk_np = np.ascontiguousarray(chunk, dtype=np.float32)
            waveform = torch.from_numpy(chunk_np).unsqueeze(0)
            result = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})

            annotation = result.speaker_diarization
            # Convert embeddings to CPU numpy arrays before label matching.
            chunk_embeddings: dict[str, np.ndarray] = {}
            for label, emb in zip(annotation.labels(), result.speaker_embeddings):
                if isinstance(emb, torch.Tensor):
                    emb_np = emb.detach().cpu().numpy()
                else:
                    emb_np = np.asarray(emb)
                chunk_embeddings[label] = emb_np
            label_map = match_speakers(chunk_embeddings, global_registry)

            offset = chunk_start / sample_rate
            for turn, _, local_speaker in annotation.itertracks(yield_label=True):
                segments.append(
                    DiarizationSegment(
                        speaker=label_map[local_speaker],
                        start=turn.start + offset,
                        end=turn.end + offset,
                    )
                )

        return segments
