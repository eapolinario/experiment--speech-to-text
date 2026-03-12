"""Pyannote speaker-diarization-3.1 backend (default)."""

import numpy as np
import torch
from pyannote.audio import Pipeline

from src.diarizers.base import DiarizationBackend, DiarizationSegment

MODEL_ID = "pyannote/speaker-diarization-3.1"
CHUNK_DURATION = 30  # seconds per chunk to stay within GPU memory
SIMILARITY_THRESHOLD = 0.75


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _match_speakers(
    chunk_embeddings: dict[str, np.ndarray],
    global_registry: list[tuple[str, np.ndarray]],
) -> dict[str, str]:
    """Map local chunk speaker labels to globally consistent labels.

    Compares each chunk speaker's embedding against the global registry via
    cosine similarity. Reuses an existing label when similarity exceeds
    SIMILARITY_THRESHOLD; otherwise mints a new one.
    """
    mapping: dict[str, str] = {}
    for local_label, emb in chunk_embeddings.items():
        best_score, best_label = max(
            ((_cosine_similarity(emb, g_emb), g_label) for g_label, g_emb in global_registry),
            default=(0.0, None),
        )
        if best_score >= SIMILARITY_THRESHOLD and best_label is not None:
            mapping[local_label] = best_label
        else:
            new_label = f"SPEAKER_{len(global_registry):02d}"
            global_registry.append((new_label, emb))
            mapping[local_label] = new_label
    return mapping


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
            waveform = torch.tensor(chunk).unsqueeze(0)
            result = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})

            annotation = result.speaker_diarization
            # speaker_embeddings rows are ordered to match annotation.labels()
            chunk_embeddings = dict(zip(annotation.labels(), result.speaker_embeddings))
            label_map = _match_speakers(chunk_embeddings, global_registry)

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
