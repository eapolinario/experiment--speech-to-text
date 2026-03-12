"""WhisperX backend — combined ASR + diarization with word-level speaker assignment.

Unlike other backends, WhisperX handles both transcription and diarization in a
single pipeline, producing word-level speaker labels via forced alignment.

GitHub: https://github.com/m-bain/whisperX
Requires: pip install whisperx
"""

import numpy as np
import pandas as pd

from src.diarizers.base import DiarizationBackend, DiarizationSegment
from src.diarizers.speaker_matching import match_speakers

CHUNK_DURATION = 30  # seconds per diarization chunk to stay within GPU memory


class WhisperXBackend(DiarizationBackend):
    """WhisperX: Whisper + pyannote diarization + forced alignment."""

    def __init__(self, asr_model: str = "large-v3-turbo") -> None:
        self._asr_model_name = asr_model
        self._model = None
        self._device = "cpu"
        self._hf_token: str | None = None

    def load(self, device: str, hf_token: str | None = None) -> None:
        import whisperx

        if device == "mps":
            raise RuntimeError("WhisperX is not supported on MPS. Use a CUDA device or CPU.")

        self._device = device
        self._hf_token = hf_token

        # int8_float16 uses significantly less VRAM than float16 on CUDA.
        if "cuda" in device.lower():
            compute_type = "int8_float16"
        else:
            compute_type = "float32"

        self._model = whisperx.load_model(
            self._asr_model_name, device=device, compute_type=compute_type
        )

    def _diarize_chunked(self, audio: np.ndarray, diarize_model) -> pd.DataFrame:
        """Run diarization in 30s chunks and reconcile speaker labels across chunks."""
        chunk_samples = CHUNK_DURATION * 16_000  # whisperx always uses 16kHz
        global_registry: list[tuple[str, np.ndarray]] = []
        frames = []

        for chunk_start in range(0, len(audio), chunk_samples):
            chunk = audio[chunk_start : chunk_start + chunk_samples]
            df, embeddings = diarize_model(chunk, return_embeddings=True)

            if embeddings:
                label_map = match_speakers(
                    {spk: np.array(emb) for spk, emb in embeddings.items()},
                    global_registry,
                )
                df["speaker"] = df["speaker"].map(label_map)

            offset = chunk_start / 16_000
            df["start"] += offset
            df["end"] += offset
            frames.append(df)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _run_pipeline(self, audio: np.ndarray) -> dict:
        """Run the full WhisperX pipeline and return the result with speaker labels."""
        import whisperx

        if self._model is None:
            raise RuntimeError("Call load() first")

        import gc
        import torch

        # WhisperX expects a float32 numpy array.
        audio = np.asarray(audio, dtype=np.float32)

        result = self._model.transcribe(audio, batch_size=4)

        # Free GPU memory from transcription before loading the alignment model.
        torch.cuda.empty_cache()

        # Align for word-level timestamps.
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self._device
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, self._device
        )
        del align_model
        gc.collect()
        torch.cuda.empty_cache()

        # Run diarization in chunks and assign speakers to words.
        diarize_model = whisperx.diarize.DiarizationPipeline(
            token=self._hf_token, device=self._device
        )
        diarize_segments = self._diarize_chunked(audio, diarize_model)
        del diarize_model
        gc.collect()
        torch.cuda.empty_cache()

        return whisperx.assign_word_speakers(diarize_segments, result)

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[DiarizationSegment]:
        result = self._run_pipeline(audio)

        segments = []
        for seg in result["segments"]:
            speaker = seg.get("speaker", "UNKNOWN")
            segments.append(
                DiarizationSegment(speaker=speaker, start=seg["start"], end=seg["end"])
            )
        return segments

    def transcribe_with_speakers(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[tuple[str, str]]:
        """Full WhisperX pipeline: returns (speaker, text) pairs directly.

        This skips the separate ASR step since WhisperX already transcribes.
        """
        result = self._run_pipeline(audio)

        pairs = []
        for seg in result["segments"]:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "").strip()
            if text:
                pairs.append((speaker, text))
        return pairs
