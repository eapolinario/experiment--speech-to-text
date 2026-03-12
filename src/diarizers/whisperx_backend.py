"""WhisperX backend — combined ASR + diarization with word-level speaker assignment.

Unlike other backends, WhisperX handles both transcription and diarization in a
single pipeline, producing word-level speaker labels via forced alignment.

GitHub: https://github.com/m-bain/whisperX
Requires: pip install whisperx
"""

import numpy as np

from src.diarizers.base import DiarizationBackend, DiarizationSegment


class WhisperXBackend(DiarizationBackend):
    """WhisperX: Whisper + pyannote diarization + forced alignment."""

    def __init__(self, asr_model: str = "large-v3-turbo") -> None:
        self._asr_model_name = asr_model
        self._model = None
        self._device = "cpu"
        self._hf_token: str | None = None

    def load(self, device: str, hf_token: str | None = None) -> None:
        import whisperx

        self._device = device
        self._hf_token = hf_token

        # Use lower precision on CUDA for better performance; float32 elsewhere.
        if "cuda" in device.lower():
            compute_type = "float16"
        else:
            compute_type = "float32"

        self._model = whisperx.load_model(
            self._asr_model_name, device=device, compute_type=compute_type
        )

    def _run_pipeline(self, audio: np.ndarray) -> dict:
        """Run the full WhisperX pipeline and return the result with speaker labels."""
        import whisperx

        if self._model is None:
            raise RuntimeError("Call load() first")

        # WhisperX expects a float32 numpy array.
        audio = np.asarray(audio, dtype=np.float32)

        result = self._model.transcribe(audio, batch_size=16)

        # Align for word-level timestamps.
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self._device
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, self._device
        )

        # Run diarization and assign speakers to words.
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self._hf_token, device=self._device
        )
        diarize_segments = diarize_model(audio)
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
