import os
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.main import Diarizer, transcribe_segments


def test_transcribe_segments_filters_short_chunks():
    diarizer = MagicMock()
    diarizer.diarize.return_value = [
        MagicMock(speaker="SPEAKER_00", start=0.0, end=0.05),  # too short — filtered
        MagicMock(speaker="SPEAKER_01", start=1.0, end=2.0),
    ]

    asr = MagicMock(return_value={"text": "hello"})
    audio = np.zeros(32_000, dtype=np.float32)

    results = transcribe_segments(audio, asr, diarizer)

    assert results == [("SPEAKER_01", "hello")]
    assert asr.call_count == 1


def test_transcribe_segments_skips_empty_text():
    diarizer = MagicMock()
    diarizer.diarize.return_value = [
        MagicMock(speaker="SPEAKER_00", start=0.0, end=1.0),
    ]

    asr = MagicMock(return_value={"text": "  "})
    audio = np.zeros(32_000, dtype=np.float32)

    results = transcribe_segments(audio, asr, diarizer)

    assert results == []


def test_diarize_writes_wav_with_correct_params():
    """diarize() writes a mono 16-bit WAV at the given sample rate, then deletes it."""
    diarizer = Diarizer()
    diarizer._pipeline = MagicMock()

    annotation = MagicMock()
    annotation.itertracks.return_value = []

    captured = {}

    def fake_source(path, sample_rate):
        with wave.open(path, "rb") as wf:
            captured["channels"] = wf.getnchannels()
            captured["sampwidth"] = wf.getsampwidth()
            captured["framerate"] = wf.getframerate()
            captured["nframes"] = wf.getnframes()
        captured["path"] = path
        return MagicMock()

    with patch("src.main.FileAudioSource", side_effect=fake_source), \
         patch("src.main.StreamingInference") as mock_inf_cls:
        mock_inf_cls.return_value.return_value = annotation
        audio = np.zeros(3_200, dtype=np.float32)  # 0.2 s at 16 kHz
        result = diarizer.diarize(audio, 16_000)

    assert result == []
    assert captured["channels"] == 1
    assert captured["sampwidth"] == 2   # 16-bit
    assert captured["framerate"] == 16_000
    assert captured["nframes"] == 3_200
    # Temp file must be deleted after diarize() returns
    assert not os.path.exists(captured["path"])


def test_diarize_writes_wav_in_multiple_chunks():
    """diarize() writes all samples even when audio exceeds _WAV_CHUNK size."""
    diarizer = Diarizer()
    diarizer._pipeline = MagicMock()

    annotation = MagicMock()
    annotation.itertracks.return_value = []

    captured = {}

    def fake_source(path, sample_rate):
        with wave.open(path, "rb") as wf:
            captured["nframes"] = wf.getnframes()
        return MagicMock()

    # Use audio longer than _WAV_CHUNK (16_000) to exercise multi-chunk path
    n_samples = Diarizer._WAV_CHUNK * 3 + 500
    with patch("src.main.FileAudioSource", side_effect=fake_source), \
         patch("src.main.StreamingInference") as mock_inf_cls:
        mock_inf_cls.return_value.return_value = annotation
        audio = np.zeros(n_samples, dtype=np.float32)
        diarizer.diarize(audio, 16_000)

    assert captured["nframes"] == n_samples


def test_diarize_cleans_up_on_inference_error():
    """Temp WAV file is deleted even when inference() raises."""
    diarizer = Diarizer()
    diarizer._pipeline = MagicMock()

    captured_path = []

    def fake_source(path, sample_rate):
        captured_path.append(path)
        return MagicMock()

    with patch("src.main.FileAudioSource", side_effect=fake_source), \
         patch("src.main.StreamingInference") as mock_inf_cls:
        mock_inf_cls.return_value.side_effect = RuntimeError("inference failed")
        audio = np.zeros(3_200, dtype=np.float32)
        with pytest.raises(RuntimeError, match="inference failed"):
            diarizer.diarize(audio, 16_000)

    assert len(captured_path) == 1
    assert not os.path.exists(captured_path[0])
