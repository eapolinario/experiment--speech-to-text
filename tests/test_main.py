from unittest.mock import MagicMock, patch

import numpy as np

from src.main import transcribe_segments


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
