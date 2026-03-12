import argparse
import os

import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline

from src.diarizers import BACKENDS, get_backend
from src.diarizers.base import DiarizationBackend

SAMPLE_RATE = 16_000
ASR_MODEL = "openai/whisper-large-v3-turbo"


def load_models(hf_token: str | None, backend_name: str):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading models on {device} (first run will download weights)...")

    backend = get_backend(backend_name)

    # WhisperX handles ASR internally, so skip loading a separate ASR model.
    if backend_name == "whisperx":
        asr = None
    else:
        asr = hf_pipeline("automatic-speech-recognition", model=ASR_MODEL, device=device)

    backend.load(device, hf_token)
    return asr, backend


def record_until_enter() -> np.ndarray:
    frames = []

    def callback(indata, _frame_count, _time, _status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input("  [recording] Press Enter to stop...")

    return np.concatenate(frames).flatten()


def transcribe_segments(
    audio: np.ndarray, asr, diarizer: DiarizationBackend, backend_name: str
) -> list[tuple[str, str]]:
    # WhisperX does ASR + diarization together.
    if backend_name == "whisperx":
        return diarizer.transcribe_with_speakers(audio, SAMPLE_RATE)

    segments = diarizer.diarize(audio, SAMPLE_RATE)

    results = []
    for seg in segments:
        start = int(seg.start * SAMPLE_RATE)
        end = int(seg.end * SAMPLE_RATE)
        chunk = audio[start:end]

        if len(chunk) < SAMPLE_RATE * 0.1:
            continue

        text = asr({"raw": chunk, "sampling_rate": SAMPLE_RATE})["text"].strip()
        if text:
            results.append((seg.speaker, text))

    return results


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Speech-to-text with speaker diarization")
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS.keys()),
        default="pyannote",
        help="Diarization backend to use (default: pyannote)",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    # Only pyannote and whisperx backends use Hugging Face gated models.
    _HF_TOKEN_REQUIRED_BACKENDS = {"pyannote", "whisperx"}
    if args.backend in _HF_TOKEN_REQUIRED_BACKENDS and not hf_token:
        raise RuntimeError(
            f"HF_TOKEN not set. The '{args.backend}' backend requires a HuggingFace token. "
            "Add it to your .env file."
        )

    print(f"Using diarization backend: {args.backend}")
    asr, diarizer = load_models(hf_token, args.backend)
    print("Ready. Press Enter to start recording. Ctrl-C or 'quit' to exit.\n")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ("quit", "exit"):
            break

        audio = record_until_enter()
        segments = transcribe_segments(audio, asr, diarizer, args.backend)

        print()
        for speaker, text in segments:
            print(f"  {speaker}: {text}")
        print()


if __name__ == "__main__":
    main()
