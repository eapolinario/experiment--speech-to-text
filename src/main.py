import os

import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline

SAMPLE_RATE = 16_000
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
ASR_MODEL = "openai/whisper-large-v3-turbo"


def load_models(hf_token: str):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading models on {device} (first run will download weights)...")
    asr = hf_pipeline("automatic-speech-recognition", model=ASR_MODEL, device=device)
    diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
    diarizer.to(torch.device(device))
    return asr, diarizer


def record_until_enter() -> np.ndarray:
    frames = []

    def callback(indata, _frame_count, _time, _status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input("  [recording] Press Enter to stop...")

    return np.concatenate(frames).flatten()


def transcribe_segments(audio: np.ndarray, asr, diarizer) -> list[tuple[str, str]]:
    waveform = torch.tensor(audio).unsqueeze(0)  # [1, samples]
    diarization = diarizer({"waveform": waveform, "sample_rate": SAMPLE_RATE})

    results = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        start = int(turn.start * SAMPLE_RATE)
        end = int(turn.end * SAMPLE_RATE)
        segment = audio[start:end]

        if len(segment) < SAMPLE_RATE * 0.1:  # skip segments shorter than 100ms
            continue

        text = asr({"raw": segment, "sampling_rate": SAMPLE_RATE})["text"].strip()
        if text:
            results.append((speaker, text))

    return results


def main():
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set. Add it to your .env file.")

    asr, diarizer = load_models(hf_token)
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
        segments = transcribe_segments(audio, asr, diarizer)

        print()
        for speaker, text in segments:
            print(f"  {speaker}: {text}")
        print()


if __name__ == "__main__":
    main()
