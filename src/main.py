import numpy as np
import sounddevice as sd
from transformers import pipeline

MODEL_ID = "openai/whisper-base"
SAMPLE_RATE = 16_000


def load_model():
    print("Loading model (first run will download weights)...")
    return pipeline("automatic-speech-recognition", model=MODEL_ID)


def record_until_enter() -> np.ndarray:
    frames = []

    def callback(indata, _frame_count, _time, _status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input("  [recording] Press Enter to stop...")

    return np.concatenate(frames).flatten()


def transcribe(audio: np.ndarray, asr) -> str:
    result = asr({"raw": audio, "sampling_rate": SAMPLE_RATE})
    return result["text"].strip()


def main():
    asr = load_model()
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
        text = transcribe(audio, asr)
        print(f"  {text}\n")


if __name__ == "__main__":
    main()
