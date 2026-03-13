import os
import warnings

import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline

from src.diarizers import get_backend
from src.diarizers.base import DiarizationBackend

SAMPLE_RATE = 16_000
ASR_MODEL = "openai/whisper-large-v3-turbo"


def load_models(hf_token: str | None):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Loading models on {device} (first run will download weights)...")

    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=ASR_MODEL,
        device=device,
        generate_kwargs={"language": "en", "task": "transcribe"},
    )
    backend = get_backend("pyannote")
    backend.load(device, hf_token)
    return asr, backend


def record_until_enter() -> np.ndarray:
    frames = []
    device_rate = int(sd.query_devices(kind="input")["default_samplerate"])

    def callback(indata, _frame_count, _time, _status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=device_rate, channels=1, dtype="float32", callback=callback):
        input("  [recording] Press Enter to stop...")

    audio = np.concatenate(frames).flatten()

    if device_rate != SAMPLE_RATE:
        # Resample using torch to avoid depending on torchaudio as a transitive dependency.
        audio_tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, T)
        orig_len = audio_tensor.shape[-1]
        new_len = int(round(orig_len * SAMPLE_RATE / device_rate))
        audio_resampled = torch.nn.functional.interpolate(
            audio_tensor, size=new_len, mode="linear", align_corners=False
        )
        audio = audio_resampled.squeeze().numpy()
    return audio


def transcribe_segments(
    audio: np.ndarray, asr, diarizer: DiarizationBackend
) -> list[tuple[str, str]]:
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


def main():
    # Suppress noisy third-party warnings that we cannot fix upstream.
    warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\..*")
    warnings.filterwarnings("ignore", message=r".*return_token_timestamps.*")
    warnings.filterwarnings("ignore", message=r".*forced_decoder_ids.*")
    warnings.filterwarnings("ignore", message=r".*multilingual Whisper.*")
    warnings.filterwarnings("ignore", message=r".*Mean of empty slice.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=r".*invalid value encountered in divide.*", category=RuntimeWarning)

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. pyannote requires a HuggingFace token. "
            "Add it to your .env file."
        )

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
