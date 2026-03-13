import os
import warnings
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline

SAMPLE_RATE = 16_000
ASR_MODEL = "openai/whisper-large-v3-turbo"
DIARIZER_MODEL = "pyannote/speaker-diarization-community-1"


@dataclass
class DiarizationSegment:
    """A single speaker turn with start/end times."""

    speaker: str
    start: float
    end: float


class Diarizer:
    """Wraps pyannote/speaker-diarization-community-1.

    Streaming is not possible: the model uses VBx clustering (AHC followed by
    iterative Variational Bayes over all speaker embeddings), which is a batch
    algorithm that requires the complete embedding matrix before it can assign
    any speaker labels. The full audio must therefore be collected before
    diarization starts.

    Neural network inference (segmentation and embedding) does use a sliding
    window internally, so GPU/CPU compute is bounded regardless of recording
    length. Memory is not, however: the full audio waveform stays in CPU RAM
    throughout (~230 MB/hour at 16 kHz float32, on top of ~4 GB of model
    weights). For GPU VRAM OOM pass device="cpu" to load_models(); for CPU RAM
    OOM the only options are shorter recordings or more RAM.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None

    def load(self, device: str, hf_token: str | None = None) -> None:
        self._pipeline = Pipeline.from_pretrained(DIARIZER_MODEL, token=hf_token)
        self._pipeline.to(torch.device(device))

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[DiarizationSegment]:
        if self._pipeline is None:
            raise RuntimeError("Call load() first")

        audio_np = np.ascontiguousarray(audio, dtype=np.float32)
        waveform = torch.from_numpy(audio_np).unsqueeze(0)
        result = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})

        return [
            DiarizationSegment(speaker=speaker, start=turn.start, end=turn.end)
            for turn, _, speaker in result.speaker_diarization.itertracks(yield_label=True)
        ]


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
    diarizer = Diarizer()
    diarizer.load(device, hf_token)
    return asr, diarizer


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
    audio: np.ndarray, asr, diarizer: Diarizer
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
