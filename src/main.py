import contextlib
import os
import tempfile
import warnings
import wave
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.sources import FileAudioSource
from transformers import pipeline as hf_pipeline

SAMPLE_RATE = 16_000
ASR_MODEL = "openai/whisper-large-v3-turbo"


@dataclass
class DiarizationSegment:
    """A single speaker turn with start/end times."""

    speaker: str
    start: float
    end: float


class Diarizer:
    """Wraps diart's SpeakerDiarization for offline recordings.

    Streaming is not possible with the underlying VBx-based pyannote batch
    pipeline, but diart uses NearestMeansClustering — an online algorithm —
    so it processes audio in a sliding window without needing all embeddings
    in memory at once.  This bounds GPU/CPU memory to the window size
    regardless of recording length.

    For GPU VRAM OOM (e.g. on a 4 GB card where model weights alone fill the
    budget), pass device="cpu" to load_models().
    """

    # Number of float32 samples converted to int16 per write call (~1 s at 16 kHz).
    # Writing in chunks keeps only a small int16 slice in memory at a time instead
    # of allocating a full-recording int16 copy alongside the float32 source array.
    _WAV_CHUNK = 16_384

    def __init__(self) -> None:
        self._pipeline: SpeakerDiarization | None = None

    def load(self, device: str, hf_token: str | None = None) -> None:
        import huggingface_hub

        if hf_token:
            huggingface_hub.login(token=hf_token, add_to_git_credential=False)
        config = SpeakerDiarizationConfig(device=torch.device(device))
        self._pipeline = SpeakerDiarization(config)

    def diarize(self, audio: np.ndarray, sample_rate: int) -> list[DiarizationSegment]:
        if self._pipeline is None:
            raise RuntimeError("Call load() first")

        # Write to a temp WAV so diart can stream it in fixed-size windows.
        # stdlib wave module writes 16-bit PCM; no extra dependency needed.
        # Conversion is done in chunks to avoid materialising a full int16 copy of
        # the entire recording alongside the float32 source array.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                for offset in range(0, len(audio), self._WAV_CHUNK):
                    chunk = audio[offset : offset + self._WAV_CHUNK]
                    wf.writeframes(
                        (chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                    )

            source = FileAudioSource(tmp_path, sample_rate=sample_rate)
            inference = StreamingInference(self._pipeline, source, do_plot=False)
            annotation = inference()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

        return [
            DiarizationSegment(speaker=speaker, start=turn.start, end=turn.end)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
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
