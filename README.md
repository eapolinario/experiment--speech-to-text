# experiment--speech-to-text

A Python REPL that records microphone input, identifies speakers via diarization, and transcribes each speaker's speech.

## How it works

1. Records audio until you press Enter
2. Runs speaker diarization to split audio by speaker
3. Transcribes each segment with [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
4. Prints labeled output: `SPEAKER_00: ...`, `SPEAKER_01: ...`

Both models automatically use the best available hardware:

| Backend | Hardware | Notes |
|---|---|---|
| [CUDA](https://developer.nvidia.com/cuda-toolkit) | NVIDIA GPUs | Preferred when available |
| [MPS](https://developer.apple.com/metal/) | Apple Silicon (M1/M2/M3/M4) | Metal Performance Shaders via macOS |
| CPU | Any | Fallback when no GPU is detected |

Device selection is automatic — no configuration needed.

## Diarization

Speaker diarization uses [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1).

## Setup

### Prerequisites

- [Nix](https://nixos.org/) with flakes enabled
- [direnv](https://direnv.net/) (recommended, for automatic shell activation)
- A [HuggingFace token](https://huggingface.co/settings/tokens) with access to:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) (accept gated access)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (accept gated access)

> **Note:** Setting `HF_TOKEN` in a `.env` file will authenticate your requests to the HuggingFace Hub and alleviate rate limit issues.

### Install

With direnv (recommended — shell activates automatically on `cd`):

```bash
direnv allow
just sync
```

Without direnv:

```bash
nix develop
just sync
```

### Run

```bash
just run
```

## Commands

| Command | Description |
|---|---|
| `just run` | Start the REPL |
| `just test` | Run tests |
| `just test-cov` | Run tests with coverage |
| `just sync` | Install/update dependencies |
