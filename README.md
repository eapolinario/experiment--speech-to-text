# experiment--speech-to-text

A Python REPL that records microphone input, identifies speakers via diarization, and transcribes each speaker's speech.

## How it works

1. Records audio until you press Enter
2. Runs [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) to split audio by speaker
3. Transcribes each segment with [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
4. Prints labeled output: `SPEAKER_00: ...`, `SPEAKER_01: ...`

## Setup

### Prerequisites

- [Nix](https://nixos.org/) with flakes enabled
- A [HuggingFace token](https://huggingface.co/settings/tokens) with access to:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) (accept gated access)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (accept gated access)

> **Note:** Setting `HF_TOKEN` in a `.env` file will authenticate your requests to the HuggingFace Hub and alleviate rate limit issues.

### Install

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
