# Real-Time Speech-to-Text (Enhanced Edition)

This project is modified and extended from:

https://github.com/jianchang512/realtime-stt

The original repository provides a lightweight real-time speech-to-text desktop application built with sherpa-onnx and PySide6.

This version enhances usability, transcript structure, and session control while preserving the core recognition functionality.

---

## Overview

A desktop real-time speech-to-text tool supporting Chinese and English speech recognition.

Features include:

- Real-time microphone transcription
- Automatic punctuation restoration
- Sentence-level segmentation (each sentence on a new line)
- Optional timestamp display (dynamic toggle)
- Pause and resume without breaking the recording session
- Synchronized TXT export
- WAV recording of the entire session

---

## Key Enhancements

### 1. Sentence-Level Segmentation
Transcribed text is automatically split into sentences based on punctuation.
Each sentence is displayed on a new line for better readability.

---

### 2. Optional Timestamp Toggle (Real-Time)
A timestamp button in the top control bar allows enabling or disabling timestamps instantly.

When enabled, transcript lines are displayed in the format:

[00:01:23.450 - 00:01:28.120] Sentence text

- Switching the toggle refreshes all previous transcript content immediately.
- The upper real-time preview panel does not display timestamps.
- Exported TXT files follow the current toggle state.

---

### 3. Pause / Resume Without Session Reset
- Dedicated Pause button.
- Pausing does not create a new recording segment.
- Resuming continues within the same WAV file.
- The Start button is disabled while paused for clarity.

---

### 4. Structured Transcript Management
Internally redesigned transcript handling:

- Each segment is stored as structured data:
  ```python
  {"start": float, "end": float, "text": str}

Display logic and data logic are separated.

Timestamp toggling re-renders all stored segments dynamically.

Session TXT output remains synchronized with the current UI state.

Display logic and data logic are separated.

Timestamp toggling re-renders all stored segments dynamically.

Session TXT output remains synchronized with the current UI state.

## Project Structure

```
realtime-stt-main/
│
├── stt.py
├── data/
├── onnx/
├── ffmpeg/
├── output/
```

Models must be placed in:

```
./onnx/
```

------

## Packaging

Recommended packaging method:

```
uv run pyinstaller -w -n realtime-stt \
  --collect-all PySide6 \
  --add-data "data;data" \
  --add-data "onnx;onnx" \
  --add-data "ffmpeg;ffmpeg" \
  stt.py
```

Packaging mode: `onedir` (recommended).

------

## Requirements

- Python 3.12
- sherpa-onnx
- onnxruntime
- PySide6
- sounddevice

------

## Download

Pre-built Windows executable is available in the Releases section.

Enhancements:
\- Sentence-level segmentation (newline per sentence)
\- Timestamp toggle (instant re-render of transcript)
\- Pause/resume without breaking the recording session
\- TXT export follows current timestamp display state

Due to GitHub release file size limits, this package does not include ASR model files.

Please download models from the original project

## Attribution

This project is based on the original repository:

[https://github.com/jianchang512/realtime-stt](https://github.com/jianchang512/realtime-stt?utm_source=chatgpt.com)

All original credits belong to the original author.
 This repository focuses on functional extension and structural improvements.
