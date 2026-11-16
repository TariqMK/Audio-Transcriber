"""Transcribes a batch of files on a background thread and saves each result as text
and subtitle files, without ever overwriting an existing result."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

# Output formats the user can choose, by file extension
OUTPUT_FORMATS = {
    ".txt": "Text (.txt)",
    ".srt": "Subtitles (.srt)",
    ".vtt": "Subtitles (.vtt)",
}

AUDIO_EXTENSIONS = ("mp3", "m4a", "wav", "flac", "ogg", "mp4", "mov", "avi", "mkv")
AUDIO_FILE_FILTER = f"Audio and Video ({' '.join('*.' + e for e in AUDIO_EXTENSIONS)});;All Files (*)"

# faster-whisper decodes every file to 16 kHz mono before it transcribes it
SAMPLE_RATE = 16000

# Number of bars in the waveform the window draws for each file. The window merges bars to fit.
WAVEFORM_BARS = 160

# Skip long silences before transcribing. Faster, and avoids the text Whisper sometimes
# invents in silent parts ("Thanks for watching!").
VAD_FILTER = True

# One search path (greedy decoding), as openai-whisper's transcribe() used by default.
# faster-whisper defaults to 5, which was about 1.7x slower here with the same accuracy.
BEAM_SIZE = 1


def add_unique_files(existing: list, new_files) -> int:
    """Append the files that are not already in `existing` (compared by real path).
    Returns how many were skipped as duplicates."""
    seen = {os.path.normcase(os.path.realpath(f)) for f in existing}
    skipped = 0
    for f in new_files:
        key = os.path.normcase(os.path.realpath(f))
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        existing.append(f)
    return skipped


def unique_output_base(output_dir, stem: str, extensions) -> Path:
    """Return a base path (without extension) for which none of the output files exist yet,
    so an existing transcript is never overwritten: 'talk', then 'talk (2)', 'talk (3)'..."""
    output_dir = Path(output_dir)
    n = 1
    while True:
        name = stem if n == 1 else f"{stem} ({n})"
        if not any((output_dir / (name + ext)).exists() for ext in extensions):
            return output_dir / name
        n += 1


def format_timestamp(seconds: float, decimal_marker: str) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{ms:03d}"


def to_srt(segments) -> str:
    cues = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start"], ",")
        end = format_timestamp(seg["end"], ",")
        cues.append(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n")
    return "\n".join(cues)


def to_vtt(segments) -> str:
    cues = ["WEBVTT\n"]
    for seg in segments:
        start = format_timestamp(seg["start"], ".")
        end = format_timestamp(seg["end"], ".")
        cues.append(f"{start} --> {end}\n{seg['text'].strip()}\n")
    return "\n".join(cues)


class TranscriptionCancelled(Exception):
    pass


def waveform_peaks(audio, bars: int = WAVEFORM_BARS) -> list:
    """Loudness of `bars` equal slices of the audio, from 0.0 to 1.0, for the window to draw.
    Uses the RMS of each slice relative to the loudest slice. Silence stays near 0."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return [0.0] * bars
    slices = np.array_split(audio, bars)
    rms = np.array([float(np.sqrt(np.mean(np.square(s)))) if s.size else 0.0 for s in slices])
    loudest = rms.max()
    if loudest <= 1e-6:
        return [0.0] * bars
    # A small power curve lifts quiet speech, so a soft voice doesn't read as silence
    return [round(float(v), 3) for v in np.power(rms / loudest, 0.7)]


def load_audio(file_path):
    """Decode a file the way faster-whisper does (16 kHz mono) and measure its waveform.
    Returns (audio, peaks, duration in seconds). The audio is then handed to the model,
    so each file is decoded only once."""
    from faster_whisper.audio import decode_audio  # imported here because it loads PyAV

    audio = decode_audio(str(file_path), sampling_rate=SAMPLE_RATE)
    return audio, waveform_peaks(audio), len(audio) / SAMPLE_RATE


def transcribe_file(model, file_path, on_progress, should_cancel, on_audio=lambda peaks, duration: None) -> dict:
    """Transcribe one file. faster-whisper returns segments one at a time as it decodes them
    (about every 30 seconds of audio), which gives progress and a point to stop after each one."""
    audio, peaks, duration = load_audio(file_path)
    on_audio(peaks, duration)
    if should_cancel():
        raise TranscriptionCancelled()
    segments, info = model.transcribe(audio, vad_filter=VAD_FILTER, beam_size=BEAM_SIZE)
    collected = []
    for segment in segments:
        if should_cancel():
            raise TranscriptionCancelled()
        collected.append({"start": segment.start, "end": segment.end, "text": segment.text})
        if info.duration:
            on_progress(min(segment.end / info.duration, 1.0))
    if should_cancel():
        raise TranscriptionCancelled()
    return {"text": "".join(s["text"] for s in collected), "segments": collected}


class TranscriptionWorker(QThread):
    """Transcribes `files` in order. Per-file signals carry the file's index in `files`."""

    progress = pyqtSignal(int)  # whole batch, 0-100
    file_started = pyqtSignal(int)  # index
    file_audio = pyqtSignal(int, object, float)  # index, waveform peaks, duration in seconds
    file_progress = pyqtSignal(int, float)  # index, 0.0-1.0
    file_complete = pyqtSignal(int, str, float, int)  # index, output path without extension, seconds taken, words
    all_complete = pyqtSignal(float, int, int, int, bool)  # time, words, succeeded, failed, cancelled
    error = pyqtSignal(int, str, str)  # index (-1 for the error log itself), file name, message

    def __init__(self, files, model, output_dir, error_log_dir, formats=(".txt",)):
        super().__init__()
        self.files = files
        self.model = model
        self.output_dir = output_dir
        self.error_log_dir = error_log_dir
        self.formats = list(formats)
        self.is_cancelled = False
        self.errors = []

    def run(self):
        total_files = len(self.files)
        total_words = 0
        successful_files = 0
        start_time = time.time()

        for idx, file_path in enumerate(self.files):
            if self.is_cancelled:
                break

            file_name = Path(file_path).name
            self.file_started.emit(idx)

            def on_progress(fraction, idx=idx):
                self.file_progress.emit(idx, fraction)
                self.progress.emit(int((idx + fraction) / total_files * 100))

            try:
                # Transcribe, with progress and cancel checks inside the file
                file_start = time.time()
                result = transcribe_file(
                    self.model,
                    file_path,
                    on_progress,
                    lambda: self.is_cancelled,
                    lambda peaks, duration, idx=idx: self.file_audio.emit(idx, peaks, duration),
                )
                file_duration = time.time() - file_start

                word_count = len(result["text"].split())
                total_words += word_count

                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
                output_base = unique_output_base(self.output_dir, Path(file_path).stem, self.formats)
                self.save_outputs(result, output_base)

                successful_files += 1
                self.file_complete.emit(idx, str(output_base), file_duration, word_count)

            except TranscriptionCancelled:
                break

            except Exception as e:
                self.errors.append((file_name, str(e)))
                self.error.emit(idx, file_name, str(e))

            self.progress.emit(int(((idx + 1) / total_files) * 100))

        if self.errors and self.error_log_dir:
            self.write_error_log()

        total_duration = time.time() - start_time
        self.all_complete.emit(total_duration, total_words, successful_files,
                               len(self.errors), self.is_cancelled)

    def save_outputs(self, result, output_base: Path):
        renderers = {
            ".txt": lambda: result["text"],
            ".srt": lambda: to_srt(result["segments"]),
            ".vtt": lambda: to_vtt(result["segments"]),
        }
        for ext in self.formats:
            with open(f"{output_base}{ext}", "w", encoding="utf-8") as f:
                f.write(renderers[ext]())

    def write_error_log(self):
        try:
            log_dir = Path(self.error_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"transcription_errors_{timestamp}.log"

            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Transcription Error Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                for filename, error in self.errors:
                    f.write(f"File: {filename}\n")
                    f.write(f"Error: {error}\n")
                    f.write("-" * 70 + "\n\n")
        except Exception as e:
            self.error.emit(-1, "Error log", f"Could not write the error log: {e}")

    def cancel(self):
        self.is_cancelled = True
