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

AUDIO_FILE_FILTER = "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.mp4 *.avi *.mkv);;All Files (*)"

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


def transcribe_file(model, file_path, on_progress, should_cancel) -> dict:
    """Transcribe one file. faster-whisper returns segments one at a time as it decodes them
    (about every 30 seconds of audio), which gives progress and a point to stop after each one."""
    segments, info = model.transcribe(str(file_path), vad_filter=VAD_FILTER, beam_size=BEAM_SIZE)
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
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    file_complete = pyqtSignal(str, float, int)
    all_complete = pyqtSignal(float, int, int, int, bool)  # time, words, succeeded, failed, cancelled
    error = pyqtSignal(str, str)  # filename, error_message

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
            self.status.emit(f"Transcribing {file_name} ({idx + 1}/{total_files})...")

            try:
                # Transcribe, with progress and cancel checks inside the file
                file_start = time.time()
                result = transcribe_file(
                    self.model,
                    file_path,
                    lambda fraction: self.progress.emit(int((idx + fraction) / total_files * 100)),
                    lambda: self.is_cancelled,
                )
                file_duration = time.time() - file_start

                word_count = len(result["text"].split())
                total_words += word_count

                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
                output_base = unique_output_base(self.output_dir, Path(file_path).stem, self.formats)
                self.save_outputs(result, output_base)

                successful_files += 1
                label = file_name if output_base.name == Path(file_path).stem else f"{file_name} → {output_base.name}"
                self.file_complete.emit(label, file_duration, word_count)

            except TranscriptionCancelled:
                break

            except Exception as e:
                self.errors.append((file_name, str(e)))
                self.error.emit(file_name, str(e))
                self.status.emit(f"⚠️ Skipped {file_name} due to error")

            self.progress.emit(int(((idx + 1) / total_files) * 100))

        if self.errors and self.error_log_dir:
            self.write_error_log()

        if self.is_cancelled:
            self.status.emit("Transcription cancelled")

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
            self.error.emit("Error Log", f"Could not write error log: {e}")

    def cancel(self):
        self.is_cancelled = True
