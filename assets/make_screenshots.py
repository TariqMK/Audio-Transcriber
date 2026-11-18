"""Takes the README screenshots of the real Verbatica window, filled with made-up sample data.

Writes, at 2x:
  assets/screenshots/main-window-dark.png    a batch in progress, dark appearance
  assets/screenshots/main-window-light.png   the same batch, light appearance
  assets/screenshots/empty-dark.png          the window on first open
  assets/screenshots/settings-dark.png       the Settings window
  assets/screenshots/queue-<theme>.png       close-up of the queue: waveforms, progress, word counts
  assets/screenshots/silence-<theme>.png     close-up of two files with silent parts shown as flat lines
  assets/screenshots/outputs-<theme>.png     one transcript saved as Text, SRT and VTT
  assets/screenshots/models-<theme>.png      the Model menu open with all six sizes

The window runs off screen with a temporary data folder, so no real settings, paths or files
are loaded, and nothing goes online. All paths, file names and waveforms below are made up.

Run from the project root:
  pip install -e .
  python assets/make_screenshots.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_SCALE_FACTOR"] = "2"  # saved at 2x so it stays sharp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QPoint, QRect, Qt  # noqa: E402
from PyQt6.QtGui import QPainter  # noqa: E402
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget  # noqa: E402

from verbatica import theme  # noqa: E402

from verbatica import app as verbatica_app  # noqa: E402
from verbatica.settings import SettingsStore  # noqa: E402
from verbatica.transcription import WAVEFORM_BARS, to_srt, to_vtt  # noqa: E402

OUT = ROOT / "assets" / "screenshots"
WINDOW_SIZE = (860, 560)  # logical px; the saved image is twice this
RELEASE_VERSION = "v2.0.1"  # newest release tag, shown instead of the development version

# Made-up sample data.
HOME = "/Users/alex"
MODELS_DIR = f"{HOME}/Applications/Verbatica-data/models"
LOGS_DIR = f"{HOME}/Applications/Verbatica-data/logs"
OUTPUT_DIR = f"{HOME}/Documents/Interviews/Transcripts"
MODEL = "small"
FILES = [  # (file name, length in seconds, state, words or progress)
    ("interview-maria-chen.m4a", 2892, "done", 2964),
    ("team-standup-monday.mp3", 724, "done", 811),
    ("lecture-03-thermodynamics.mp4", 5500, "active", 0.62),
    ("podcast-episode-12.wav", 0, "waiting", None),
]

SILENCE_FILES = [  # (file name, length in seconds, state, words or progress, silent parts as fractions)
    ("board-meeting-q3.m4a", 3310, "done", 3402, [(0.18, 0.27), (0.63, 0.70)]),
    ("field-recording-harbour.wav", 1460, "active", 0.58, [(0.30, 0.44)]),
]
TRANSCRIPT_NAME = "interview-maria-chen"
SEGMENTS = [  # made-up transcript; the gap after 8.6 s is skipped silence
    {"start": 0.0, "end": 4.2, "text": " Thanks for making the time today."},
    {"start": 4.2, "end": 8.6, "text": " Of course. Where would you like to start?"},
    {"start": 15.1, "end": 20.4, "text": " Let's begin with how the project started."},
    {"start": 20.4, "end": 26.0, "text": " It began as a small study with three schools."},
]


def speech_like(seed: int, gap=None) -> list:
    """A made-up waveform that looks like speech: phrases with short pauses, and an optional silent gap."""
    rnd = random.Random(seed)
    peaks, level = [], 0.6
    for i in range(WAVEFORM_BARS):
        t = i / WAVEFORM_BARS
        if gap and gap[0] <= t < gap[1]:
            peaks.append(0.02)
            continue
        level += rnd.uniform(-0.18, 0.18)
        level = min(max(level, 0.25), 1.0)
        pause = 0.15 if rnd.random() < 0.08 else 1.0
        peaks.append(round(max(0.05, level * pause * (0.55 + 0.45 * abs(math.sin(i * 0.9)))), 3))
    return peaks


class RunningWorker:
    """Stands in for a batch that is still running, so the window shows its running state."""

    def isRunning(self):
        return True

    def cancel(self):
        pass


def make_window(store, dark: bool):
    window = verbatica_app.MainWindow(store, dark=dark)
    window.save_settings = lambda: None  # keep the temporary data folder untouched
    window.settings.models_dir = MODELS_DIR
    window.settings.output_dir = OUTPUT_DIR
    window.settings.error_log_dir = LOGS_DIR
    window.settings.output_formats = [".txt", ".srt"]
    for ext, chip in window.format_checks.items():
        chip.blockSignals(True)
        chip.setChecked(ext in window.settings.output_formats)
        chip.blockSignals(False)
    window.paint_chips()
    window.resize(*WINDOW_SIZE)
    window.menuBar().setVisible(False)  # on macOS the menus sit in the system menu bar, not the window
    return window


def batch_in_progress(window) -> None:
    """Drives the window's own handlers into a batch of four files with the third one running."""
    window.model_combo.setCurrentIndex(window.model_combo.findData(MODEL))
    window.model_ready(object(), MODEL)
    window.add_files([f"{HOME}/Documents/Interviews/{name}" for name, *_ in FILES])
    window.batch_rows = list(window.rows)
    window.worker = RunningWorker()
    for i, (name, length, state, value) in enumerate(FILES):
        row = window.rows[i]
        if state == "waiting":
            continue
        window.file_started(i)
        row.set_audio(speech_like(i + 3, gap=(0.41, 0.47) if i == 2 else None), length)
        if state == "done":
            window.file_completed(i, f"{OUTPUT_DIR}/{Path(name).stem}", length / 6, value)
        else:
            row.waveform.set_fraction(value, animate=False)
            row.result_label.setText(f"{int(value * 100)}%")
    window.progress_bar.setValue(65)
    window.refresh()


def silence_in_progress(window) -> None:
    """Two files whose silent parts show as flat lines: one finished, one running."""
    window.model_combo.setCurrentIndex(window.model_combo.findData(MODEL))
    window.model_ready(object(), MODEL)
    window.add_files([f"{HOME}/Documents/Recordings/{name}" for name, *_ in SILENCE_FILES])
    window.batch_rows = list(window.rows)
    window.worker = RunningWorker()
    for i, (name, length, state, value, gaps) in enumerate(SILENCE_FILES):
        row = window.rows[i]
        window.file_started(i)
        peaks = speech_like(i + 11)
        for start, end in gaps:
            for b in range(int(start * WAVEFORM_BARS), int(end * WAVEFORM_BARS)):
                peaks[b] = 0.02
        row.set_audio(peaks, length)
        if state == "done":
            window.file_completed(i, f"{OUTPUT_DIR}/{Path(name).stem}", length / 6, value)
        else:
            row.waveform.set_fraction(value, animate=False)
            row.result_label.setText(f"{int(value * 100)}%")
    window.refresh()


def outputs_view(dark: bool) -> QWidget:
    """The same made-up transcript as the app writes it in each format, side by side.
    The file contents come from the app's own writers (to_srt, to_vtt)."""
    t = theme.current(dark)
    mono = 'font-family: "SF Mono", Menlo, Consolas, "DejaVu Sans Mono", monospace; font-size: 12px;'
    text = " ".join(seg["text"].strip() for seg in SEGMENTS)
    contents = [(".txt", text), (".srt", to_srt(SEGMENTS)), (".vtt", to_vtt(SEGMENTS))]
    view = QWidget()
    view.setStyleSheet(f"background: {t.window};")
    layout = QHBoxLayout(view)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(12)
    for (ext, body), stretch in zip(contents, (4, 5, 5)):
        card = QFrame()
        card.setStyleSheet(f"QFrame {{ background: {t.surface}; border: 1px solid {t.line}; border-radius: 8px; }}"
                           "QLabel { border: none; background: transparent; }")
        inner = QVBoxLayout(card)
        inner.setContentsMargins(14, 12, 14, 14)
        inner.setSpacing(10)
        head = QLabel(TRANSCRIPT_NAME + ext)
        head.setStyleSheet(f"color: {t.muted}; {mono}")
        label = QLabel(body.strip())
        label.setWordWrap(ext == ".txt")
        label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        label.setStyleSheet(f"color: {t.text}; {mono}")
        inner.addWidget(head)
        inner.addWidget(label, 1)
        layout.addWidget(card, stretch)
    view.resize(WINDOW_SIZE[0], 10)
    view.resize(WINDOW_SIZE[0], view.heightForWidth(WINDOW_SIZE[0]) if view.hasHeightForWidth() else view.sizeHint().height())
    return view


def models_open(app, window, name) -> None:
    """The window's header with the Model menu open, saved as one image."""
    combo = window.model_combo
    combo.setCurrentIndex(combo.findData(MODEL))
    window.show()
    app.processEvents()
    combo.showPopup()
    app.processEvents()
    app.processEvents()
    popup = combo.view().window()
    shot = window.grab()
    painter = QPainter(shot)
    # Off screen there is no real screen to place the menu on, so draw it just below the Model button.
    offset = combo.mapTo(window, QPoint(0, combo.height() + 4))
    painter.drawPixmap(offset, popup.grab())
    painter.end()
    bottom = min(offset.y() + popup.height() + 16, WINDOW_SIZE[1])
    ratio = shot.devicePixelRatio()
    shot = shot.copy(0, 0, int(WINDOW_SIZE[0] * ratio), int(bottom * ratio))
    combo.hidePopup()
    path = OUT / name
    shot.save(str(path))
    print(path)


def save_rows(app, window, name):
    """The queue card cropped just below its last row."""
    card = window.queue_card
    window.show()
    app.processEvents()
    app.processEvents()
    last = window.rows[-1]
    bottom = last.mapTo(card, QPoint(0, last.height())).y() + 1
    path = OUT / name
    card.grab(QRect(0, 0, card.width(), bottom)).save(str(path))
    print(path)


def save(app, widget, name):
    widget.show()
    app.processEvents()
    app.processEvents()
    path = OUT / name
    widget.grab().save(str(path))
    print(path)


def main() -> int:
    app = QApplication(sys.argv[:1])
    app.setWindowIcon(verbatica_app.app_icon())
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        store = SettingsStore(Path(folder) / "app", legacy_file=Path(folder) / "none.json")

        for dark, name in ((True, "main-window-dark.png"), (False, "main-window-light.png")):
            window = make_window(store, dark)
            batch_in_progress(window)
            save(app, window, name)
            window.worker = None
            window.close()

        for dark, suffix in ((True, "dark"), (False, "light")):
            window = make_window(store, dark)
            batch_in_progress(window)
            save_rows(app, window, f"queue-{suffix}.png")
            window.worker = None
            window.close()

            window = make_window(store, dark)
            silence_in_progress(window)
            save_rows(app, window, f"silence-{suffix}.png")
            window.worker = None
            window.close()

            save(app, outputs_view(dark), f"outputs-{suffix}.png")

            window = make_window(store, dark)
            window.refresh()
            models_open(app, window, f"models-{suffix}.png")
            window.close()

        window = make_window(store, True)
        window.refresh()
        save(app, window, "empty-dark.png")

        window.show_settings()
        window.settings_window.version_label.setText(RELEASE_VERSION)
        window.settings_window.refresh(window)
        save(app, window.settings_window, "settings-dark.png")
        window.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
