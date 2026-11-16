"""Takes the README screenshots of the real Verbatica window, filled with made-up sample data.

Writes, at 2x:
  assets/screenshots/main-window-dark.png    a batch in progress, dark appearance
  assets/screenshots/main-window-light.png   the same batch, light appearance
  assets/screenshots/empty-dark.png          the window on first open
  assets/screenshots/settings-dark.png       the Settings window

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

from PyQt6.QtWidgets import QApplication  # noqa: E402

from verbatica import app as verbatica_app  # noqa: E402
from verbatica.settings import SettingsStore  # noqa: E402
from verbatica.transcription import WAVEFORM_BARS  # noqa: E402

OUT = ROOT / "assets" / "screenshots"
WINDOW_SIZE = (860, 560)  # logical px; the saved image is twice this
RELEASE_VERSION = "v1.0.0"  # newest release tag, shown instead of the development version

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
