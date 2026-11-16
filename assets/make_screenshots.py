"""Takes the README screenshot of the real Verbatica window, filled with made-up sample data.

Writes:
  assets/screenshots/main-window-dark.png   the main window after a finished batch, at 2x

The app has only a dark theme, so there is no light copy. The window runs off screen with a
temporary data folder, so no real settings, paths or files are loaded, and nothing goes online.
All paths and file names below are made up.

Run from the project root:
  pip install -e .
  python assets/make_screenshots.py
"""

from __future__ import annotations

import os
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

OUT = ROOT / "assets" / "screenshots"
WINDOW_SIZE = (1000, 860)  # logical px; the saved image is twice this
RELEASE_VERSION = "v1.0.0"  # newest release tag, shown instead of the development version

# Made-up sample data.
HOME = "/Users/alex"
MODELS_DIR = f"{HOME}/Applications/Verbatica-data/models"
LOGS_DIR = f"{HOME}/Applications/Verbatica-data/logs"
OUTPUT_DIR = f"{HOME}/Documents/Interviews/Transcripts"
MODEL = "small"
FILES = [  # (file name, seconds taken, words)
    ("interview-maria-chen.m4a", 48.31, 2964),
    ("team-standup-monday.mp3", 12.07, 811),
    ("lecture-03-thermodynamics.mp4", 131.62, 8207),
    ("podcast-episode-12.wav", 74.88, 5130),
]


def finished_batch(window) -> None:
    """Drives the window's own handlers into the state after a successful batch of four files."""
    window.settings.models_dir = MODELS_DIR
    window.settings.output_dir = OUTPUT_DIR
    window.settings.error_log_dir = LOGS_DIR
    window.models_dir_input.setText(MODELS_DIR)
    window.output_dir_input.setText(OUTPUT_DIR)
    window.error_log_input.setText(LOGS_DIR)
    for ext, check in window.format_checks.items():
        check.setChecked(ext in (".txt", ".srt"))
    window.version_btn.setText(RELEASE_VERSION)

    window.model_combo.setCurrentIndex(window.model_combo.findData(MODEL))
    window.model_ready(object(), MODEL)

    window.files = [f"{HOME}/Documents/Interviews/{name}" for name, _, _ in FILES]
    window.file_list.addItems([name for name, _, _ in FILES])
    for name, seconds, words in FILES:
        window.file_completed(name, seconds, words)
    window.progress_bar.setValue(100)
    window.all_completed(sum(f[1] for f in FILES), sum(f[2] for f in FILES), len(FILES), 0, False)
    window.file_list.clearSelection()
    window.setFocus()


def main() -> int:
    app = QApplication(sys.argv[:1])
    app.setWindowIcon(verbatica_app.app_icon())
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        store = SettingsStore(Path(folder) / "app", legacy_file=Path(folder) / "none.json")
        window = verbatica_app.MainWindow(store)
        window.save_settings = lambda: None  # keep the temporary data folder untouched
        window.resize(*WINDOW_SIZE)
        finished_batch(window)
        window.show()
        app.processEvents()
        window.stats_text.verticalScrollBar().setValue(0)  # show the per-file results first
        app.processEvents()
        path = OUT / "main-window-dark.png"
        window.grab().save(str(path))
        print(path)
        window.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
