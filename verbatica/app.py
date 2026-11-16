"""The Verbatica main window: choose a model, folders, formats and files,
then transcribe them. Also holds the version button that checks for updates."""

from __future__ import annotations

import base64
import sys
import traceback
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPalette, QPixmap
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
                             QMessageBox, QProgressBar, QPushButton, QTextEdit, QVBoxLayout,
                             QWidget)

from verbatica import APP_NAME, __version__, update_check
from verbatica._icon import ICON_PNGS
from verbatica.models import DEFAULT_MODEL, MODEL_SIZES, MODELS, ModelDownloadWorker
from verbatica.settings import SettingsStore
from verbatica.transcription import (AUDIO_FILE_FILTER, OUTPUT_FORMATS, TranscriptionWorker,
                                             add_unique_files)

VERSION_TEXT = f"v{__version__}"
VERSION_MUTED = "#a6a6a6"
VERSION_ACCENT = "#64b5f6"
VERSION_STYLE = """
QPushButton#version {{
    color: {color}; background: transparent; border: 1px solid transparent;
    border-radius: 4px; padding: 2px 6px; font-size: 12px;
}}
QPushButton#version:hover {{ background: rgba(255, 255, 255, 0.10); }}
QPushButton#version:focus {{ border: 1px solid #2a82da; }}
QPushButton#version:disabled {{ color: {color}; }}
"""

def app_icon() -> QIcon:
    """The window and taskbar icon, built from PNG data inside the package (no files to ship)."""
    icon = QIcon()
    for data in ICON_PNGS.values():
        pixmap = QPixmap()
        pixmap.loadFromData(base64.b64decode(data), "PNG")
        icon.addPixmap(pixmap)
    return icon


# Main action buttons. One accent colour (the theme's highlight blue) marks the primary action;
# the rest are neutral. White on #2a82da is 4.0:1, above the 3:1 needed for bold 14 px text.
BUTTON_STYLE = """
QPushButton#primary, QPushButton#secondary {
    font-size: 14px; font-weight: bold; padding: 10px 18px; border-radius: 6px;
}
QPushButton#primary { background: #2a82da; color: #ffffff; border: 1px solid #2a82da; }
QPushButton#primary:hover { background: #3a8fe3; }
QPushButton#primary:pressed { background: #2272c2; }
QPushButton#secondary { background: #454545; color: #f2f2f2; border: 1px solid #5a5a5a; }
QPushButton#secondary:hover { background: #4f4f4f; }
QPushButton#secondary:pressed { background: #3d3d3d; }
QPushButton#primary:disabled, QPushButton#secondary:disabled {
    background: #3a3a3a; color: #7a7a7a; border: 1px solid #444444;
}
QPushButton#primary:focus, QPushButton#secondary:focus { border: 1px solid #9cc8f2; }
"""


class UpdateCheckWorker(QThread):
    result = pyqtSignal(object)  # update_check.UpdateInfo
    failed = pyqtSignal(str)

    def run(self):
        try:
            self.result.emit(update_check.check())
        except update_check.UpdateError as e:
            self.failed.emit(str(e))
        except Exception:
            self.failed.emit(f"GitHub sent an answer {APP_NAME} can't read. Try again later.")


class MainWindow(QMainWindow):
    def __init__(self, store: SettingsStore | None = None):
        super().__init__()
        self.files = []
        self.worker = None
        self.download_worker = None
        self.update_worker = None
        self.model = None
        self.model_loaded = False
        self.settings_warning_shown = False
        self.store = store or SettingsStore()
        self.settings = self.store.load()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, 950, 800)
        self.set_dark_theme()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header: title, with the version button added at the end of init_ui (see below)
        self.header_layout = QHBoxLayout()
        self.header_layout.addStretch()
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(QSize(34, 34), logo.devicePixelRatioF()))  # sharp on Retina
        logo.setAccessibleName(f"{APP_NAME} icon")
        self.header_layout.addWidget(logo)
        title = QLabel(APP_NAME)
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.header_layout.addWidget(title)
        main_layout.addLayout(self.header_layout)

        # Models Directory Configuration
        models_dir_group = QGroupBox("Models Directory")
        models_dir_layout = QHBoxLayout()

        models_dir_layout.addWidget(QLabel("Models Folder:"))
        self.models_dir_input = QLineEdit()
        self.models_dir_input.setText(self.settings.models_dir)
        self.models_dir_input.setReadOnly(True)
        self.models_dir_input.setPlaceholderText("Click 'Browse' to select models directory")
        models_dir_layout.addWidget(self.models_dir_input)

        self.browse_models_dir_btn = QPushButton("Browse")
        self.browse_models_dir_btn.clicked.connect(self.select_models_dir)
        models_dir_layout.addWidget(self.browse_models_dir_btn)

        models_dir_group.setLayout(models_dir_layout)
        main_layout.addWidget(models_dir_group)

        # Model Selection & Download
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        for model_name in MODELS:
            self.model_combo.addItem(f"{model_name} ({MODEL_SIZES[model_name]})", model_name)
        self.model_combo.setCurrentIndex(self.model_combo.findData(DEFAULT_MODEL))
        model_select_layout.addWidget(self.model_combo)

        self.load_model_btn = QPushButton("Download && Load Model")  # "&&" shows one "&" in Qt
        self.load_model_btn.setObjectName("secondary")
        self.load_model_btn.setStyleSheet(BUTTON_STYLE)
        self.load_model_btn.clicked.connect(self.download_and_load_model)
        model_select_layout.addWidget(self.load_model_btn)

        model_select_layout.addStretch()
        model_layout.addLayout(model_select_layout)

        self.download_progress = QProgressBar()
        self.download_progress.setValue(0)
        self.download_progress.setVisible(False)
        model_layout.addWidget(self.download_progress)

        self.model_status_label = QLabel("⚪ Model not loaded")
        self.model_status_label.setStyleSheet("color: #ff9800; font-weight: bold;")
        model_layout.addWidget(self.model_status_label)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # Output Settings
        settings_group = QGroupBox("Output Settings")
        settings_layout = QVBoxLayout()

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(self.settings.output_dir)
        self.output_dir_input.setReadOnly(True)
        self.output_dir_input.setPlaceholderText("Click 'Browse' to select output directory")
        output_dir_layout.addWidget(self.output_dir_input)

        self.browse_output_dir_btn = QPushButton("Browse")
        self.browse_output_dir_btn.clicked.connect(self.select_output_dir)
        output_dir_layout.addWidget(self.browse_output_dir_btn)
        settings_layout.addLayout(output_dir_layout)

        error_log_layout = QHBoxLayout()
        error_log_layout.addWidget(QLabel("Error Log Directory:"))
        self.error_log_input = QLineEdit()
        self.error_log_input.setText(self.settings.error_log_dir)
        self.error_log_input.setReadOnly(True)
        self.error_log_input.setPlaceholderText("Click 'Browse' to select error log directory")
        error_log_layout.addWidget(self.error_log_input)

        self.browse_error_log_btn = QPushButton("Browse")
        self.browse_error_log_btn.clicked.connect(self.select_error_log_dir)
        error_log_layout.addWidget(self.browse_error_log_btn)
        settings_layout.addLayout(error_log_layout)

        formats_layout = QHBoxLayout()
        formats_layout.addWidget(QLabel("Save As:"))
        self.format_checks = {}
        for ext, label in OUTPUT_FORMATS.items():
            check = QCheckBox(label)
            check.setChecked(ext in self.settings.output_formats)
            check.toggled.connect(self.formats_changed)
            formats_layout.addWidget(check)
            self.format_checks[ext] = check
        formats_layout.addStretch()
        settings_layout.addLayout(formats_layout)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # File selection
        file_layout = QHBoxLayout()
        self.select_file_btn = QPushButton("Select File(s)")
        self.select_file_btn.clicked.connect(self.select_files)
        file_layout.addWidget(self.select_file_btn)

        self.clear_btn = QPushButton("Clear List")
        self.clear_btn.clicked.connect(self.clear_files)
        file_layout.addWidget(self.clear_btn)

        file_layout.addStretch()
        main_layout.addLayout(file_layout)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        main_layout.addWidget(QLabel("Selected Files:"))
        main_layout.addWidget(self.file_list)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready - Select a model and click 'Download & Load Model'")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #ff9800; font-size: 14px; font-weight: bold;")
        main_layout.addWidget(self.status_label)

        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setPlaceholderText("Statistics will appear here after transcription...")
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        main_layout.addWidget(stats_group)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Transcription")
        self.start_btn.setObjectName("primary")
        self.start_btn.setStyleSheet(BUTTON_STYLE)
        self.start_btn.setEnabled(False)  # Disabled until model is loaded
        self.start_btn.clicked.connect(self.start_transcription)
        button_layout.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setObjectName("secondary")
        self.cancel_btn.setStyleSheet(BUTTON_STYLE)
        self.cancel_btn.clicked.connect(self.cancel_transcription)
        button_layout.addWidget(self.cancel_btn)

        main_layout.addLayout(button_layout)

        # Version button: created last so it comes after the main content in the tab order
        self.version_btn = QPushButton(VERSION_TEXT)
        self.version_btn.setObjectName("version")
        self.version_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.version_btn.setToolTip("Check for updates")
        self.version_btn.setAccessibleName(f"Version {VERSION_TEXT}")
        self.version_btn.setAccessibleDescription("Check GitHub for a newer release.")
        self.version_btn.setStyleSheet(VERSION_STYLE.format(color=VERSION_MUTED))
        self.version_btn.clicked.connect(self.check_for_updates)
        self.header_layout.addWidget(self.version_btn, alignment=Qt.AlignmentFlag.AlignBottom)
        self.header_layout.addStretch()

        self.browse_models_dir_btn.setFocus()

    def set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(palette)

    # --- Settings ------------------------------------------------------------

    def save_settings(self):
        try:
            self.store.save(self.settings)
        except OSError as e:
            if self.settings_warning_shown:
                return
            self.settings_warning_shown = True
            QMessageBox.warning(
                self,
                "Settings Not Saved",
                f"{APP_NAME} can't save its settings in:\n{self.store.data_dir}\n\n{e.strerror or e}\n\n"
                "Move the app to a folder you can write to, for example your Documents folder. "
                "Your choices stay in use until you close the app.",
            )

    def select_models_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Models Directory", self.settings.models_dir)
        if directory:
            self.settings.models_dir = directory
            self.models_dir_input.setText(directory)
            self.save_settings()
            self.status_label.setText("Models directory set - Select a model and click 'Download & Load Model'")
            self.status_label.setStyleSheet("color: #2196F3; font-size: 14px; font-weight: bold;")

    def select_error_log_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Error Log Directory", self.settings.error_log_dir)
        if directory:
            self.settings.error_log_dir = directory
            self.error_log_input.setText(directory)
            self.save_settings()

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory for Transcriptions",
                                                     self.settings.output_dir)
        if directory:
            self.settings.output_dir = directory
            self.output_dir_input.setText(directory)
            self.save_settings()

    def formats_changed(self):
        self.settings.output_formats = [ext for ext, check in self.format_checks.items() if check.isChecked()]
        self.save_settings()

    # --- Model ---------------------------------------------------------------

    def download_and_load_model(self):
        if not self.settings.models_dir:
            QMessageBox.warning(self, "Warning", "Please select a models directory first!")
            return
        self.start_model_worker(self.model_combo.currentData(), replace_existing=False)

    def start_model_worker(self, model_name, replace_existing):
        self.load_model_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.browse_models_dir_btn.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)

        self.download_worker = ModelDownloadWorker(model_name, self.settings.models_dir, replace_existing)
        self.download_worker.progress.connect(self.download_progress.setValue)
        self.download_worker.status.connect(self.update_download_status)
        self.download_worker.model_loaded.connect(self.model_ready)
        self.download_worker.corrupt_file.connect(self.corrupt_model_file)
        self.download_worker.error.connect(self.download_error)
        self.download_worker.start()

    def update_download_status(self, message):
        self.model_status_label.setText(message)
        self.model_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")

    def reenable_model_controls(self):
        self.load_model_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.browse_models_dir_btn.setEnabled(True)
        self.download_progress.setVisible(False)

    def model_ready(self, model, model_name):
        # The model was loaded in the worker thread, so the window stays responsive
        self.model = model
        self.model_loaded = True
        self.model_status_label.setText(f"✅ Model '{model_name}' loaded and ready")
        self.model_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.status_label.setText("Ready to transcribe")
        self.status_label.setStyleSheet("color: #4CAF50; font-size: 14px; font-weight: bold;")
        self.start_btn.setEnabled(True)
        self.reenable_model_controls()

    def corrupt_model_file(self, model_name, model_folder):
        self.reenable_model_controls()
        self.model_status_label.setText("❌ Model files are incomplete or damaged")
        self.model_status_label.setStyleSheet("color: #f44336; font-weight: bold;")
        reply = QMessageBox.question(
            self,
            "Damaged Model Files",
            f"Some model files failed their checksum check:\n{model_folder}\n\n"
            "Download new copies of the damaged files?\n"
            "Each damaged file is replaced only after its new download is verified.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.start_model_worker(model_name, replace_existing=True)

    def download_error(self, message):
        self.model_status_label.setText(f"❌ {message}")
        self.model_status_label.setStyleSheet("color: #f44336; font-weight: bold;")
        QMessageBox.critical(self, "Model Error", message)
        self.reenable_model_controls()

    # --- Files and transcription ---------------------------------------------

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Audio Files", "", AUDIO_FILE_FILTER)
        if files:
            skipped = add_unique_files(self.files, files)
            self.file_list.clear()
            self.file_list.addItems([Path(f).name for f in self.files])
            if skipped:
                self.status_label.setText(f"Skipped {skipped} file(s) already in the list")
                self.status_label.setStyleSheet("color: #ff9800; font-size: 14px; font-weight: bold;")

    def clear_files(self):
        self.files = []
        self.file_list.clear()
        self.stats_text.clear()

    def set_controls_enabled(self, enabled):
        for widget in (self.start_btn, self.select_file_btn, self.clear_btn, self.load_model_btn,
                       self.browse_models_dir_btn, self.browse_error_log_btn, self.browse_output_dir_btn,
                       self.model_combo, *self.format_checks.values()):
            widget.setEnabled(enabled)
        self.cancel_btn.setEnabled(not enabled)

    def start_transcription(self):
        if not self.model_loaded:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return

        if not self.files:
            self.status_label.setText("⚠️ Please select at least one file")
            self.status_label.setStyleSheet("color: #ff9800; font-size: 14px; font-weight: bold;")
            return

        if not self.settings.output_dir:
            QMessageBox.warning(self, "Warning", "Please select an output directory for transcriptions!")
            return

        if not self.settings.output_formats:
            QMessageBox.warning(self, "Warning", "Please choose at least one output format under 'Save As'!")
            return

        if not self.settings.error_log_dir:
            reply = QMessageBox.question(
                self,
                "No Error Log Directory",
                "You haven't set an error log directory. Errors will not be saved to a log file.\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.set_controls_enabled(False)
        self.progress_bar.setValue(0)
        self.stats_text.clear()

        self.worker = TranscriptionWorker(
            self.files,
            self.model,
            self.settings.output_dir,
            self.settings.error_log_dir,
            self.settings.output_formats,
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.update_status)
        self.worker.file_complete.connect(self.file_completed)
        self.worker.all_complete.connect(self.all_completed)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(lambda: self.set_controls_enabled(True))
        self.worker.start()

    def update_status(self, message):
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #2196F3; font-size: 14px; font-weight: bold;")

    def file_completed(self, filename, duration, word_count):
        self.stats_text.append(f"✓ {filename}: {duration:.2f}s, {word_count} words")

    def show_error(self, filename, error):
        self.stats_text.append(f"✗ {filename}: ERROR - {error}")

    def all_completed(self, total_time, total_words, successful_files, error_count, cancelled):
        if cancelled:
            self.status_label.setText(f"⏹ Cancelled - {successful_files} of {len(self.files)} file(s) transcribed")
            self.status_label.setStyleSheet("color: #ff9800; font-size: 14px; font-weight: bold;")
        elif error_count == 0:
            self.status_label.setText("✅ All transcriptions complete!")
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 14px; font-weight: bold;")
        else:
            self.status_label.setText(f"⚠️ Completed with {error_count} error(s)")
            self.status_label.setStyleSheet("color: #ff9800; font-size: 14px; font-weight: bold;")

        stats = f"\n{'=' * 50}\n"
        stats += f"Total time: {total_time:.2f}s\n"
        stats += f"Total words: {total_words}\n"
        stats += f"Files processed: {successful_files}/{len(self.files)}\n"
        if successful_files > 0:
            stats += f"Average: {total_time / successful_files:.2f}s per file\n"
        if error_count > 0 and self.settings.error_log_dir:
            stats += f"\n⚠️ {error_count} file(s) failed - check error log in:\n{self.settings.error_log_dir}\n"
        self.stats_text.append(stats)

    def cancel_transcription(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Cancelling...")
            self.status_label.setStyleSheet("color: #ff9800; font-size: 14px; font-weight: bold;")

    # --- Update check --------------------------------------------------------

    def check_for_updates(self):
        if self.update_worker and self.update_worker.isRunning():
            return
        self.version_btn.setText("checking…")
        self.version_btn.setEnabled(False)
        self.update_worker = UpdateCheckWorker()
        self.update_worker.result.connect(self.show_update_result)
        self.update_worker.failed.connect(self.show_update_error)
        self.update_worker.finished.connect(self.update_check_finished)
        self.update_worker.start()

    def update_check_finished(self):
        self.version_btn.setText(VERSION_TEXT)
        self.version_btn.setEnabled(True)

    def show_update_result(self, info):
        self.update_check_finished()
        box = QMessageBox(self)
        box.setWindowTitle("Check for updates")
        offer_page = True
        if info.dev_build:
            box.setText("This is a development build.")
            box.setInformativeText(f"You are running {VERSION_TEXT}. The latest release is {info.latest}.")
        elif info.newer:
            box.setText(f"{APP_NAME} {info.latest} is available.")
            box.setInformativeText(f"You have {VERSION_TEXT}. Open the release page to download the new version.")
            self.version_btn.setStyleSheet(VERSION_STYLE.format(color=VERSION_ACCENT))
            self.version_btn.setToolTip(f"{info.latest} is available. Click for details.")
        else:
            box.setText(f"{APP_NAME} is up to date.")
            box.setInformativeText(f"{info.latest} is the latest release.")
            offer_page = False

        open_btn = None
        if offer_page:
            open_btn = box.addButton("Open release page", QMessageBox.ButtonRole.AcceptRole)
            if info.notes:
                box.setDetailedText(info.notes)
        close_btn = box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(open_btn or close_btn)
        box.exec()
        if open_btn is not None and box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl(info.url))

    def show_update_error(self, message):
        self.update_check_finished()
        QMessageBox.warning(self, "Check for updates", message)


def self_test() -> int:
    """Used by the release workflow: checks that a build includes faster-whisper, CTranslate2,
    the audio decoder and the voice-activity model."""
    import tempfile
    import wave

    import ctranslate2
    import numpy as np
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import get_speech_timestamps

    if "int8" not in ctranslate2.get_supported_compute_types("cpu"):
        raise RuntimeError("CTranslate2 has no int8 support on this CPU")

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "silence.wav"
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b"\x00\x00" * 16000)
        audio = decode_audio(str(path))
    if len(audio) != 16000:
        raise RuntimeError(f"Audio decoder returned {len(audio)} samples, expected 16000")

    get_speech_timestamps(np.zeros(16000, dtype=np.float32))
    print(f"{APP_NAME} {VERSION_TEXT} self-test passed")
    return 0


def main():
    if "--self-test" in sys.argv:
        # Catch everything: an uncaught error in a windowed Windows build opens a dialog and stalls CI
        try:
            code = self_test()
        except Exception:
            traceback.print_exc()
            code = 1
        sys.exit(code)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(app_icon())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
