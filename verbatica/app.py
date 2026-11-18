"""The Verbatica main window: a queue of recordings, each with its waveform, the model in the
header and where to save in the footer. Folders set once and the update check live in a
separate Settings window."""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import traceback
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QDesktopServices, QGuiApplication, QIcon, QKeySequence, QPixmap
from PyQt6.QtWidgets import (QApplication, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                             QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
                             QStackedWidget, QVBoxLayout, QWidget)

from verbatica import APP_NAME, __version__, theme, update_check
from verbatica._icon import ICON_PNGS
from verbatica.models import DEFAULT_MODEL, MODEL_SIZES, MODELS, ModelDownloadWorker, is_downloaded
from verbatica.settings import SettingsStore
from verbatica.transcription import (AUDIO_EXTENSIONS, AUDIO_FILE_FILTER, OUTPUT_FORMATS,
                                     TranscriptionWorker, add_unique_files)
from verbatica.widgets import ElidedLabel, FileRow, Glyph, ModelPopup, format_taken

VERSION_TEXT = f"v{__version__}"

# Short names for the format chips in the footer
FORMAT_CHIPS = {".txt": "Text", ".srt": "SRT", ".vtt": "VTT"}


def app_icon() -> QIcon:
    """The window and taskbar icon, built from PNG data inside the package (no files to ship)."""
    icon = QIcon()
    for data in ICON_PNGS.values():
        pixmap = QPixmap()
        pixmap.loadFromData(base64.b64decode(data), "PNG")
        icon.addPixmap(pixmap)
    return icon


def display_path(path: str) -> str:
    """A folder as people read it: ~/Documents/Transcripts instead of the full home path."""
    if not path:
        return "Not set"
    home = str(Path.home())
    if sys.platform != "win32" and (path == home or path.startswith(home + os.sep)):
        return "~" + path[len(home):]
    return path


def reveal_in_file_manager(path: str):
    """Open the file manager with the file selected (only the folder on Linux)."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path])
    elif sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))


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


def label(text: str, name: str = "") -> QLabel:
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    return widget


def button(text: str, name: str = "", slot=None) -> QPushButton:
    widget = QPushButton(text)
    if name:
        widget.setObjectName(name)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    if slot:
        widget.clicked.connect(slot)
    return widget


class PathField(QWidget):
    """A folder shown in a well with a folder icon, and a Change… button."""

    def __init__(self, window: "MainWindow", path: str, on_change, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        well = QFrame()
        well.setObjectName("well")
        well.setFixedHeight(28)
        inner = QHBoxLayout(well)
        inner.setContentsMargins(8, 0, 8, 0)
        inner.setSpacing(8)
        self.icon = QLabel()
        inner.addWidget(self.icon)
        self.text = ElidedLabel()
        inner.addWidget(self.text, 1)
        layout.addWidget(well, 1)
        self.change_btn = button("Change…", slot=on_change)
        layout.addWidget(self.change_btn)
        window.on_theme(lambda t: self.icon.setPixmap(theme.icon("folder", t.muted).pixmap(16, 16)))
        self.set_path(path)

    def set_path(self, path: str):
        self.text.setText(display_path(path))
        self.text.setToolTip(path)


class SettingsWindow(QDialog):
    """Things set once: where models and error logs go, and the update check."""

    def __init__(self, window: "MainWindow"):
        super().__init__(window)
        self.setObjectName("settings")
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        grid = QGridLayout(self)
        grid.setContentsMargins(24, 24, 24, 24)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        grid.setColumnMinimumWidth(0, 112)
        grid.setColumnStretch(1, 1)
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        grid.addWidget(label("Models folder"), 0, 0, right)
        self.models_field = PathField(window, window.settings.models_dir, window.select_models_dir)
        grid.addWidget(self.models_field, 0, 1)
        grid.addWidget(label("Downloaded models are kept here", "caption"), 1, 1)
        grid.setRowMinimumHeight(2, 16)

        grid.addWidget(label("Error logs"), 3, 0, right)
        self.logs_field = PathField(window, window.settings.error_log_dir, window.select_error_log_dir)
        grid.addWidget(self.logs_field, 3, 1)
        grid.addWidget(label("A file that fails gets a line in a log here", "caption"), 4, 1)
        grid.setRowMinimumHeight(5, 20)

        rule = QFrame()
        rule.setObjectName("rule")
        grid.addWidget(rule, 6, 0, 1, 2)
        grid.setRowMinimumHeight(7, 20)

        grid.addWidget(label("Updates"), 8, 0, right)
        row = QHBoxLayout()
        row.setSpacing(12)
        self.update_btn = button("Check for updates", slot=window.check_for_updates)
        row.addWidget(self.update_btn)
        self.version_label = label(VERSION_TEXT, "secondary")
        self.version_label.setFont(theme.figures(self.version_label.font()))
        row.addWidget(self.version_label)
        self.update_note = label("", "updateNote")
        row.addWidget(self.update_note)
        row.addStretch()
        grid.addLayout(row, 8, 1)
        grid.addWidget(label("One request to GitHub, only when you click", "caption"), 9, 1)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.refresh(window)

    def showEvent(self, event):
        super().showEvent(event)
        self.setFocus()  # no focus ring on the first button when the window opens

    def refresh(self, window: "MainWindow"):
        self.models_field.set_path(window.settings.models_dir)
        self.logs_field.set_path(window.settings.error_log_dir)
        self.models_field.change_btn.setEnabled(window.download_worker is None)
        checking = window.update_worker is not None and window.update_worker.isRunning()
        self.update_btn.setText("Checking…" if checking else "Check for updates")
        self.update_btn.setEnabled(not checking)
        self.update_note.setText(window.update_note)
        self.update_note.setVisible(bool(window.update_note))


class MainWindow(QMainWindow):
    def __init__(self, store: SettingsStore | None = None, dark: bool | None = None):
        super().__init__()
        self.rows: list[FileRow] = []
        self.batch_rows: list[FileRow] = []
        self.worker = None
        self.download_worker = None
        self.update_worker = None
        self.model = None
        self.model_loaded = False
        self.loaded_model_name = ""
        self.model_status = ""
        self.update_note = ""
        self.summary = ""
        self.settings_warning_shown = False
        self.settings_window = None
        self.forced_dark = dark
        self.store = store or SettingsStore()
        self.settings = self.store.load()
        self._theme_hooks = []
        self.tokens = theme.apply(QApplication.instance(), dark)

        self.init_ui()
        self.init_menus()
        self.refresh()
        self.centralWidget().setFocus()
        if dark is None:
            QGuiApplication.styleHints().colorSchemeChanged.connect(lambda _scheme: self.apply_theme())

    # --- Building the window -------------------------------------------------

    def on_theme(self, hook):
        """Run `hook(tokens)` now and again whenever the system appearance changes."""
        self._theme_hooks.append(hook)
        hook(self.tokens)

    def apply_theme(self):
        self.tokens = theme.apply(QApplication.instance(), self.forced_dark)
        for hook in self._theme_hooks:
            hook(self.tokens)
        for row in self.rows:
            row.retheme(self.tokens)

    def init_ui(self):
        self.setWindowTitle(APP_NAME)
        self.resize(860, 640)
        self.setMinimumSize(680, 480)
        self.setAcceptDrops(True)

        central = QWidget()
        central.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # no focus ring on a control at launch
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self.build_header())

        body = QVBoxLayout()
        body.setContentsMargins(16, 0, 16, 12)
        body.setSpacing(8)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_empty_view())
        self.stack.addWidget(self.build_queue_view())
        body.addWidget(self.stack, 1)

        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        add_row.setSpacing(4)
        self.add_btn = button("Add files", "quiet", self.select_files)
        self.on_theme(lambda t: self.add_btn.setIcon(theme.icon("plus", t.text_2)))
        add_row.addWidget(self.add_btn)
        add_row.addStretch()
        self.clear_btn = button("Clear", "quiet", self.clear_files)
        add_row.addWidget(self.clear_btn)
        self.add_row = QWidget()
        self.add_row.setLayout(add_row)
        body.addWidget(self.add_row)
        root.addLayout(body, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("thin")
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        root.addWidget(self.build_footer())

    def build_header(self) -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 16, 16, 16)
        layout.setSpacing(8)

        layout.addWidget(label("Queue", "title"))
        self.count_label = label("", "meta")
        self.count_label.setFont(theme.figures(self.count_label.font()))
        layout.addWidget(self.count_label, alignment=Qt.AlignmentFlag.AlignBottom)
        layout.addStretch()

        layout.addWidget(label("Model", "secondary"))
        self.model_combo = ModelPopup()
        for name in MODELS:
            self.model_combo.addItem(f"{name}  ·  {MODEL_SIZES[name].lstrip('~')}", name)
        self.model_combo.setCurrentIndex(self.model_combo.findData(DEFAULT_MODEL))
        self.model_combo.setAccessibleName("Model")
        self.model_combo.currentIndexChanged.connect(lambda _i: self.refresh())
        self.on_theme(lambda t: setattr(self.model_combo, "tokens", t))
        layout.addWidget(self.model_combo)
        layout.addSpacing(4)

        self.state_dot = QLabel()
        self.state_dot.setObjectName("stateDot")
        self.state_dot.setFixedSize(6, 6)
        layout.addWidget(self.state_dot)
        self.state_label = label("", "secondary")
        self.state_label.setFont(theme.figures(self.state_label.font()))
        layout.addWidget(self.state_label)
        self.load_model_btn = button("Load", slot=self.download_and_load_model)
        layout.addWidget(self.load_model_btn)
        return header

    def build_empty_view(self) -> QWidget:
        self.empty_card = QFrame()
        self.empty_card.setObjectName("emptyCard")
        layout = QVBoxLayout(self.empty_card)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(0)
        layout.addStretch()
        self.glyph = Glyph()
        self.on_theme(lambda t: (setattr(self.glyph, "tokens", t), self.glyph.update()))
        layout.addWidget(self.glyph, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(16)
        layout.addWidget(label("Drop recordings here", "emptyTitle"), alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(8)
        body = label(f"{APP_NAME} writes what was said into text and subtitle files.\n"
                     "Nothing leaves this computer.", "emptyBody")
        body.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(body, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(20)
        self.empty_add_btn = button("Add files…", slot=self.select_files)
        layout.addWidget(self.empty_add_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(12)
        kinds = ", ".join(e.upper() for e in AUDIO_EXTENSIONS)
        layout.addWidget(label(f"{kinds} and other audio or video", "caption"),
                         alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        return self.empty_card

    def build_queue_view(self) -> QWidget:
        self.queue_card = QFrame()
        self.queue_card.setObjectName("card")
        layout = QVBoxLayout(self.queue_card)
        layout.setContentsMargins(1, 1, 1, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rows = QWidget()
        rows.setObjectName("rows")
        self.rows_layout = QVBoxLayout(rows)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(0)
        self.rows_layout.addStretch()
        scroll.setWidget(rows)
        layout.addWidget(scroll)
        return self.queue_card

    def build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("footer")
        grid = QGridLayout(footer)
        grid.setContentsMargins(20, 12, 16, 16)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        grid.addWidget(label("Save to", "secondary"), 0, 0, left)
        save_row = QHBoxLayout()
        save_row.setSpacing(8)
        self.output_icon = QLabel()
        self.on_theme(lambda t: self.output_icon.setPixmap(theme.icon("folder", t.muted).pixmap(16, 16)))
        save_row.addWidget(self.output_icon)
        self.output_label = ElidedLabel()
        save_row.addWidget(self.output_label)
        self.browse_output_dir_btn = button("Change…", "quiet", self.select_output_dir)
        save_row.addWidget(self.browse_output_dir_btn)
        save_row.addStretch()
        grid.addLayout(save_row, 0, 1)

        grid.addWidget(label("Formats", "secondary"), 1, 0, left)
        chips = QHBoxLayout()
        chips.setSpacing(6)
        self.format_checks = {}
        for ext, name in FORMAT_CHIPS.items():
            chip = button(name, "chip")
            chip.setCheckable(True)
            chip.setChecked(ext in self.settings.output_formats)
            chip.setToolTip(OUTPUT_FORMATS[ext])
            chip.setIconSize(QSize(12, 12))
            chip.toggled.connect(self.formats_changed)
            chips.addWidget(chip)
            self.format_checks[ext] = chip
        self.on_theme(lambda t: self.paint_chips())
        chips.addStretch()
        grid.addLayout(chips, 1, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.hint_label = label("", "hint")
        self.hint_label.setFont(theme.figures(self.hint_label.font()))
        actions.addWidget(self.hint_label)
        actions.addSpacing(4)
        self.cancel_btn = button("Stop", slot=self.cancel_transcription)
        actions.addWidget(self.cancel_btn)
        self.start_btn = button("Transcribe", "primary", self.start_transcription)
        actions.addWidget(self.start_btn)
        grid.addLayout(actions, 1, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return footer

    def init_menus(self):
        menu_bar = self.menuBar()

        def action(text, slot, shortcut=None, role=None) -> QAction:
            item = QAction(text, self)
            item.triggered.connect(slot)
            if shortcut is not None:
                item.setShortcut(QKeySequence(shortcut))
            if role is not None:
                item.setMenuRole(role)
            return item

        file_menu = menu_bar.addMenu("File")
        self.add_action = action("Add Files…", self.select_files, QKeySequence.StandardKey.Open)
        file_menu.addAction(self.add_action)
        file_menu.addSeparator()
        # On macOS Qt moves these two into the Verbatica menu
        file_menu.addAction(action("Settings…", self.show_settings, QKeySequence.StandardKey.Preferences,
                                   QAction.MenuRole.PreferencesRole))
        file_menu.addAction(action("Check for Updates…", self.check_for_updates, None,
                                   QAction.MenuRole.ApplicationSpecificRole))
        file_menu.addSeparator()
        file_menu.addAction(action("Close Window", self.close, QKeySequence.StandardKey.Close))
        if sys.platform != "darwin":
            file_menu.addAction(action("Quit", QApplication.quit, QKeySequence.StandardKey.Quit,
                                       QAction.MenuRole.QuitRole))

        queue_menu = menu_bar.addMenu("Queue")
        self.start_action = action("Transcribe", self.start_transcription, "Ctrl+Return")
        self.cancel_action = action("Stop", self.cancel_transcription, "Ctrl+.")
        self.clear_done_action = action("Remove Finished Files", self.clear_done_files)
        self.clear_action = action("Remove All Files", self.clear_files)
        for item in (self.start_action, self.cancel_action):
            queue_menu.addAction(item)
        queue_menu.addSeparator()
        queue_menu.addAction(self.clear_done_action)
        queue_menu.addAction(self.clear_action)

    def paint_chips(self):
        for chip in self.format_checks.values():
            chip.setIcon(theme.icon("check-small", self.tokens.text) if chip.isChecked() else QIcon())

    # --- One place that sets what is shown and enabled -----------------------

    def refresh(self):
        running = self.worker is not None and self.worker.isRunning()
        downloading = self.download_worker is not None
        has_rows = bool(self.rows)
        pending = [r for r in self.rows if r.state in ("waiting", "failed", "stopped")]

        self.stack.setCurrentIndex(1 if has_rows else 0)
        self.add_row.setVisible(has_rows)
        done = sum(r.state == "done" for r in self.rows)
        files = f"{len(self.rows)} file{'s' if len(self.rows) != 1 else ''}"
        self.count_label.setText(f"{files}  ·  {done} done" if done else files if has_rows else "")

        # Model
        selected = self.model_combo.currentData()
        self.model_combo.setEnabled(not downloading)
        self.load_model_btn.setEnabled(not downloading)
        if downloading:
            self.set_state("busy", self.model_status or "Starting")
            self.load_model_btn.setVisible(False)
        elif self.model_loaded and self.loaded_model_name == selected:
            self.set_state("ready", "Loaded")
            self.load_model_btn.setVisible(False)
        else:
            if self.model_loaded:
                self.set_state("ready", f"{self.loaded_model_name} loaded")
            else:
                self.set_state(None, "")
            downloaded = bool(self.settings.models_dir) and is_downloaded(self.settings.models_dir, selected)
            self.load_model_btn.setText("Load" if downloaded else f"Download {MODEL_SIZES[selected].lstrip('~')}")
            self.load_model_btn.setVisible(True)

        # Footer
        self.output_label.setText(display_path(self.settings.output_dir) if self.settings.output_dir
                                  else "Choose a folder")
        self.output_label.setToolTip(self.settings.output_dir)
        for widget in (self.browse_output_dir_btn, *self.format_checks.values()):
            widget.setEnabled(not running)
        self.clear_btn.setEnabled(not running)
        self.clear_action.setEnabled(not running and has_rows)
        self.clear_done_action.setEnabled(not running and any(r.state == "done" for r in self.rows))
        for row in self.rows:
            row.set_busy(running)

        ready = (self.model_loaded and bool(pending) and bool(self.settings.output_formats)
                 and bool(self.settings.output_dir) and not running)
        self.start_btn.setEnabled(ready)
        self.start_action.setEnabled(ready)
        self.cancel_btn.setVisible(running)
        self.cancel_action.setEnabled(running)
        self.progress_bar.setVisible(running)

        if running:
            index = next((i for i, r in enumerate(self.batch_rows) if r.state == "active"), 0)
            hint = f"File {index + 1} of {len(self.batch_rows)}"
        elif not has_rows:
            hint = ""
        elif not self.model_loaded:
            hint = "Load a model to start"
        elif not self.settings.output_dir:
            hint = "Choose where to save"
        elif not self.settings.output_formats:
            hint = "Choose a format"
        elif not pending:
            hint = self.summary or "All files transcribed"
        else:
            hint = self.summary
        self.hint_label.setText(hint)

        if self.settings_window is not None:
            self.settings_window.refresh(self)

    def set_state(self, state, text: str):
        self.state_dot.setVisible(state is not None)
        self.state_label.setVisible(bool(text))
        self.state_label.setText(text)
        if state is not None:
            self.state_dot.setProperty("state", state)
            self.state_dot.style().unpolish(self.state_dot)
            self.state_dot.style().polish(self.state_dot)

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

    def show_settings(self):
        if self.settings_window is None:
            self.settings_window = SettingsWindow(self)
        self.settings_window.refresh(self)
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def select_models_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose Models Folder", self.settings.models_dir)
        if directory:
            self.settings.models_dir = directory
            self.save_settings()
            self.refresh()

    def select_error_log_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose Error Log Folder", self.settings.error_log_dir)
        if directory:
            self.settings.error_log_dir = directory
            self.save_settings()
            self.refresh()

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose Where to Save Transcripts",
                                                     self.settings.output_dir)
        if directory:
            self.settings.output_dir = directory
            self.save_settings()
            self.refresh()

    def formats_changed(self):
        self.settings.output_formats = [ext for ext, check in self.format_checks.items() if check.isChecked()]
        self.paint_chips()
        self.save_settings()
        self.refresh()

    # --- Model ---------------------------------------------------------------

    def download_and_load_model(self):
        if not self.settings.models_dir:
            QMessageBox.warning(self, "No Models Folder", "Choose a models folder in Settings first.")
            return
        self.start_model_worker(self.model_combo.currentData(), replace_existing=False)

    def start_model_worker(self, model_name, replace_existing):
        self.model_status = "Starting"
        self.download_worker = ModelDownloadWorker(model_name, self.settings.models_dir, replace_existing)
        self.download_worker.status.connect(self.update_download_status)
        self.download_worker.model_loaded.connect(self.model_ready)
        self.download_worker.corrupt_file.connect(self.corrupt_model_file)
        self.download_worker.error.connect(self.download_error)
        self.download_worker.start()
        self.refresh()

    def update_download_status(self, message):
        self.model_status = message
        self.refresh()

    def model_worker_done(self):
        self.download_worker = None
        self.model_status = ""

    def model_ready(self, model, model_name):
        # The model was loaded in the worker thread, so the window stays responsive
        self.model = model
        self.model_loaded = True
        self.loaded_model_name = model_name
        self.model_worker_done()
        self.refresh()

    def corrupt_model_file(self, model_name, model_folder):
        self.model_worker_done()
        self.refresh()
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
        self.model_worker_done()
        self.refresh()
        QMessageBox.critical(self, "Model Not Loaded", message)

    # --- Queue ---------------------------------------------------------------

    @property
    def files(self) -> list:
        return [row.path for row in self.rows]

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add Files", "", AUDIO_FILE_FILTER)
        if files:
            self.add_files(files)

    def add_files(self, paths):
        known = self.files
        skipped = add_unique_files(known, paths)
        for path in known[len(self.rows):]:
            row = FileRow(path)
            row.remove_requested.connect(self.remove_row)
            row.reveal_requested.connect(lambda r: reveal_in_file_manager(r.output_file))
            row.setProperty("first", not self.rows)
            self.rows_layout.insertWidget(self.rows_layout.count() - 1, row)
            self.rows.append(row)
        self.summary = f"Skipped {skipped} file{'s' if skipped != 1 else ''} already in the queue" if skipped else ""
        self.refresh()

    def remove_rows(self, rows):
        """Take rows out of the queue. Only the list changes: transcripts already saved stay on disk."""
        if self.worker is not None and self.worker.isRunning():
            return
        for row in list(rows):
            self.rows.remove(row)
            row.setParent(None)
            row.deleteLater()
        if self.rows:
            self.rows[0].setProperty("first", True)
            self.rows[0].style().polish(self.rows[0])
        else:
            self.summary = ""
        self.refresh()

    def remove_row(self, row: FileRow):
        self.remove_rows([row])

    def clear_done_files(self):
        self.remove_rows([r for r in self.rows if r.state == "done"])

    def clear_files(self):
        self.remove_rows(self.rows)

    def dragEnterEvent(self, event):
        if any(url.isLocalFile() and Path(url.toLocalFile()).is_file() for url in event.mimeData().urls()):
            event.acceptProposedAction()
            self.set_dropping(True)

    def dragLeaveEvent(self, event):
        self.set_dropping(False)

    def dropEvent(self, event):
        self.set_dropping(False)
        paths = [url.toLocalFile() for url in event.mimeData().urls()
                 if url.isLocalFile() and Path(url.toLocalFile()).is_file()]
        if paths:
            event.acceptProposedAction()
            self.add_files(paths)

    def set_dropping(self, on: bool):
        for card in (self.empty_card, self.queue_card):
            card.setProperty("dropping", on)
            card.style().unpolish(card)
            card.style().polish(card)

    # --- Transcription -------------------------------------------------------

    def start_transcription(self):
        pending = [r for r in self.rows if r.state in ("waiting", "failed", "stopped")]
        if not self.model_loaded or not pending or self.worker is not None and self.worker.isRunning():
            return
        if not self.settings.output_dir or not self.settings.output_formats:
            return

        if not self.settings.error_log_dir:
            reply = QMessageBox.question(
                self,
                "No Error Log Folder",
                "No error log folder is set, so errors will not be saved to a log file.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self.batch_rows = pending
        for row in pending:
            row.waveform.set_fraction(0, animate=False)
            row.set_state("waiting")
        self.summary = ""
        self.progress_bar.setValue(0)

        self.worker = TranscriptionWorker(
            [row.path for row in pending],
            self.model,
            self.settings.output_dir,
            self.settings.error_log_dir,
            self.settings.output_formats,
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.file_started.connect(self.file_started)
        self.worker.file_audio.connect(lambda i, peaks, duration: self.batch_rows[i].set_audio(peaks, duration))
        self.worker.file_progress.connect(lambda i, fraction: self.batch_rows[i].set_progress(fraction))
        self.worker.file_complete.connect(self.file_completed)
        self.worker.error.connect(self.show_error)
        self.worker.all_complete.connect(self.all_completed)
        self.worker.finished.connect(self.refresh)
        self.worker.start()
        self.refresh()

    def file_started(self, index):
        self.batch_rows[index].set_state("active")
        self.refresh()

    def file_completed(self, index, output_base, seconds_taken, word_count):
        self.batch_rows[index].set_done(output_base, self.settings.output_formats, seconds_taken, word_count)
        self.refresh()

    def show_error(self, index, filename, error):
        if index < 0:  # the error log itself could not be written
            QMessageBox.warning(self, "Error Log Not Saved", error)
            return
        self.batch_rows[index].set_state("failed", error)
        self.refresh()

    def all_completed(self, total_time, total_words, successful_files, error_count, cancelled):
        for row in self.batch_rows:
            if row.state == "active":
                row.set_state("stopped")
        if cancelled:
            self.summary = f"Stopped after {successful_files} of {len(self.batch_rows)}"
        elif error_count:
            where = " · see the error log" if self.settings.error_log_dir else ""
            self.summary = f"{error_count} failed{where}"
        else:
            self.summary = f"{total_words:,} words in {format_taken(total_time)}"
        self.refresh()

    def cancel_transcription(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.hint_label.setText("Stopping after this part")

    # --- Update check --------------------------------------------------------

    def check_for_updates(self):
        if self.update_worker and self.update_worker.isRunning():
            return
        self.update_worker = UpdateCheckWorker()
        self.update_worker.result.connect(self.show_update_result)
        self.update_worker.failed.connect(self.show_update_error)
        self.update_worker.finished.connect(self.refresh)
        self.update_worker.start()
        self.refresh()

    def show_update_result(self, info):
        box = QMessageBox(self)
        box.setWindowTitle("Check for Updates")
        offer_page = True
        if info.dev_build:
            box.setText("This is a development build.")
            box.setInformativeText(f"You are running {VERSION_TEXT}. The latest release is {info.latest}.")
        elif info.newer:
            box.setText(f"{APP_NAME} {info.latest} is available.")
            box.setInformativeText(f"You have {VERSION_TEXT}. Open the release page to download the new version.")
            self.update_note = f"{info.latest} is available"
        else:
            box.setText(f"{APP_NAME} is up to date.")
            box.setInformativeText(f"{info.latest} is the latest release.")
            offer_page = False
        self.refresh()

        open_btn = None
        if offer_page:
            open_btn = box.addButton("Open Release Page", QMessageBox.ButtonRole.AcceptRole)
            if info.notes:
                box.setDetailedText(info.notes)
        close_btn = box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(open_btn or close_btn)
        box.exec()
        if open_btn is not None and box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl(info.url))

    def show_update_error(self, message):
        self.refresh()
        QMessageBox.warning(self, "Check for Updates", message)


def self_test() -> int:
    """Used by the release workflow: checks that a build includes faster-whisper, CTranslate2,
    the audio decoder, the voice-activity model and the SVG renderer for the icons."""
    import tempfile
    import wave

    import ctranslate2
    import numpy as np
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import get_speech_timestamps
    from PyQt6.QtSvg import QSvgRenderer  # noqa: F401

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
