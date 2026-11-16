"""Tests for the main window: the version button and settings saved next to the app."""

from PyQt6.QtWidgets import QMessageBox

from audio_transcriber import __version__, app
from audio_transcriber.settings import SettingsStore
from audio_transcriber.update_check import UpdateInfo


def make_window(tmp_path):
    return app.MainWindow(SettingsStore(tmp_path / "app", legacy_file=tmp_path / "none.json"))


def test_version_button_shows_the_version_and_comes_last_in_tab_order(tmp_path):
    window = make_window(tmp_path)
    button = window.version_btn
    assert button.text() == f"v{__version__}"
    assert button.toolTip() == "Check for updates"
    assert button.accessibleName() == f"Version v{__version__}"
    assert window.cancel_btn.nextInFocusChain() is button


def test_choosing_a_format_saves_settings_in_the_data_folder(tmp_path):
    window = make_window(tmp_path)
    window.format_checks[".srt"].setChecked(True)
    saved = (tmp_path / "app" / "Audio-Transcriber-data" / "settings.json").read_text()
    assert ".srt" in saved


def test_unwritable_data_folder_warns_once(tmp_path, monkeypatch):
    window = make_window(tmp_path)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[1]))

    def fail(settings):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(window.store, "save", fail)
    window.format_checks[".srt"].setChecked(True)
    window.format_checks[".vtt"].setChecked(True)
    assert warnings == ["Settings Not Saved"]


def test_newer_release_tints_the_version(tmp_path, monkeypatch):
    window = make_window(tmp_path)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    window.show_update_result(UpdateInfo("1.0.0", "v1.1.0", "https://github.com/x", ""))
    assert app.VERSION_ACCENT in window.version_btn.styleSheet()
    assert window.version_btn.toolTip() == "v1.1.0 is available. Click for details."
    assert window.version_btn.isEnabled()
