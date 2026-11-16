"""Tests for the main window: the queue, the Transcribe button's conditions, the Settings
window, the update check and settings saved next to the app."""

from PyQt6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PyQt6.QtGui import QDropEvent
from PyQt6.QtWidgets import QMessageBox

from conftest import FakeModel

from verbatica import __version__, app, theme
from verbatica.settings import SettingsStore
from verbatica.update_check import UpdateInfo


def make_window(tmp_path, dark=True):
    return app.MainWindow(SettingsStore(tmp_path / "app", legacy_file=tmp_path / "none.json"), dark=dark)


def loaded_window(tmp_path, files=("a.mp3", "b.mp3")):
    window = make_window(tmp_path)
    window.settings.output_dir = str(tmp_path / "out")
    window.model_ready(FakeModel([(0, 30, " Hello."), (30, 60, " Bye.")]), "small")
    paths = []
    for name in files:
        path = tmp_path / name
        path.write_text("x")
        paths.append(str(path))
    window.add_files(paths)
    return window


def run_batch(window):
    window.start_transcription()
    window.worker.wait()
    from conftest import qt_app
    qt_app.processEvents()


def test_window_icon_loads_from_the_package_at_every_size():
    icon = app.app_icon()
    sizes = sorted(size.width() for size in icon.availableSizes())
    assert sizes == [16, 32, 64, 256]


def test_both_appearances_build_a_stylesheet_and_palette():
    for tokens in (theme.DARK, theme.LIGHT):
        sheet = theme.stylesheet(tokens)
        assert tokens.accent in sheet and "{" in sheet
        assert theme.palette(tokens).color(theme.QPalette.ColorRole.Window).name() == tokens.window


def test_settings_window_shows_the_version_and_folders(tmp_path):
    window = make_window(tmp_path)
    window.show_settings()
    settings = window.settings_window
    assert settings.version_label.text() == f"v{__version__}"
    assert settings.models_field.text.toolTip() == window.settings.models_dir
    assert settings.update_btn.text() == "Check for updates"


def test_choosing_a_format_saves_settings_in_the_data_folder(tmp_path):
    window = make_window(tmp_path)
    window.format_checks[".srt"].setChecked(True)
    saved = (tmp_path / "app" / "Verbatica-data" / "settings.json").read_text()
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


def test_newer_release_shows_in_settings(tmp_path, monkeypatch):
    window = make_window(tmp_path)
    window.show_settings()
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    window.show_update_result(UpdateInfo("1.0.0", "v1.1.0", "https://github.com/x", ""))
    assert window.settings_window.update_note.text() == "v1.1.0 is available"
    assert window.settings_window.update_btn.isEnabled()


def test_empty_queue_shows_the_drop_view(tmp_path):
    window = make_window(tmp_path)
    assert window.stack.currentIndex() == 0
    assert not window.start_btn.isEnabled()


def test_transcribe_needs_a_model_files_and_a_format(tmp_path):
    window = make_window(tmp_path)
    window.settings.output_dir = str(tmp_path / "out")
    (tmp_path / "a.mp3").write_text("x")
    window.add_files([str(tmp_path / "a.mp3")])
    assert window.stack.currentIndex() == 1
    assert not window.start_btn.isEnabled()
    assert window.hint_label.text() == "Load a model to start"

    window.model_ready(FakeModel(), "small")
    assert window.start_btn.isEnabled()
    assert window.state_label.text() == "small loaded"  # the menu still shows the default, medium
    window.model_combo.setCurrentIndex(window.model_combo.findData("small"))
    assert window.state_label.text() == "Loaded"
    assert not window.load_model_btn.isVisible()

    for chip in window.format_checks.values():
        chip.setChecked(False)
    assert not window.start_btn.isEnabled()
    assert window.hint_label.text() == "Choose a format"


def test_load_button_offers_a_download_for_a_missing_model(tmp_path):
    window = make_window(tmp_path)
    window.model_combo.setCurrentIndex(window.model_combo.findData("tiny"))
    assert window.load_model_btn.text().startswith("Download ")


def test_adding_the_same_file_twice_keeps_one_row(tmp_path):
    window = loaded_window(tmp_path, files=("a.mp3",))
    window.add_files([str(tmp_path / "a.mp3")])
    assert len(window.rows) == 1
    assert window.hint_label.text() == "Skipped 1 file already in the queue"


def test_dropping_files_adds_them_to_the_queue(tmp_path):
    window = make_window(tmp_path)
    (tmp_path / "talk.wav").write_text("x")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(tmp_path / "talk.wav")), QUrl.fromLocalFile(str(tmp_path))])
    event = QDropEvent(QPointF(QPoint(10, 10)), Qt.DropAction.CopyAction, mime,
                       Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    window.dropEvent(event)
    assert window.files == [str(tmp_path / "talk.wav")]  # the folder is ignored


def test_a_batch_marks_rows_done_with_their_word_counts(tmp_path):
    window = loaded_window(tmp_path)
    run_batch(window)
    assert [row.state for row in window.rows] == ["done", "done"]
    assert window.rows[0].result_label.text() == "2 words"
    assert window.rows[0].output_file == str(tmp_path / "out" / "a.txt")
    assert window.rows[0].waveform.peaks  # the waveform arrived
    assert window.count_label.text() == "2 files  ·  2 done"
    assert not window.start_btn.isEnabled()  # nothing left to transcribe


def test_a_second_run_only_transcribes_files_not_done(tmp_path):
    window = loaded_window(tmp_path, files=("a.mp3",))
    run_batch(window)
    (tmp_path / "b.mp3").write_text("x")
    window.add_files([str(tmp_path / "b.mp3")])
    run_batch(window)
    assert window.model.calls[-1][0] == str(tmp_path / "b.mp3")
    assert len(window.model.calls) == 2
    assert not (tmp_path / "out" / "a (2).txt").exists()


def test_a_failed_file_is_marked_and_can_run_again(tmp_path):
    window = loaded_window(tmp_path)
    window.model.fail = lambda path: path.endswith("a.mp3")
    run_batch(window)
    assert [row.state for row in window.rows] == ["failed", "done"]
    assert "cannot decode" in window.rows[0].result_label.toolTip()
    assert window.start_btn.isEnabled()


def test_rows_can_be_removed_and_cleared(tmp_path):
    window = loaded_window(tmp_path)
    window.remove_row(window.rows[0])
    assert window.files == [str(tmp_path / "b.mp3")]
    window.clear_files()
    assert window.rows == [] and window.stack.currentIndex() == 0


def test_long_paths_from_home_are_shortened(monkeypatch, tmp_path):
    monkeypatch.setattr(app.Path, "home", lambda: tmp_path)
    if app.sys.platform != "win32":
        assert app.display_path(str(tmp_path / "Documents")) == "~/Documents"
    assert app.display_path("") == "Not set"


def test_self_test_with_real_faster_whisper():
    """Decodes real audio and runs the voice-activity model. Skipped where faster-whisper
    isn't installed (the CI test job); the release builds run the same check."""
    import pytest
    pytest.importorskip("faster_whisper")
    assert app.self_test() == 0
