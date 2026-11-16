"""Tests for the data folder next to the app and the settings stored in it."""

import json
import sys

from verbatica import settings as st


def make_store(tmp_path, legacy=None):
    legacy_file = tmp_path / "home" / ".whisper_gui_settings.json"
    if legacy is not None:
        legacy_file.parent.mkdir(parents=True)
        legacy_file.write_text(json.dumps(legacy))
    return st.SettingsStore(tmp_path / "app", legacy_file=legacy_file)


def test_defaults_point_inside_the_data_folder(tmp_path):
    store = make_store(tmp_path)
    s = store.load()
    data = tmp_path / "app" / "Verbatica-data"
    assert s.models_dir == str(data / "models")
    assert s.error_log_dir == str(data / "logs")
    assert s.output_formats == [".txt"]
    assert not data.exists()  # nothing is created until something is saved


def test_settings_round_trip(tmp_path):
    store = make_store(tmp_path)
    s = store.load()
    s.output_dir = str(tmp_path / "elsewhere")
    s.output_formats = [".srt", ".vtt"]
    store.save(s)
    assert store.file == tmp_path / "app" / "Verbatica-data" / "settings.json"
    loaded = store.load()
    assert loaded.output_dir == str((tmp_path / "elsewhere").resolve())
    assert loaded.output_formats == [".srt", ".vtt"]


def test_paths_inside_the_app_folder_are_stored_relative_and_survive_a_move(tmp_path):
    store = make_store(tmp_path)
    store.save(store.load())
    stored = json.loads(store.file.read_text())
    assert stored["models_dir"] == "Verbatica-data/models"

    moved = tmp_path / "moved"
    (tmp_path / "app").rename(moved)
    loaded = st.SettingsStore(moved, legacy_file=tmp_path / "none.json").load()
    assert loaded.models_dir == str(moved / "Verbatica-data" / "models")


def test_old_home_folder_settings_are_carried_over_and_left_alone(tmp_path):
    legacy = {"models_dir": "/old/models", "output_dir": "/old/out", "error_log_path": "/old/logs"}
    store = make_store(tmp_path, legacy)
    s = store.load()
    assert (s.models_dir, s.output_dir, s.error_log_dir) == ("/old/models", "/old/out", "/old/logs")
    store.save(s)
    assert json.loads(store.legacy_file.read_text()) == legacy  # old file not changed


def test_new_settings_win_over_old_ones(tmp_path):
    store = make_store(tmp_path, {"output_dir": "/old/out"})
    s = store.load()
    s.output_dir = str(tmp_path / "new")
    store.save(s)
    assert store.load().output_dir == str((tmp_path / "new").resolve())


def test_data_folder_from_before_the_rename_is_renamed_and_kept(tmp_path):
    old = tmp_path / "app" / "Audio-Transcriber-data"
    (old / "models").mkdir(parents=True)
    (old / "models" / "tiny.pt").write_text("model")
    (old / "settings.json").write_text(json.dumps(
        {"models_dir": "Audio-Transcriber-data/models", "output_formats": [".srt"]}))

    store = make_store(tmp_path)
    s = store.load()

    new = tmp_path / "app" / "Verbatica-data"
    assert not old.exists()
    assert (new / "models" / "tiny.pt").read_text() == "model"
    assert s.models_dir == str(new / "models")
    assert s.output_formats == [".srt"]


def test_new_data_folder_wins_over_one_from_before_the_rename(tmp_path):
    old = tmp_path / "app" / "Audio-Transcriber-data"
    old.mkdir(parents=True)
    (tmp_path / "app" / "Verbatica-data").mkdir()
    store = make_store(tmp_path)
    assert store.data_dir == tmp_path / "app" / "Verbatica-data"
    assert old.exists()  # left alone


def test_old_data_folder_is_used_if_it_cannot_be_renamed(tmp_path, monkeypatch):
    old = tmp_path / "app" / "Audio-Transcriber-data"
    old.mkdir(parents=True)
    (old / "settings.json").write_text(json.dumps({"models_dir": "Audio-Transcriber-data/models"}))

    def refuse(self, target):
        raise PermissionError("read-only")

    monkeypatch.setattr(st.Path, "rename", refuse)
    store = make_store(tmp_path)
    assert store.data_dir == old
    assert store.load().models_dir == str(old / "models")


def test_damaged_settings_file_falls_back_to_defaults(tmp_path):
    store = make_store(tmp_path)
    store.data_dir.mkdir(parents=True)
    store.file.write_text("{not json")
    assert store.load().output_formats == [".txt"]


def test_unknown_formats_are_dropped(tmp_path):
    store = make_store(tmp_path)
    store.data_dir.mkdir(parents=True)
    store.file.write_text(json.dumps({"output_formats": [".exe", ".srt"]}))
    assert store.load().output_formats == [".srt"]


def test_app_folder_is_beside_the_macos_app_bundle(tmp_path, monkeypatch):
    exe = tmp_path / "Apps" / "Verbatica.app" / "Contents" / "MacOS" / "Verbatica"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert st.app_folder() == (tmp_path / "Apps").resolve()


def test_app_folder_is_beside_a_single_file_build(tmp_path, monkeypatch):
    exe = tmp_path / "Downloads" / "Verbatica.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert st.app_folder() == (tmp_path / "Downloads").resolve()
