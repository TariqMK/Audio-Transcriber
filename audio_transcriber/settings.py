"""Finds the app's data folder, which sits next to the app so the app is portable,
and loads and saves the user's settings there."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from audio_transcriber.transcription import OUTPUT_FORMATS

DATA_FOLDER_NAME = "Audio-Transcriber-data"
SETTINGS_FILE_NAME = "settings.json"

# Settings file used by versions before the data folder. Read once to carry settings over; never changed.
LEGACY_SETTINGS_FILE = Path.home() / ".whisper_gui_settings.json"


def app_folder() -> Path:
    """The folder the user sees the app in: beside the .exe, the Linux binary or the macOS .app.
    Run from source, it is the project root."""
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        for parent in executable.parents:
            if parent.suffix == ".app":  # .../Audio-Transcriber.app/Contents/MacOS/Audio-Transcriber
                return parent.parent
        return executable.parent
    return Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    models_dir: str = ""
    output_dir: str = ""
    error_log_dir: str = ""
    output_formats: list = field(default_factory=lambda: [".txt"])


class SettingsStore:
    """Reads and writes settings.json in the data folder. Paths inside the app folder are
    stored relative to it, so they still work after the app and its data are moved."""

    def __init__(self, app_dir: Path | None = None, legacy_file: Path = LEGACY_SETTINGS_FILE):
        self.app_dir = Path(app_dir) if app_dir else app_folder()
        self.data_dir = self.app_dir / DATA_FOLDER_NAME
        self.file = self.data_dir / SETTINGS_FILE_NAME
        self.legacy_file = legacy_file

    def defaults(self) -> Settings:
        return Settings(
            models_dir=str(self.data_dir / "models"),
            error_log_dir=str(self.data_dir / "logs"),
        )

    def load(self) -> Settings:
        settings = self.defaults()
        raw = self._read(self.file)
        if raw is None:
            raw = self._read(self.legacy_file) or {}
            if "error_log_path" in raw:
                raw["error_log_dir"] = raw["error_log_path"]

        for key in ("models_dir", "output_dir", "error_log_dir"):
            value = raw.get(key)
            if isinstance(value, str) and value:
                setattr(settings, key, self._from_stored(value))

        formats = raw.get("output_formats")
        if isinstance(formats, list):
            settings.output_formats = [f for f in formats if f in OUTPUT_FORMATS] or [".txt"]
        return settings

    def save(self, settings: Settings) -> None:
        """Raises OSError if the data folder can't be written, for example under Program Files."""
        data = {
            "models_dir": self._to_stored(settings.models_dir),
            "output_dir": self._to_stored(settings.output_dir),
            "error_log_dir": self._to_stored(settings.error_log_dir),
            "output_formats": list(settings.output_formats),
        }
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temp_file = self.file.with_name(self.file.name + ".tmp")
        temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temp_file, self.file)

    def _read(self, path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _to_stored(self, path: str) -> str:
        if not path:
            return ""
        absolute = Path(path).resolve()
        try:
            return absolute.relative_to(self.app_dir.resolve()).as_posix()
        except ValueError:
            return str(absolute)

    def _from_stored(self, value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else self.app_dir / path)
