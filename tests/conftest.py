"""Shared test setup: runs Qt without a screen, and offers a fake faster-whisper so tests
need no model files or network."""

import os
import sys
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

qt_app = QApplication.instance() or QApplication([])


@pytest.fixture
def fake_whisper(monkeypatch):
    """A fake `faster_whisper` package whose WhisperModel records how it was loaded."""
    module = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, path, device, compute_type):
            self.path, self.device, self.compute_type = path, device, compute_type

    module.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return module


class FakeModel:
    """Stands in for a loaded faster-whisper model. `segments` is a list of
    (start, end, text); `on_segment` runs before each one is handed over."""

    def __init__(self, segments=(), duration=None, on_segment=None, fail=None):
        self.segments = list(segments)
        self.duration = duration if duration is not None else (self.segments[-1][1] if self.segments else 0)
        self.on_segment = on_segment
        self.fail = fail
        self.calls = []

    def transcribe(self, path, vad_filter, beam_size):
        self.calls.append((path, vad_filter, beam_size))
        if self.fail and self.fail(path):
            raise RuntimeError("cannot decode")

        def generate():
            for i, (start, end, text) in enumerate(self.segments):
                if self.on_segment:
                    self.on_segment(path, i)
                yield types.SimpleNamespace(start=start, end=end, text=text)

        return generate(), types.SimpleNamespace(duration=self.duration)


def collect(signal):
    """Record every emission of a Qt signal as a tuple of its arguments."""
    received = []
    signal.connect(lambda *args: received.append(args))
    return received
