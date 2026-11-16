"""Shared test setup: runs Qt without a screen, and offers a fake Whisper so tests need
no PyTorch or model files."""

import os
import sys
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

qt_app = QApplication.instance() or QApplication([])


@pytest.fixture
def fake_whisper(monkeypatch):
    """A fake `whisper` package. `whisper.transcribe` exposes tqdm the way the real module
    does, so the progress and cancel hook can be tested."""
    whisper = types.ModuleType("whisper")
    whisper.load_model = lambda path: ("loaded", path)
    transcribe_module = types.ModuleType("whisper.transcribe")
    transcribe_module.tqdm = types.SimpleNamespace(tqdm=object)
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    monkeypatch.setitem(sys.modules, "whisper.transcribe", transcribe_module)
    return whisper


def collect(signal):
    """Record every emission of a Qt signal as a tuple of its arguments."""
    received = []
    signal.connect(lambda *args: received.append(args))
    return received
