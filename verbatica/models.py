"""The Whisper models the app offers, and the background worker that downloads,
verifies and loads them so the window stays responsive."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

# Official model files. OpenAI puts each file's SHA256 checksum in its URL.
MODEL_URLS = {
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}

MODEL_SIZES = {
    "tiny": "~75 MB",
    "base": "~142 MB",
    "small": "~466 MB",
    "medium": "~1.5 GB",
    "large-v2": "~3 GB",
    "large-v3": "~3 GB",
}

DEFAULT_MODEL = "medium"


def expected_sha256(url: str) -> str:
    return url.split("/")[-2]


def file_sha256(path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelDownloadWorker(QThread):
    """Downloads (if needed), verifies and loads a Whisper model, all off the UI thread."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    model_loaded = pyqtSignal(object, str)  # model, model_name
    corrupt_file = pyqtSignal(str)  # path of an existing model file that failed its checksum
    error = pyqtSignal(str)

    def __init__(self, model_name: str, models_dir: str, replace_existing: bool = False):
        super().__init__()
        self.model_name = model_name
        self.models_dir = models_dir
        self.replace_existing = replace_existing

    def run(self):
        try:
            url = MODEL_URLS[self.model_name]
            sha256 = expected_sha256(url)
            model_path = Path(self.models_dir) / f"{self.model_name}.pt"

            if model_path.exists() and not self.replace_existing:
                self.status.emit("Verifying existing model file...")
                if file_sha256(model_path) != sha256:
                    self.corrupt_file.emit(str(model_path))
                    return
            else:
                self.download(url, sha256, model_path)
        except Exception as e:
            self.error.emit(f"Download failed: {e}")
            return

        self.status.emit("⏳ Loading model into memory...")
        try:
            import whisper  # imported here because it loads PyTorch, which is slow

            model = whisper.load_model(str(model_path))
        except Exception as e:
            self.error.emit(f"Failed to load model: {e}")
            return
        self.model_loaded.emit(model, self.model_name)

    def download(self, url: str, sha256: str, model_path: Path):
        """Download to a .part file and only move it into place once its checksum matches,
        so an interrupted or damaged download never looks like a finished model."""
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        part_path = model_path.with_name(model_path.name + ".part")
        self.status.emit(f"Downloading {self.model_name} model ({MODEL_SIZES[self.model_name]})...")

        last_percent = [-1]

        def report_progress(block_num, block_size, total_size):
            if total_size <= 0:
                return
            downloaded = min(block_num * block_size, total_size)
            percent = int((downloaded / total_size) * 100)
            if percent == last_percent[0]:
                return  # only update the UI when the percentage changes
            last_percent[0] = percent
            self.progress.emit(percent)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            self.status.emit(f"Downloading: {mb_downloaded:.1f} MB / {mb_total:.1f} MB")

        try:
            urllib.request.urlretrieve(url, str(part_path), reporthook=report_progress)
            self.status.emit("Verifying download...")
            if file_sha256(part_path) != sha256:
                raise ValueError("the downloaded file is damaged (checksum mismatch). Please try again.")
            os.replace(part_path, model_path)
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise
