"""The Whisper models the app offers, and the background worker that downloads,
verifies and loads them so the window stays responsive. Models run with faster-whisper
(CTranslate2) and come from Hugging Face, pinned to an exact revision with checksums."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

HF_BASE = "https://huggingface.co"
DEFAULT_MODEL = "medium"

# CPU only: CUDA would need extra NVIDIA libraries on the user's machine.
# int8 roughly halves memory use and speeds up CPU work, with a very small accuracy cost.
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
# CPU threads are left at faster-whisper's default. Using every core made an Apple M4
# (4 performance + 6 efficiency cores) 2x slower, because work waits for the slow cores.


@dataclass(frozen=True)
class ModelInfo:
    repo: str
    revision: str  # exact Hugging Face commit, so the files and checksums never change
    files: tuple  # (file name, size in bytes, SHA256)

    @property
    def total_size(self) -> int:
        return sum(size for _, size, _ in self.files)


MODELS = {
    "tiny": ModelInfo(
        repo="Systran/faster-whisper-tiny",
        revision="d90ca5fe260221311c53c58e660288d3deb8d356",
        files=(
            ("config.json", 2249, "a73a28cdfe1c43ccc7202fa333d1f89c202477271407ae9a7f19afa52039cac8"),
            ("model.bin", 75538270, "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1"),
            ("tokenizer.json", 2203239, "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
            ("vocabulary.txt", 459861, "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"),
        ),
    ),
    "base": ModelInfo(
        repo="Systran/faster-whisper-base",
        revision="ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
        files=(
            ("config.json", 2309, "56a6d8110d311f19c8f0471e562832c7527f146b567275bfca59fcf7c184da9a"),
            ("model.bin", 145217532, "d01c3014881c9c6f3133c182f3d2887eb6ca1c789a7538c5c007196857a0a6a9"),
            ("tokenizer.json", 2203239, "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
            ("vocabulary.txt", 459861, "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"),
        ),
    ),
    "small": ModelInfo(
        repo="Systran/faster-whisper-small",
        revision="536b0662742c02347bc0e980a01041f333bce120",
        files=(
            ("config.json", 2370, "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828"),
            ("model.bin", 483546902, "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671"),
            ("tokenizer.json", 2203239, "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
            ("vocabulary.txt", 459861, "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"),
        ),
    ),
    "medium": ModelInfo(
        repo="Systran/faster-whisper-medium",
        revision="08e178d48790749d25932bbc082711ddcfdfbc4f",
        files=(
            ("config.json", 2257, "3622a2ddc41ec0e0fd4e68c13c6830f03b90c38d89aaad184de02c8c642cf807"),
            ("model.bin", 1527906378, "9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae"),
            ("tokenizer.json", 2203239, "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
            ("vocabulary.txt", 459861, "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"),
        ),
    ),
    "large-v2": ModelInfo(
        repo="Systran/faster-whisper-large-v2",
        revision="f0fe81560cb8b68660e564f55dd99207059c092e",
        files=(
            ("config.json", 2796, "d86b7a7664a12559d644aa210a32ce9a7e03913e794b7ea4fb7182de69e273a7"),
            ("model.bin", 3086912962, "bf2a9746382e1aa7ffff6b3a0d137ed9edbd9670c3b87e5d35f5e85e70d0333a"),
            ("tokenizer.json", 2203239, "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab"),
            ("vocabulary.txt", 459861, "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913"),
        ),
    ),
    "large-v3": ModelInfo(
        repo="Systran/faster-whisper-large-v3",
        revision="edaa852ec7e145841d8ffdb056a99866b5f0a478",
        files=(
            ("config.json", 2394, "a9306624f5ec14270a014b647e5c316b6e03a662c369758d1b90697a7b0655b9"),
            ("model.bin", 3087284237, "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1"),
            ("preprocessor_config.json", 340, "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
            ("tokenizer.json", 2480617, "6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca"),
            ("vocabulary.json", 1068114, "c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1"),
        ),
    ),
}


def human_size(num_bytes: int) -> str:
    if num_bytes >= 1024 ** 3:
        return f"~{num_bytes / 1024 ** 3:.1f} GB"
    return f"~{num_bytes / 1024 ** 2:.0f} MB"


MODEL_SIZES = {name: human_size(info.total_size) for name, info in MODELS.items()}


def model_folder(models_dir, name: str) -> Path:
    return Path(models_dir) / f"faster-whisper-{name}"


def file_url(info: ModelInfo, file_name: str) -> str:
    return f"{HF_BASE}/{info.repo}/resolve/{info.revision}/{file_name}"


def file_sha256(path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model_files(folder: Path, info: ModelInfo, on_status=lambda message: None):
    """Return (missing, damaged): the file names that are not there, and those that are
    there but have the wrong size or checksum."""
    missing, damaged = [], []
    for name, size, sha256 in info.files:
        path = folder / name
        if not path.exists():
            missing.append(name)
            continue
        on_status(f"Verifying {name}...")
        if path.stat().st_size != size or file_sha256(path) != sha256:
            damaged.append(name)
    return missing, damaged


class ModelDownloadWorker(QThread):
    """Downloads (if needed), verifies and loads a model, all off the UI thread."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    model_loaded = pyqtSignal(object, str)  # model, model_name
    corrupt_file = pyqtSignal(str, str)  # model_name, model folder with files that failed their checksum
    error = pyqtSignal(str)

    def __init__(self, model_name: str, models_dir: str, replace_existing: bool = False):
        super().__init__()
        self.model_name = model_name
        self.models_dir = models_dir
        self.replace_existing = replace_existing

    def run(self):
        info = MODELS[self.model_name]
        folder = model_folder(self.models_dir, self.model_name)
        try:
            missing, damaged = check_model_files(folder, info, self.status.emit)
            if damaged and not self.replace_existing:
                self.corrupt_file.emit(self.model_name, str(folder))
                return
            to_download = [f for f in info.files if f[0] in missing or f[0] in damaged]
            if to_download:
                self.download(info, folder, to_download)
        except Exception as e:
            self.error.emit(f"Download failed: {e}")
            return

        self.status.emit("⏳ Loading model into memory...")
        try:
            from faster_whisper import WhisperModel  # imported here because it loads CTranslate2

            model = WhisperModel(str(folder), device=DEVICE, compute_type=COMPUTE_TYPE)
        except Exception as e:
            self.error.emit(f"Failed to load model: {e}")
            return
        self.model_loaded.emit(model, self.model_name)

    def download(self, info: ModelInfo, folder: Path, files):
        """Download each file to a .part file and only move it into place once its checksum
        matches, so an interrupted or damaged download never looks like a finished model."""
        folder.mkdir(parents=True, exist_ok=True)
        total = sum(size for _, size, _ in files)
        done_before = 0
        last_percent = [-1]
        self.status.emit(f"Downloading {self.model_name} model ({human_size(total)})...")

        for name, size, sha256 in files:
            target = folder / name
            part_path = target.with_name(name + ".part")

            def report_progress(block_num, block_size, _total_size, done_before=done_before, size=size):
                downloaded = done_before + min(block_num * block_size, size)
                percent = int(downloaded / total * 100) if total else 100
                if percent == last_percent[0]:
                    return  # only update the UI when the percentage changes
                last_percent[0] = percent
                self.progress.emit(percent)
                self.status.emit(f"Downloading: {downloaded / 1024 ** 2:.1f} MB / {total / 1024 ** 2:.1f} MB")

            try:
                urllib.request.urlretrieve(file_url(info, name), str(part_path), reporthook=report_progress)
                self.status.emit(f"Verifying {name}...")
                if part_path.stat().st_size != size or file_sha256(part_path) != sha256:
                    raise ValueError(f"{name} is damaged (checksum mismatch). Please try again.")
                os.replace(part_path, target)
            except BaseException:
                part_path.unlink(missing_ok=True)
                raise
            done_before += size
