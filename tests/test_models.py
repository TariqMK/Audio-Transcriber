"""Tests for downloading, verifying and loading model folders."""

import hashlib

import pytest
from conftest import collect

from verbatica import models

FILES = {"config.json": b'{"fake": true}', "model.bin": b"fake weights" * 5000, "tokenizer.json": b"{}"}


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Serve a fake 'tiny' model from file:// URLs laid out like Hugging Face's."""
    server = tmp_path / "server" / "test" / "tiny" / "resolve" / "rev1"
    server.mkdir(parents=True)
    for name, data in FILES.items():
        (server / name).write_bytes(data)
    info = models.ModelInfo(
        repo="test/tiny",
        revision="rev1",
        files=tuple((name, len(data), hashlib.sha256(data).hexdigest()) for name, data in FILES.items()),
    )
    monkeypatch.setattr(models, "HF_BASE", (tmp_path / "server").as_uri())
    monkeypatch.setitem(models.MODELS, "tiny", info)
    return server


def run_model_worker(models_dir, replace_existing=False):
    worker = models.ModelDownloadWorker("tiny", str(models_dir), replace_existing)
    results = {name: collect(getattr(worker, name))
               for name in ("model_loaded", "corrupt_file", "error", "progress")}
    worker.run()
    return results


def test_catalogue_is_pinned_with_checksums():
    for info in models.MODELS.values():
        assert len(info.revision) == 40  # a full commit hash, not a branch name
        names = {name for name, _, _ in info.files}
        assert {"config.json", "model.bin", "tokenizer.json"} <= names
        for _, size, sha256 in info.files:
            assert size > 0 and len(sha256) == 64


def test_file_url():
    info = models.MODELS["tiny"]
    assert models.file_url(info, "model.bin") == (
        f"https://huggingface.co/Systran/faster-whisper-tiny/resolve/{info.revision}/model.bin")


def test_download_verifies_and_loads_in_worker(tmp_path, fake_hub, fake_whisper):
    results = run_model_worker(tmp_path / "models")
    folder = tmp_path / "models" / "faster-whisper-tiny"
    model, name = results["model_loaded"][0]
    assert name == "tiny"
    assert (model.path, model.device, model.compute_type) == (str(folder), "cpu", "int8")
    for file_name, data in FILES.items():
        assert (folder / file_name).read_bytes() == data
    assert not list(folder.glob("*.part"))
    assert results["progress"][-1] == (100,)


def test_interrupted_download_resumes_missing_files_only(tmp_path, fake_hub, fake_whisper):
    folder = tmp_path / "models" / "faster-whisper-tiny"
    folder.mkdir(parents=True)
    (folder / "config.json").write_bytes(FILES["config.json"])  # already complete
    (folder / "model.bin.part").write_bytes(b"half")  # leftover from an interrupted download
    (fake_hub / "config.json").unlink()  # would fail if downloaded again
    results = run_model_worker(tmp_path / "models")
    assert len(results["model_loaded"]) == 1
    assert (folder / "model.bin").read_bytes() == FILES["model.bin"]


def test_damaged_model_file_is_not_loaded_or_replaced_without_consent(tmp_path, fake_hub, fake_whisper):
    folder = tmp_path / "models" / "faster-whisper-tiny"
    folder.mkdir(parents=True)
    for file_name, data in FILES.items():
        (folder / file_name).write_bytes(data)
    (folder / "model.bin").write_bytes(FILES["model.bin"][:100])
    results = run_model_worker(tmp_path / "models")
    assert results["model_loaded"] == []
    assert results["corrupt_file"] == [("tiny", str(folder))]
    assert (folder / "model.bin").read_bytes() == FILES["model.bin"][:100]


def test_replacing_damaged_model_files(tmp_path, fake_hub, fake_whisper):
    folder = tmp_path / "models" / "faster-whisper-tiny"
    folder.mkdir(parents=True)
    (folder / "model.bin").write_bytes(b"damaged")
    results = run_model_worker(tmp_path / "models", replace_existing=True)
    assert len(results["model_loaded"]) == 1
    assert (folder / "model.bin").read_bytes() == FILES["model.bin"]


def test_bad_download_leaves_no_model_file(tmp_path, fake_hub, fake_whisper):
    (fake_hub / "model.bin").write_bytes(b"tampered" * 7500)  # wrong content, right size
    results = run_model_worker(tmp_path / "models")
    folder = tmp_path / "models" / "faster-whisper-tiny"
    assert results["model_loaded"] == []
    assert "checksum" in results["error"][0][0]
    assert not (folder / "model.bin").exists()
    assert not (folder / "model.bin.part").exists()


def test_load_failure_is_reported(tmp_path, fake_hub, fake_whisper):
    def broken(*args, **kwargs):
        raise RuntimeError("unsupported model")
    fake_whisper.WhisperModel = broken
    results = run_model_worker(tmp_path / "models")
    assert results["error"] == [("Failed to load model: unsupported model",)]
