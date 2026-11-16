"""Tests for downloading, verifying and loading models."""

import hashlib

import pytest
from conftest import collect

from verbatica import models


@pytest.fixture
def fake_model_url(tmp_path, monkeypatch):
    """Serve a fake 'tiny' model from a file:// URL with its checksum in the path,
    the way OpenAI's real URLs are built."""
    content = b"fake model weights" * 1000
    sha = hashlib.sha256(content).hexdigest()
    server = tmp_path / "server" / sha
    server.mkdir(parents=True)
    (server / "tiny.pt").write_bytes(content)
    monkeypatch.setitem(models.MODEL_URLS, "tiny", (server / "tiny.pt").as_uri())
    return content


def run_model_worker(models_dir, replace_existing=False):
    worker = models.ModelDownloadWorker("tiny", str(models_dir), replace_existing)
    results = {name: collect(getattr(worker, name)) for name in ("model_loaded", "corrupt_file", "error")}
    worker.run()
    return results


def test_every_model_url_carries_a_checksum():
    for url in models.MODEL_URLS.values():
        assert len(models.expected_sha256(url)) == 64


def test_download_verifies_and_loads_in_worker(tmp_path, fake_model_url, fake_whisper):
    folder = tmp_path / "models"
    results = run_model_worker(folder)
    assert results["model_loaded"] == [(("loaded", str(folder / "tiny.pt")), "tiny")]
    assert (folder / "tiny.pt").read_bytes() == fake_model_url
    assert not (folder / "tiny.pt.part").exists()


def test_partial_model_file_is_not_loaded(tmp_path, fake_model_url, fake_whisper):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "tiny.pt").write_bytes(fake_model_url[:100])  # interrupted download
    results = run_model_worker(folder)
    assert results["model_loaded"] == []
    assert results["corrupt_file"] == [(str(folder / "tiny.pt"),)]
    assert (folder / "tiny.pt").read_bytes() == fake_model_url[:100]  # not touched without consent


def test_replacing_a_damaged_model_file(tmp_path, fake_model_url, fake_whisper):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "tiny.pt").write_bytes(b"damaged")
    results = run_model_worker(folder, replace_existing=True)
    assert len(results["model_loaded"]) == 1
    assert (folder / "tiny.pt").read_bytes() == fake_model_url


def test_bad_download_leaves_no_model_file(tmp_path, fake_model_url, fake_whisper, monkeypatch):
    bad_server = tmp_path / "bad" / ("0" * 64)
    bad_server.mkdir(parents=True)
    (bad_server / "tiny.pt").write_bytes(fake_model_url)
    monkeypatch.setitem(models.MODEL_URLS, "tiny", (bad_server / "tiny.pt").as_uri())

    folder = tmp_path / "models"
    results = run_model_worker(folder)
    assert results["model_loaded"] == []
    assert "checksum" in results["error"][0][0]
    assert not (folder / "tiny.pt").exists()
    assert not (folder / "tiny.pt.part").exists()
