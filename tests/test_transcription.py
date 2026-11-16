"""Tests for output file names, duplicate files, subtitle output, and cancelling."""

import sys

from conftest import collect

from verbatica import transcription as tr

# --- Output file names -------------------------------------------------------


def test_output_base_uses_plain_name_when_free(tmp_path):
    assert tr.unique_output_base(tmp_path, "talk", [".txt"]) == tmp_path / "talk"


def test_output_base_never_overwrites_existing_transcript(tmp_path):
    (tmp_path / "talk.txt").write_text("earlier result")
    (tmp_path / "talk (2).txt").write_text("another")
    assert tr.unique_output_base(tmp_path, "talk", [".txt"]) == tmp_path / "talk (3)"


def test_output_base_checks_every_chosen_format(tmp_path):
    (tmp_path / "talk.srt").write_text("old subtitles")
    assert tr.unique_output_base(tmp_path, "talk", [".txt", ".srt"]) == tmp_path / "talk (2)"


def test_same_name_files_in_one_batch_get_separate_outputs(tmp_path, fake_whisper):
    files = [str(tmp_path / "a" / "talk.mp3"), str(tmp_path / "b" / "talk.wav")]
    texts = iter(["first", "second"])

    class Model:
        def transcribe(self, path, verbose):
            return {"text": next(texts), "segments": []}

    out = tmp_path / "out"
    tr.TranscriptionWorker(files, Model(), str(out), "", [".txt"]).run()

    assert (out / "talk.txt").read_text() == "first"
    assert (out / "talk (2).txt").read_text() == "second"


# --- Duplicate files ---------------------------------------------------------


def test_adding_the_same_file_twice_is_skipped(tmp_path):
    f = tmp_path / "talk.mp3"
    f.write_text("x")
    files = []
    assert tr.add_unique_files(files, [str(f)]) == 0
    assert tr.add_unique_files(files, [str(f), str(tmp_path / "." / "talk.mp3")]) == 2
    assert files == [str(f)]


def test_different_files_with_same_name_are_both_kept(tmp_path):
    files = []
    tr.add_unique_files(files, [str(tmp_path / "a" / "talk.mp3"), str(tmp_path / "b" / "talk.mp3")])
    assert len(files) == 2


# --- Subtitles ---------------------------------------------------------------

SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": " Hello there."},
    {"start": 3661.5, "end": 3663.0, "text": " One hour in."},
]


def test_srt_output():
    assert tr.to_srt(SEGMENTS) == (
        "1\n00:00:00,000 --> 00:00:02,500\nHello there.\n"
        "\n"
        "2\n01:01:01,500 --> 01:01:03,000\nOne hour in.\n"
    )


def test_vtt_output():
    assert tr.to_vtt(SEGMENTS) == (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:02.500\nHello there.\n"
        "\n"
        "01:01:01.500 --> 01:01:03.000\nOne hour in.\n"
    )


def test_worker_writes_every_chosen_format(tmp_path, fake_whisper):
    class Model:
        def transcribe(self, path, verbose):
            return {"text": " Hello there. One hour in.", "segments": SEGMENTS}

    tr.TranscriptionWorker([str(tmp_path / "talk.mp3")], Model(), str(tmp_path), "",
                           [".txt", ".srt", ".vtt"]).run()
    assert (tmp_path / "talk.txt").exists()
    assert (tmp_path / "talk.srt").read_text().startswith("1\n00:00:00,000")
    assert (tmp_path / "talk.vtt").read_text().startswith("WEBVTT")


# --- Cancelling and progress -------------------------------------------------


def test_cancel_stops_partway_through_a_file(tmp_path, fake_whisper):
    windows_processed = []

    class Model:
        def transcribe(self, path, verbose):
            # Mimic Whisper: one progress update per 30-second window
            with sys.modules["whisper.transcribe"].tqdm.tqdm(total=10) as bar:
                for _ in range(10):
                    windows_processed.append(path)
                    if len(windows_processed) == 3:
                        worker.cancel()
                    bar.update(1)
            return {"text": "done", "segments": []}

    files = [str(tmp_path / "one.mp3"), str(tmp_path / "two.mp3")]
    worker = tr.TranscriptionWorker(files, Model(), str(tmp_path), "", [".txt"])
    done = collect(worker.all_complete)
    worker.run()

    assert len(windows_processed) == 3  # stopped inside the first file
    assert not (tmp_path / "one.txt").exists()
    assert not (tmp_path / "two.txt").exists()
    _, _, succeeded, failed, cancelled = done[0]
    assert (succeeded, failed, cancelled) == (0, 0, True)
    assert sys.modules["whisper.transcribe"].tqdm.tqdm is object  # real tqdm put back


def test_progress_moves_within_a_file(tmp_path, fake_whisper):
    class Model:
        def transcribe(self, path, verbose):
            with sys.modules["whisper.transcribe"].tqdm.tqdm(total=4) as bar:
                for _ in range(4):
                    bar.update(1)
            return {"text": "", "segments": []}

    worker = tr.TranscriptionWorker([str(tmp_path / "a.mp3"), str(tmp_path / "b.mp3")],
                                    Model(), str(tmp_path), "", [".txt"])
    progress = collect(worker.progress)
    worker.run()
    assert [p for (p,) in progress][:4] == [12, 25, 37, 50]


def test_failed_file_is_logged_and_the_batch_continues(tmp_path, fake_whisper):
    class Model:
        def transcribe(self, path, verbose):
            if path.endswith("bad.mp3"):
                raise RuntimeError("cannot decode")
            return {"text": "ok", "segments": []}

    logs = tmp_path / "logs"
    worker = tr.TranscriptionWorker([str(tmp_path / "bad.mp3"), str(tmp_path / "good.mp3")],
                                    Model(), str(tmp_path), str(logs), [".txt"])
    done = collect(worker.all_complete)
    worker.run()
    assert (tmp_path / "good.txt").exists()
    assert done[0][2:] == (1, 1, False)
    assert "cannot decode" in next(logs.iterdir()).read_text()


# --- ffmpeg ------------------------------------------------------------------


def test_ffmpeg_in_homebrew_folder_is_found_without_path(tmp_path, monkeypatch):
    if sys.platform == "win32":
        return
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(tr, "EXTRA_FFMPEG_DIRS", [str(tmp_path)])
    assert tr.find_ffmpeg() == str(fake)
    assert tr.shutil.which("ffmpeg") == str(fake)  # Whisper finds it too


def test_missing_ffmpeg_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(tr, "EXTRA_FFMPEG_DIRS", [])
    assert tr.find_ffmpeg() is None
