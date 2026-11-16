"""Tests for output file names, duplicate files, subtitle output, progress and cancelling."""

from conftest import FakeModel, collect

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


def test_same_name_files_in_one_batch_get_separate_outputs(tmp_path):
    files = [str(tmp_path / "a" / "talk.mp3"), str(tmp_path / "b" / "talk.wav")]
    out = tmp_path / "out"
    tr.TranscriptionWorker(files, FakeModel([(0, 1, " Hello.")]), str(out), "", [".txt"]).run()
    assert (out / "talk.txt").read_text() == " Hello."
    assert (out / "talk (2).txt").read_text() == " Hello."


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


# --- Text and subtitles ------------------------------------------------------

SEGMENTS = [(0.0, 2.5, " Hello there."), (3661.5, 3663.0, " One hour in.")]
SEGMENT_DICTS = [{"start": s, "end": e, "text": t} for s, e, t in SEGMENTS]


def test_srt_output():
    assert tr.to_srt(SEGMENT_DICTS) == (
        "1\n00:00:00,000 --> 00:00:02,500\nHello there.\n"
        "\n"
        "2\n01:01:01,500 --> 01:01:03,000\nOne hour in.\n"
    )


def test_vtt_output():
    assert tr.to_vtt(SEGMENT_DICTS) == (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:02.500\nHello there.\n"
        "\n"
        "01:01:01.500 --> 01:01:03.000\nOne hour in.\n"
    )


def test_worker_writes_every_chosen_format(tmp_path):
    model = FakeModel(SEGMENTS)
    tr.TranscriptionWorker([str(tmp_path / "talk.mp3")], model, str(tmp_path), "",
                           [".txt", ".srt", ".vtt"]).run()
    assert (tmp_path / "talk.txt").read_text() == " Hello there. One hour in."
    assert (tmp_path / "talk.srt").read_text() == tr.to_srt(SEGMENT_DICTS)
    assert (tmp_path / "talk.vtt").read_text() == tr.to_vtt(SEGMENT_DICTS)
    assert model.calls[0][1:] == (True, 1)  # silence filter on, greedy decoding


# --- Cancelling and progress -------------------------------------------------


def test_cancel_stops_partway_through_a_file(tmp_path):
    seen = []

    def on_segment(path, index):
        seen.append(index)
        if index == 2:
            worker.cancel()

    model = FakeModel([(i * 30, i * 30 + 30, f" part {i}") for i in range(10)], on_segment=on_segment)
    files = [str(tmp_path / "one.mp3"), str(tmp_path / "two.mp3")]
    worker = tr.TranscriptionWorker(files, model, str(tmp_path), "", [".txt"])
    done = collect(worker.all_complete)
    worker.run()

    assert seen == [0, 1, 2]  # stopped inside the first file
    assert len(model.calls) == 1  # second file never started
    assert not (tmp_path / "one.txt").exists()
    _, _, succeeded, failed, cancelled = done[0]
    assert (succeeded, failed, cancelled) == (0, 0, True)


def test_progress_moves_within_a_file(tmp_path):
    model = FakeModel([(0, 25, "a"), (25, 50, "b"), (50, 75, "c"), (75, 100, "d")])
    worker = tr.TranscriptionWorker([str(tmp_path / "a.mp3"), str(tmp_path / "b.mp3")],
                                    model, str(tmp_path), "", [".txt"])
    progress = collect(worker.progress)
    worker.run()
    assert [p for (p,) in progress][:4] == [12, 25, 37, 50]


def test_silent_file_gives_empty_result(tmp_path):
    tr.TranscriptionWorker([str(tmp_path / "quiet.mp3")], FakeModel([], duration=60),
                           str(tmp_path), "", [".txt", ".srt"]).run()
    assert (tmp_path / "quiet.txt").read_text() == ""
    assert (tmp_path / "quiet.srt").read_text() == ""


def test_failed_file_is_logged_and_the_batch_continues(tmp_path):
    model = FakeModel([(0, 1, "ok")], fail=lambda path: path.endswith("bad.mp3"))
    logs = tmp_path / "logs"
    worker = tr.TranscriptionWorker([str(tmp_path / "bad.mp3"), str(tmp_path / "good.mp3")],
                                    model, str(tmp_path), str(logs), [".txt"])
    done = collect(worker.all_complete)
    worker.run()
    assert (tmp_path / "good.txt").exists()
    assert done[0][2:] == (1, 1, False)
    assert "cannot decode" in next(logs.iterdir()).read_text()
