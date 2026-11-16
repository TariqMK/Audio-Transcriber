"""Tests for the manual update check. The GitHub request is stubbed; no network is used."""

import io
import urllib.error

import pytest

from verbatica import update_check as uc

SAMPLE = {
    "tag_name": "v1.3.0",
    "html_url": "https://github.com/TariqMK/Verbatica/releases/tag/v1.3.0",
    "body": "  Faster transcription.  ",
}


@pytest.mark.parametrize("text, expected", [
    ("1.2.3", (1, 2, 3)), ("v1.2.3", (1, 2, 3)), ("0.0.0.dev0", None), ("v1.2", None), ("nightly", None),
])
def test_parse_version(text, expected):
    assert uc.parse_version(text) == expected


def test_versions_compare_as_numbers():
    assert uc.is_newer("v0.10.0", "0.9.0")
    assert not uc.is_newer("v0.9.0", "0.10.0")
    assert not uc.is_newer("v1.0.0", "1.0.0")
    assert not uc.is_newer("v1.0.0", "0.0.0.dev0")


def test_reading_a_github_response():
    info = uc.parse_release(SAMPLE, "1.2.0")
    assert (info.latest, info.url, info.notes) == ("v1.3.0", SAMPLE["html_url"], "Faster transcription.")
    assert info.newer and not info.dev_build


def test_up_to_date():
    info = uc.parse_release(SAMPLE, "1.3.0")
    assert not info.newer and not info.dev_build


def test_development_build_is_never_up_to_date_or_older():
    info = uc.parse_release(SAMPLE, "0.0.0.dev0")
    assert info.dev_build and not info.newer


@pytest.mark.parametrize("url", [None, "http://github.com/x", "javascript:alert(1)", 42])
def test_missing_or_unsafe_release_url_falls_back_to_releases_page(url):
    info = uc.parse_release({**SAMPLE, "html_url": url}, "1.0.0")
    assert info.url == "https://github.com/TariqMK/Verbatica/releases"


@pytest.mark.parametrize("data", [None, [], {"tag_name": 5}, {"tag_name": "latest"}])
def test_unreadable_response(data):
    with pytest.raises(uc.UpdateError, match="can't read"):
        uc.parse_release(data, "1.0.0")


def http_error(code):
    return urllib.error.HTTPError(uc.LATEST_URL, code, "error", {}, io.BytesIO())


@pytest.mark.parametrize("error, message", [
    (http_error(404), "No Verbatica releases are published yet."),
    (http_error(403), "GitHub limits how often it can be checked. Try again in an hour."),
    (http_error(429), "GitHub limits how often it can be checked. Try again in an hour."),
    (http_error(500), "GitHub answered with error 500. Try again later."),
    (urllib.error.URLError("no route"), "Can't reach GitHub: no route. Check the internet connection."),
    (TimeoutError("timed out"), "Can't reach GitHub: timed out. Check the internet connection."),
])
def test_error_messages(monkeypatch, error, message):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(uc.urllib.request, "urlopen", fail)
    with pytest.raises(uc.UpdateError) as caught:
        uc.check("1.0.0")
    assert str(caught.value) == message


def test_request_sends_only_user_agent_and_accept(monkeypatch):
    sent = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        sent["url"], sent["headers"], sent["timeout"] = request.full_url, dict(request.header_items()), timeout
        sent["method"], sent["data"] = request.get_method(), request.data
        return Response(b'{"tag_name": "v2.0.0", "html_url": "https://github.com/x"}')

    monkeypatch.setattr(uc.urllib.request, "urlopen", fake_urlopen)
    info = uc.check("1.0.0")
    assert info.newer
    assert sent["url"] == "https://api.github.com/repos/TariqMK/Verbatica/releases/latest"
    assert sent["method"] == "GET" and sent["data"] is None and sent["timeout"] == 10.0
    assert set(sent["headers"]) == {"User-agent", "Accept"}


def test_garbage_body(monkeypatch):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(uc.urllib.request, "urlopen", lambda *a, **k: Response(b"<html>"))
    with pytest.raises(uc.UpdateError, match="can't read"):
        uc.check("1.0.0")
