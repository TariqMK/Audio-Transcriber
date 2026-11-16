"""Checks GitHub for a newer Audio Transcriber release. Runs only when the user selects
the version number; nothing is downloaded, installed or stored."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from audio_transcriber import APP_NAME, REPOSITORY, __version__

LATEST_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPOSITORY}/releases"
TIMEOUT_S = 10.0
USER_AGENT = f"{APP_NAME.replace(' ', '-')}/{__version__}"  # GitHub refuses API requests without one.

_RELEASE_TAG = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
_UNREADABLE = f"GitHub sent an answer {APP_NAME} can't read. Try again later."


class UpdateError(Exception):
    """The check failed. The message is written for the user."""


@dataclass(frozen=True)
class UpdateInfo:
    current: str  # The running version, for example "1.2.3" or "0.0.0.dev0".
    latest: str  # The newest release tag, for example "v1.3.0".
    url: str  # The release page on GitHub.
    notes: str  # Release notes as written on GitHub (Markdown).

    @property
    def dev_build(self) -> bool:
        """True for a source run or local build, which has no real version to compare."""
        return parse_version(self.current) is None

    @property
    def newer(self) -> bool:
        return not self.dev_build and is_newer(self.latest, self.current)


def parse_version(text: str) -> tuple | None:
    """(major, minor, patch) for "1.2.3" or "v1.2.3"; None for anything else, such as a dev version."""
    match = _RELEASE_TAG.match(text.strip())
    return tuple(int(part) for part in match.groups()) if match else None


def is_newer(latest: str, current: str) -> bool:
    """True when latest is a higher release than current. Numbers are compared, so 0.10.0 > 0.9.0."""
    latest_parts = parse_version(latest)
    current_parts = parse_version(current)
    if latest_parts is None or current_parts is None:
        return False
    return latest_parts > current_parts


def parse_release(data: object, current: str) -> UpdateInfo:
    """Build the result from the GitHub "latest release" response."""
    if not isinstance(data, dict) or not isinstance(data.get("tag_name"), str):
        raise UpdateError(_UNREADABLE)
    tag = data["tag_name"]
    if parse_version(tag) is None:
        raise UpdateError(_UNREADABLE)
    url = data.get("html_url")
    notes = data.get("body")
    return UpdateInfo(
        current=current,
        latest=tag,
        url=url if isinstance(url, str) and url.startswith("https://") else RELEASES_PAGE,
        notes=notes.strip() if isinstance(notes, str) else "",
    )


def check(current: str = __version__) -> UpdateInfo:
    """Ask GitHub for the latest release. Blocks, so call it on a background thread."""
    request = urllib.request.Request(
        LATEST_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise UpdateError(f"No {APP_NAME} releases are published yet.") from error
        if error.code in (403, 429):
            raise UpdateError("GitHub limits how often it can be checked. Try again in an hour.") from error
        raise UpdateError(f"GitHub answered with error {error.code}. Try again later.") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise UpdateError(f"Can't reach GitHub: {reason}. Check the internet connection.") from error
    try:
        data = json.loads(raw)
    except ValueError as error:
        raise UpdateError(_UNREADABLE) from error
    return parse_release(data, current)
