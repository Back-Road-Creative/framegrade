"""Reading the published-titles ledger must distinguish a MISSING
ledger (fresh channel — legitimately no history) from an UNREADABLE one
(OSError/decode error — dedup protection can't be verified), and must emit a
greppable WARN when it can't be read. The ``"test"`` skip-filter must be
word-boundary so a real title like ``"Testament Creek"`` is not dropped.

``DEFAULT_SEO_STORAGE`` is redirected to a fresh per-test tempdir by the
autouse ``isolate_seo_storage_from_workspace`` conftest fixture.
"""

import json
import logging
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline import longform_metadata as lm


def _ledger_path():
    return lm.DEFAULT_SEO_STORAGE / "gomoveshift" / "events.jsonl"


def _write_ledger(rows):
    events = _ledger_path()
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return events


def _make_unreadable_ledger():
    """Create a directory where the ledger file is expected.

    ``events.exists()`` is True but ``open()`` raises ``IsADirectoryError``
    (an ``OSError`` subclass) — a portable way to force the unreadable path
    without relying on chmod (no-op for root in containers).
    """
    events = _ledger_path()
    events.parent.mkdir(parents=True, exist_ok=True)
    events.mkdir()
    return events


def test_missing_ledger_returns_missing_status():
    titles, status = lm.read_published_titles()
    assert titles == set()
    assert status == lm.LEDGER_MISSING


def test_unreadable_ledger_warns_and_returns_unreadable(caplog):
    _make_unreadable_ledger()
    with caplog.at_level(logging.WARNING):
        titles, status = lm.read_published_titles()
    assert titles == set()
    assert status == lm.LEDGER_UNREADABLE
    assert any("unreadable" in r.getMessage().lower() for r in caplog.records)


def test_load_published_titles_degrades_and_warns_on_unreadable(caplog):
    _make_unreadable_ledger()
    with caplog.at_level(logging.WARNING):
        titles = lm.load_published_titles()
    assert titles == set()
    assert any("unreadable" in r.getMessage().lower() for r in caplog.records)


def test_refresh_channel_title_ledger_merges_fake_client_title():
    """A fake client returning one live title surfaces it
    (normalized, with its video id) for the BC dedup refresh."""
    fake_sync = MagicMock()
    fake_sync.youtube_data.channels.return_value.list.return_value.execute.return_value = {
        "items": [{"contentDetails": {"relatedPlaylists": {"uploads": "PL123"}}}]
    }
    fake_sync.youtube_data.playlistItems.return_value.list.return_value.execute.return_value = {
        "items": [
            {
                "snippet": {
                    "title": "Pennsylvania Scenic Drive in 4K | Gettysburg, PA",
                    "resourceId": {"videoId": "vidA"},
                }
            }
        ],
        "nextPageToken": None,
    }
    with (
        patch(
            "gomoveshift.pipeline._yt_creds.load_youtube_credentials",
            return_value={"token": "fake"},
        ),
        patch(
            "seo_manager.analytics.youtube_sync.YouTubeAnalyticsSync",
            return_value=fake_sync,
        ),
    ):
        titles = lm.refresh_channel_title_ledger()
    assert titles == {"pennsylvania scenic drive in 4k | gettysburg, pa": "vidA"}


def test_refresh_channel_title_ledger_no_creds_warns_and_continues(caplog):
    """Offline/credential-less runs must not hard-fail Phase BC — they
    degrade to an empty mapping with a WARNING naming the risk."""
    with (
        patch(
            "gomoveshift.pipeline._yt_creds.load_youtube_credentials",
            side_effect=FileNotFoundError("no creds"),
        ),
        caplog.at_level(logging.WARNING),
    ):
        titles = lm.refresh_channel_title_ledger()
    assert titles == {}
    assert any("credentials unavailable" in r.getMessage().lower() for r in caplog.records)


def test_test_word_boundary_keeps_testament_creek():
    _write_ledger(
        [
            {"data": {"title": "Testament Creek, WV in 4K"}},
            {"data": {"title": "Test Drive"}},  # standalone word -> filtered
            {"data": {"title": "TEST upload"}},  # standalone word -> filtered
            {"data": {"title": "Blue Ridge Parkway, NC in 4K"}},
        ]
    )
    titles = lm.load_published_titles()
    assert "Testament Creek, WV in 4K" in titles
    assert "Blue Ridge Parkway, NC in 4K" in titles
    assert "Test Drive" not in titles
    assert "TEST upload" not in titles
