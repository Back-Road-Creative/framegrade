"""``gomoveshift library-hygiene``: list the public, low-view,
templated shorts from the LIVE uploads listing, and on ``--apply`` set the first
``--max`` private, heaviest series first, writing a ledger line (prior status)
BEFORE each call so ``--undo`` can restore it. Dry-run by default; ``--ids``
privates an explicit list (unlisted test renders)."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import httplib2
import pytest
from googleapiclient.errors import HttpError

from gomoveshift.cli.handlers import library_hygiene as lh
from gomoveshift.cli.handlers.library_hygiene import (
    LiveVideo,
    cmd_library_hygiene,
    parse_iso8601_duration,
    series_from_title,
    tracked_shorts,
)

TODAY = date(2026, 9, 25)


# --------------------------------------------------------------------------- fakes


class _Req:
    def __init__(self, payload):
        self._payload = payload

    def execute(self, **_kwargs):
        return self._payload


class _FakeService:
    """Just enough of the Data API: one channel, one uploads playlist, N videos."""

    def __init__(self, videos):
        self._videos = videos
        # playlistId -> ids it lists; unset ids 404 (default: the plain listing has every video).
        self.listings = {"UUx": [v["id"] for v in videos]}

    def channels(self):
        return SimpleNamespace(
            list=lambda **kw: _Req(
                {
                    "items": [
                        {"id": "UCx", "contentDetails": {"relatedPlaylists": {"uploads": "UUx"}}}
                    ]
                }
            )
        )

    def playlistItems(self):
        def _list(**kw):
            if kw["playlistId"] not in self.listings:
                raise HttpError(httplib2.Response({"status": 404}), b"{}")
            ids = self.listings[kw["playlistId"]]
            return _Req({"items": [{"snippet": {"resourceId": {"videoId": i}}} for i in ids]})

        return SimpleNamespace(list=_list)

    def videos(self):
        def _list(**kw):
            wanted = kw["id"].split(",")
            return _Req(
                {
                    "items": [
                        {
                            "id": v["id"],
                            "snippet": {
                                "channelId": "UCx",
                                "title": v["title"],
                                "publishedAt": v["published"] + "T10:00:00Z",
                            },
                            "status": {"privacyStatus": v["privacy"]},
                            "statistics": {"viewCount": str(v["views"])},
                            "contentDetails": {"duration": v.get("duration", "PT39S")},
                        }
                        for v in self._videos
                        if v["id"] in wanted
                    ]
                }
            )

        return SimpleNamespace(list=_list)


class _FakeUploader:
    def __init__(self, videos):
        self.service = _FakeService(videos)
        self.calls = []

    def _build_service(self):
        return self.service

    def update_privacy(self, video_id, *, privacy, publish_at=None):
        self.calls.append((video_id, privacy))


def _video(id_, title, *, views=5, published="2026-06-20", privacy="public", duration="PT39S"):
    return {
        "id": id_,
        "title": title,
        "views": views,
        "published": published,
        "privacy": privacy,
        "duration": duration,
    }


def _heavy(series, n, **kw):
    return [_video(f"{series[:3]}{i}", f"Open Road: {series} 🚗", **kw) for i in range(n)]


def _args(**kw):
    base = {"apply": False, "max": 50, "ids": None, "undo": None, "ledger": None}
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Install a fake uploader over ``videos`` and point the ledger at tmp."""
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(lh, "_today", lambda: TODAY)

    def _wire(videos):
        fake = _FakeUploader(videos)
        monkeypatch.setattr(lh, "YouTubeUploader", lambda: fake)
        return fake

    return _wire


# --------------------------------------------------------------------- pure parts


def test_series_is_the_location_captured_by_the_shortest_template_match():
    assert series_from_title("Open Road: Example County 🚗") == "Example County"
    assert series_from_title("Behind the Wheel — Sampleton 🚗") == "Sampleton"
    assert series_from_title("Mockwood County — Raw Dashcam") == "Mockwood County"


def test_an_untemplated_title_has_no_series_and_is_never_guessed():
    assert series_from_title("Example Forest Drive: Road 12 (Alpha → Beta)") is None
    assert series_from_title("untemplated clip two") is None
    assert series_from_title("untemplated clip three") is None


def test_iso8601_durations_parse_to_seconds():
    assert parse_iso8601_duration("PT39S") == 39
    assert parse_iso8601_duration("PT11M18S") == 678
    assert parse_iso8601_duration("PT11H54M56S") == 42896


def test_tracked_shorts_drop_long_videos_and_untemplated_titles():
    long_form = LiveVideo("L", "Example County Scenic Loop, SC", TODAY, "public", 19, 574)
    untemplated = LiveVideo("U", "untemplated clip four", TODAY, "public", 3, 39)
    short = LiveVideo("S", "Leaving Example County", TODAY, "public", 3, 39)
    rows = tracked_shorts([long_form, untemplated, short])
    assert [r.id for r in rows] == ["S"]
    assert rows[0].series == "Example County" and rows[0].privacy == "public"


# ----------------------------------------------------------------- the command


def test_dry_run_lists_candidates_heaviest_series_first_and_changes_nothing(wired, capsys):
    fake = wired(_heavy("Example County", 8) + _heavy("Sampleton", 6) + _heavy("Small", 3))
    assert cmd_library_hygiene(_args(), config=None) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("LIBRARY_HYGIENE_CANDIDATE")]
    assert len(lines) == 14
    assert all("Example County" in ln for ln in lines[:8])
    assert all("Sampleton" in ln for ln in lines[8:])
    assert "14 candidates" in out and "--apply" in out
    assert fake.calls == []


def test_apply_privates_at_most_max_writes_the_ledger_first_and_names_it(wired, capsys, tmp_path):
    fake = wired(_heavy("Example County", 8) + _heavy("Sampleton", 6))
    assert cmd_library_hygiene(_args(apply=True, max=10), config=None) == 0
    assert len(fake.calls) == 10 and all(p == "private" for _, p in fake.calls)
    assert all(v.startswith("Exa") for v, _ in fake.calls[:8])
    ledger = tmp_path / "library-hygiene" / "2026-09-25-ledger.jsonl"
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    assert len(rows) == 10
    assert rows[0]["prior_privacy"] == "public" and rows[0]["new_privacy"] == "private"
    assert rows[0]["video_id"] == fake.calls[0][0]
    assert str(ledger) in capsys.readouterr().out


def test_apply_default_cap_is_fifty(wired):
    fake = wired(_heavy("Example County", 60))
    assert cmd_library_hygiene(_args(apply=True), config=None) == 0
    assert len(fake.calls) == 50


def test_recent_or_watched_or_non_public_shorts_are_never_touched(wired):
    fake = wired(
        _heavy("Example County", 6, published="2026-09-20")  # too young
        + _heavy("Sampleton", 6, views=250)  # watched
        + _heavy("Mockwood", 6, privacy="unlisted")  # not public
    )
    assert cmd_library_hygiene(_args(apply=True), config=None) == 0
    assert fake.calls == []


def test_ids_privates_the_named_videos_whatever_their_selection_state(wired, capsys, tmp_path):
    fake = wired(
        [
            _video("vidA", "untemplated clip one", privacy="unlisted", published="2026-09-24"),
            _video("vidB", "untemplated clip two", privacy="unlisted", published="2026-09-24"),
            _video("done", "already private", privacy="private"),
        ]
    )
    assert cmd_library_hygiene(_args(apply=True, ids=["vidA", "vidB", "done"]), config=None) == 0
    assert fake.calls == [("vidA", "private"), ("vidB", "private")]
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "library-hygiene" / "2026-09-25-ledger.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [r["prior_privacy"] for r in rows] == ["unlisted", "unlisted"]
    assert "already private" in capsys.readouterr().out


def test_ids_not_in_the_channel_refuse_before_any_change(wired, capsys):
    fake = wired([_video("a", "untemplated clip two", privacy="unlisted")])
    assert cmd_library_hygiene(_args(apply=True, ids=["a", "nope"]), config=None) == 2
    assert fake.calls == [] and "nope" in capsys.readouterr().err


def test_ids_without_apply_only_shows_what_would_change(wired, capsys):
    fake = wired([_video("a", "untemplated clip two", privacy="unlisted")])
    assert cmd_library_hygiene(_args(ids=["a"]), config=None) == 0
    assert fake.calls == [] and "unlisted" in capsys.readouterr().out


def test_undo_restores_each_ledger_rows_prior_status(wired, tmp_path):
    fake = wired([])
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"video_id": "x", "prior_privacy": "public", "new_privacy": "private"})
        + "\n"
        + json.dumps({"video_id": "y", "prior_privacy": "unlisted", "new_privacy": "private"})
        + "\n"
    )
    assert cmd_library_hygiene(_args(undo=str(ledger)), config=None) == 0
    assert fake.calls == [("x", "public"), ("y", "unlisted")]


def test_parser_registers_the_command_with_a_dry_run_default():
    from gomoveshift.cli.parser import _build_parser

    args = _build_parser().parse_args(["library-hygiene"])
    assert args.apply is False and args.max == 50 and args.ids is None and args.undo is None
    args = _build_parser().parse_args(
        ["library-hygiene", "--apply", "--ids", "a", "b", "--max", "3"]
    )
    assert args.apply is True and args.ids == ["a", "b"] and args.max == 3


# ------------------------------------------------- retired shapes absent from the pools


@pytest.mark.parametrize(
    "title, location",
    [
        ("4K Dashcam: Fableport County", "Fableport County"),
        (
            "4K Road Trip — Sampleton to Ballylickey via the Famine Tunnel",
            "Sampleton to Ballylickey via the Famine Tunnel",
        ),
        ("First Time Driving Fableport County", "Fableport County"),
        ("Through Example County in 4K", "Example County"),
    ],
)
def test_the_four_pre_pool_shapes_seen_in_practice_resolve_a_series(title, location):
    assert series_from_title(title) == location


def test_the_pre_pool_shapes_are_declared_beside_the_retired_pool_not_inside_it():
    from gomoveshift.publishing.anti_spam_gate import LEGACY_TITLE_POOL_RAW, PRE_POOL_TITLE_SHAPES

    assert len(PRE_POOL_TITLE_SHAPES) == 4
    assert not set(PRE_POOL_TITLE_SHAPES) & set(LEGACY_TITLE_POOL_RAW)


def test_ledger_rows_carry_the_series(wired, tmp_path):
    wired(_heavy("Example County", 6))
    assert cmd_library_hygiene(_args(apply=True, max=2), config=None) == 0
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "library-hygiene" / "2026-09-25-ledger.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [r["series"] for r in rows] == ["Example County", "Example County"]


def test_list_uploads_sees_videos_only_the_format_split_playlists_list(tmp_path, monkeypatch):
    """The plain uploads listing omits videos; hygiene must union the splits."""
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))
    service = _FakeService([_video("plain", "Example County 1"), _video("omitted", "Example 2")])
    service.listings = {"UUx": ["plain"], "UUSHx": ["omitted", "plain"]}

    assert [v.id for v in lh.list_uploads(service)] == ["plain", "omitted"]


def test_ledger_ids_reads_every_ledger_and_skips_bad_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))
    d = tmp_path / "library-hygiene"
    d.mkdir()
    (d / "2026-09-01-ledger.jsonl").write_text(
        json.dumps({"video_id": "one"}) + "\nnot json\n" + json.dumps({"id": "two"}) + "\n"
    )
    (d / "2026-09-02-ledger.jsonl").write_text(json.dumps({"video_id": "one"}) + "\n\n")
    (d / "notes.txt").write_text(json.dumps({"video_id": "ignored"}))

    assert lh.ledger_video_ids() == ["one", "two"]


def test_ledger_ids_is_empty_without_a_ledger_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "nowhere"))

    assert lh.ledger_video_ids() == []


def test_list_uploads_sees_a_privated_video_only_the_ledger_remembers(tmp_path, monkeypatch):
    """A private short no listing returns is still on the channel: the hygiene ledger
    (every video this command ever privated) names it, videos.list confirms it."""
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "library-hygiene").mkdir()
    (tmp_path / "library-hygiene" / "2026-09-25-ledger.jsonl").write_text(
        json.dumps({"video_id": "hidden"}) + "\n"
    )
    service = _FakeService(
        [_video("plain", "Example County 1"), _video("hidden", "Example 2", privacy="private")]
    )
    service.listings = {"UUx": ["plain"]}

    assert sorted(v.id for v in lh.list_uploads(service)) == ["hidden", "plain"]


def test_ids_batch_accepts_a_private_video_only_the_ledger_knows(wired, tmp_path, capsys):
    (tmp_path / "library-hygiene").mkdir()
    (tmp_path / "library-hygiene" / "2026-09-20-ledger.jsonl").write_text(
        json.dumps({"video_id": "SYNTH000020"}) + "\n"
    )
    fake = wired([_video("SYNTH000020", "Example 2", privacy="private")])
    fake.service.listings = {"UUx": []}  # no listing returns it

    assert cmd_library_hygiene(_args(ids=["SYNTH000020"]), config=None) == 0
    assert "already private" in capsys.readouterr().out


def test_list_uploads_unions_the_ledger_with_the_frozen_seed(tmp_path, monkeypatch):
    """Ids the ledgers never recorded (a private short from before the ledger existed)
    still resolve: the frozen seed is always asked for alongside the ledger ids."""
    monkeypatch.setenv("GMS_OUTPUT_DIR", str(tmp_path / "nowhere"))
    monkeypatch.setattr(lh, "load_known_ids", lambda: ("seeded",))
    asked = {}

    def fake_enumerate(service, *, known_ids=()):
        asked["ids"] = list(known_ids)
        return {}

    monkeypatch.setattr(lh, "enumerate_uploads", fake_enumerate)

    lh.list_uploads(object())

    assert asked["ids"] == ["seeded"]
