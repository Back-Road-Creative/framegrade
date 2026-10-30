"""A pending entry whose journey file is gone is flushed into the journey that
holds its day, never kept pending forever.

Ace Basin (2026-02-21) deferred both long-forms into a draft
``usa-2026-02-21`` the drive hook made because no trip range held the run
date; the day belongs to ``sc-2026``. With the draft set aside, replay kept
both entries pending every hour with no way to land them. Replay now resolves
the entry's day the way the drive hook does (a trip holding the date, else a
trip that ended the day before) and writes there.
"""

from __future__ import annotations

import json

import yaml

from gomoveshift.cli.handlers.publish_journal import run_replay_pending
from gomoveshift.pipeline.journal_publisher import load_journey
from gomoveshift.pipeline.pending_journal import PENDING_FILENAME, load_pending

_DAY1 = """---
title: "Lowcountry, 2026"
draft: false
trip:
  start_date: '2026-02-20'
  end_date: '2026-02-20'
  duration_days: 1
days:
- day_number: 1
  date: '2026-02-20'
  day_of_week: Friday
  heading: "Beaufort"
  narrative: ""
  videos: []
video_count: 0
---
body text
"""


def _site(tmp_path, **journeys):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    for slug, text in journeys.items():
        (jdir / f"{slug}.md").write_text(text)
    return site


def _entry(site, vid, day, publish_at="2026-01-01T00:00:00Z", slug="usa-2026-02-21", lf=1):
    return {
        "youtube_id": vid,
        "publish_at": publish_at,
        "title": f"{vid} title",
        "duration": "10:00",
        "day_date": day,
        "journey_path": str(site / "content" / "journeys" / f"{slug}.md"),
        "lf_index": lf,
    }


def _seed(tmp_path, entries):
    out = tmp_path / "out"
    out.mkdir()
    path = out / PENDING_FILENAME
    path.write_text(json.dumps(entries))
    return path


def _ids(site, slug):
    doc, _ = load_journey(site / "content" / "journeys" / f"{slug}.md")
    return {d["date"]: [v["youtube_id"] for v in d.get("videos") or []] for d in doc["days"]}


def test_missing_journey_flushes_into_the_trip_ending_the_day_before(tmp_path):
    site = _site(tmp_path, **{"sc-2026": _DAY1})
    pending = _seed(tmp_path, [_entry(site, "LF1", "2026-02-21")])

    rc = run_replay_pending(pending_files=[pending], apply=True)

    assert rc == 0
    assert _ids(site, "sc-2026") == {"2026-02-20": [], "2026-02-21": ["LF1"]}
    assert not pending.exists() or load_pending(pending) == []
    routes = yaml.safe_load((site / "data" / "video_routing.yaml").read_text())["routes"]
    assert routes["LF1"] == {"journey": "sc-2026", "date": "2026-02-21"}


def test_two_entries_on_the_same_missing_day_land_in_one_extended_trip(tmp_path):
    site = _site(tmp_path, **{"sc-2026": _DAY1})
    pending = _seed(
        tmp_path, [_entry(site, "LF1", "2026-02-21"), _entry(site, "LF2", "2026-02-21", lf=2)]
    )

    assert run_replay_pending(pending_files=[pending], apply=True) == 0

    assert sorted(_ids(site, "sc-2026")["2026-02-21"]) == ["LF1", "LF2"]
    doc, _ = load_journey(site / "content" / "journeys" / "sc-2026.md")
    assert doc["trip"]["end_date"] == "2026-02-21"


def test_future_entry_keeps_its_stored_path_until_it_is_due(tmp_path):
    site = _site(tmp_path, **{"sc-2026": _DAY1})
    future = _entry(site, "LF2", "2026-02-21", publish_at="2999-01-01T00:00:00Z")
    pending = _seed(tmp_path, [future])

    assert run_replay_pending(pending_files=[pending], apply=True) == 0

    assert _ids(site, "sc-2026") == {"2026-02-20": []}
    assert [e.journey_path for e in load_pending(pending)] == [future["journey_path"]]


def test_missing_journey_with_no_trip_for_the_day_stays_pending(tmp_path):
    site = _site(tmp_path, **{"sc-2026": _DAY1})
    stranded = _entry(site, "LF9", "2026-03-15")
    pending = _seed(tmp_path, [stranded])

    run_replay_pending(pending_files=[pending], apply=True)

    assert [e.youtube_id for e in load_pending(pending)] == ["LF9"]
    assert _ids(site, "sc-2026") == {"2026-02-20": []}


def test_dry_run_neither_extends_the_trip_nor_consumes_the_entry(tmp_path):
    site = _site(tmp_path, **{"sc-2026": _DAY1})
    before = (site / "content" / "journeys" / "sc-2026.md").read_text()
    pending = _seed(tmp_path, [_entry(site, "LF1", "2026-02-21")])

    run_replay_pending(pending_files=[pending], apply=False)

    assert (site / "content" / "journeys" / "sc-2026.md").read_text() == before
    assert [e.youtube_id for e in load_pending(pending)] == ["LF1"]


def test_ambiguous_day_keeps_the_entry_pending(tmp_path):
    other = _DAY1.replace("Lowcountry", "Other")
    site = _site(tmp_path, **{"sc-2026": _DAY1, "other-2026": other})
    pending = _seed(tmp_path, [_entry(site, "LF1", "2026-02-21")])

    run_replay_pending(pending_files=[pending], apply=True)

    assert [e.youtube_id for e in load_pending(pending)] == ["LF1"]
    assert _ids(site, "sc-2026") == {"2026-02-20": []}
