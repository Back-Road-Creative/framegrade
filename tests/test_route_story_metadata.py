"""Tests for the Route Story metadata builder.

``build_route_story_metadata`` turns a film receipt (``film_receipt`` in
``route_story/film.py``) into the title/description/tags a video upload
needs, plus a ``format``/``playlist_hint`` a caller writes onto upload
metadata for ``playlist_resolver.resolve_playlist`` to match a
``route_story`` rule against.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline.route_story.metadata import (
    DuplicateTitleError,
    LedgerUnreadableError,
    build_route_story_metadata,
    check_title_against_ledger,
    preflight_title,
)


def _receipt(**overrides):
    base = {
        "route_summary": {
            "distance_m": 19_312.0,  # ~12.0 mi / 19.3 km
            "elapsed_seconds": 7_200.0,
            "moving_seconds": 6_300.0,  # 1h 45m
            "stopped_seconds": 900.0,
            "ascent_m": 365.76,  # ~1200 ft
            "descent_m": 300.0,
            "max_elevation_m": 900.0,
            "moving_average_speed_mps": 3.0,  # hiking band
        },
        "map_labels": {
            "count": 2,
            "names": ["Trailhead Park", "Summit Overlook"],
            "kinds": ["park", "overlook"],
        },
        "photos": [],
        "local_imagery": {"credit": "Local County GIS"},
        "regional_imagery": [{"credit": "State Regional Survey"}],
        "national_imagery": {"credit": "USGS"},
    }
    base.update(overrides)
    return base


def test_explicit_title_used_verbatim():
    meta = build_route_story_metadata(
        _receipt(), recorded_on="2026-06-01", title="  My Own Title  "
    )
    assert meta.title == "My Own Title"


def test_derived_title_uses_activity_and_first_last_labels():
    meta = build_route_story_metadata(_receipt(), recorded_on="2026-06-01")
    assert meta.title == "Hiking Trailhead Park → Summit Overlook"


def test_derived_title_single_label_no_arrow():
    receipt = _receipt(map_labels={"count": 1, "names": ["Ridge Trail"], "kinds": ["trail"]})
    meta = build_route_story_metadata(receipt, recorded_on="2026-06-01")
    assert meta.title == "Hiking Ridge Trail"


def test_derived_title_falls_back_to_date_with_no_labels():
    receipt = _receipt(map_labels={"count": 0, "names": [], "kinds": []})
    meta = build_route_story_metadata(receipt, recorded_on="2026-06-01")
    assert meta.title == "Hiking 2026-06-01"


@pytest.mark.parametrize(
    "speed_mps,activity",
    [
        (1.0, "Hiking"),
        (5.0, "Riding"),
        (12.0, "Driving"),
        (None, "Driving"),
    ],
)
def test_activity_bands_from_moving_average_speed(speed_mps, activity):
    receipt = _receipt()
    receipt["route_summary"]["moving_average_speed_mps"] = speed_mps
    meta = build_route_story_metadata(receipt, recorded_on="2026-06-01")
    assert meta.title.startswith(activity)


def test_explicit_activity_wins_over_moving_average_speed():
    receipt = _receipt()
    receipt["route_summary"]["moving_average_speed_mps"] = 3.0  # hiking band
    meta = build_route_story_metadata(receipt, recorded_on="2026-06-01", activity="cycling")
    assert meta.title == "Cycling Trailhead Park → Summit Overlook"
    assert "Cycling" in meta.tags


def test_description_reports_distance_ascent_time_date_and_credits_in_order():
    meta = build_route_story_metadata(
        _receipt(), recorded_on="2026-06-01", credits=("Drone Footage",)
    )
    lines = meta.description.splitlines()
    assert any("12.0 mi" in line and "19.3 km" in line for line in lines)
    assert any("1200 ft" in line and "366 m" in line for line in lines)
    assert any("1h 45m" in line for line in lines)
    assert any("2026-06-01" in line for line in lines)
    credit_lines = [line for line in lines if line.startswith("Imagery:")]
    assert credit_lines == [
        "Imagery: USGS",
        "Imagery: State Regional Survey",
        "Imagery: Local County GIS",
        "Imagery: Drone Footage",
    ]


def test_description_credits_order_receipt_then_extra_credits_arg():
    meta = build_route_story_metadata(
        _receipt(), recorded_on="2026-06-01", credits=("Drone Footage", "Bonus Credit")
    )
    credit_lines = [line for line in meta.description.splitlines() if line.startswith("Imagery:")]
    assert credit_lines[-2:] == ["Imagery: Drone Footage", "Imagery: Bonus Credit"]


def test_tags_include_activity_route_story_and_labels_deduped():
    receipt = _receipt(
        map_labels={
            "count": 3,
            "names": ["Trailhead Park", "Summit Overlook", "trailhead park"],
            "kinds": ["park", "overlook", "park"],
        }
    )
    meta = build_route_story_metadata(receipt, recorded_on="2026-06-01")
    assert "hiking" in [t.lower() for t in meta.tags]
    assert "route story" in [t.lower() for t in meta.tags]
    lowered = [t.lower() for t in meta.tags]
    assert lowered.count("trailhead park") == 1


def test_tags_capped_at_fifteen_and_thirty_chars_each():
    long_names = [f"Extremely Long Label Name Number {i} Overflow" for i in range(30)]
    receipt = _receipt(map_labels={"count": len(long_names), "names": long_names, "kinds": []})
    meta = build_route_story_metadata(receipt, recorded_on="2026-06-01")
    assert len(meta.tags) <= 15
    assert all(len(tag) <= 30 for tag in meta.tags)
    assert sum(len(tag) for tag in meta.tags) <= 500


def test_format_and_playlist_hint_are_route_story():
    meta = build_route_story_metadata(_receipt(), recorded_on="2026-06-01")
    assert meta.format == "route_story"
    assert meta.playlist_hint == "route_story"
    assert meta.recording_date == "2026-06-01"


def test_duplicate_title_raises():
    with pytest.raises(DuplicateTitleError):
        build_route_story_metadata(
            _receipt(),
            recorded_on="2026-06-01",
            title="Taken Title",
            published_titles={"Taken Title"},
        )


def test_ledger_unreadable_status_raises():
    with pytest.raises(LedgerUnreadableError):
        build_route_story_metadata(
            _receipt(),
            recorded_on="2026-06-01",
            title="Fine Title",
            ledger_status=lm.LEDGER_UNREADABLE,
        )


def test_missing_ledger_status_is_ok():
    meta = build_route_story_metadata(
        _receipt(),
        recorded_on="2026-06-01",
        title="Fine Title",
        ledger_status=lm.LEDGER_MISSING,
    )
    assert meta.title == "Fine Title"


def test_check_title_against_ledger_reads_real_ledger(monkeypatch):
    def fake_read_published_titles():
        return {"Already Published"}, lm.LEDGER_OK

    monkeypatch.setattr(lm, "read_published_titles", fake_read_published_titles)
    meta = build_route_story_metadata(
        _receipt(), recorded_on="2026-06-01", title="Already Published"
    )
    with pytest.raises(DuplicateTitleError):
        check_title_against_ledger(meta)


def test_check_title_against_ledger_passes_when_unique(monkeypatch):
    def fake_read_published_titles():
        return {"Some Other Title"}, lm.LEDGER_OK

    monkeypatch.setattr(lm, "read_published_titles", fake_read_published_titles)
    meta = build_route_story_metadata(_receipt(), recorded_on="2026-06-01", title="Fine Title")
    check_title_against_ledger(meta)  # must not raise


def test_preflight_title_derives_the_same_title_as_build(monkeypatch):
    monkeypatch.setattr(lm, "read_published_titles", lambda: (set(), lm.LEDGER_OK))
    meta = build_route_story_metadata(_receipt(), recorded_on="2026-06-01", activity="cycling")

    resolved = preflight_title(
        recorded_on="2026-06-01",
        title=None,
        activity="cycling",
        names=["Trailhead Park", "Summit Overlook"],
    )

    assert resolved == meta.title == "Cycling Trailhead Park → Summit Overlook"


def test_preflight_title_raises_duplicate_before_any_render_context(monkeypatch):
    monkeypatch.setattr(lm, "read_published_titles", lambda: ({"Fine Title"}, lm.LEDGER_OK))

    with pytest.raises(DuplicateTitleError):
        preflight_title(
            recorded_on="2026-06-01",
            title="Fine Title",
            activity=None,
            names=[],
        )


def test_preflight_title_raises_when_ledger_unreadable(monkeypatch):
    monkeypatch.setattr(lm, "read_published_titles", lambda: (set(), lm.LEDGER_UNREADABLE))

    with pytest.raises(LedgerUnreadableError):
        preflight_title(
            recorded_on="2026-06-01",
            title="Fine Title",
            activity=None,
            names=[],
        )
