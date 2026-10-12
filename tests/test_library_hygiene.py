"""Library hygiene: a weekly, suggest-only list of public shorts that are
old, low-viewed and part of a heavy series. Pure selection — this module never changes any platform's
privacy; ``gomoveshift library-hygiene --apply`` (``test_library_hygiene_cli.py``) does."""

from __future__ import annotations

from datetime import date

from gomoveshift.tracking.schema import ShortRecord, TrackingRecord
from gomoveshift.tracking.store import TrackingStore
from gomoveshift.tracking.volume import (
    DEFAULT_MAX_VIEWS,
    DEFAULT_MIN_AGE_DAYS,
    DEFAULT_MIN_SERIES_SIZE,
    TrackedShort,
    library_hygiene_report,
    main as hygiene_main,
    select_library_hygiene_candidates,
)

TODAY = date(2026, 9, 22)


def _short(id_, days_old, privacy="public", views=10, series="Example County"):
    return TrackedShort(
        id=id_,
        published_at=date.fromordinal(TODAY.toordinal() - days_old),
        privacy=privacy,
        views=views,
        series=series,
    )


def _heavy_series(n, series="Example County", days_old=60, **kwargs):
    return [_short(f"{series}-{i}", days_old=days_old, series=series, **kwargs) for i in range(n)]


def test_excludes_shorts_31_days_old_or_younger():
    shorts = _heavy_series(6, days_old=DEFAULT_MIN_AGE_DAYS)  # exactly at the floor
    batches = select_library_hygiene_candidates(shorts, TODAY)
    assert batches == []


def test_includes_shorts_older_than_the_floor():
    old = [_short(f"s{i}", days_old=DEFAULT_MIN_AGE_DAYS + 1) for i in range(6)]
    batches = select_library_hygiene_candidates(old, TODAY)
    assert sum(len(b) for b in batches) == 6


def test_excludes_views_at_or_above_the_ceiling():
    shorts = [_short(f"s{i}", days_old=60, views=DEFAULT_MAX_VIEWS) for i in range(6)]
    assert select_library_hygiene_candidates(shorts, TODAY) == []


def test_includes_views_below_the_ceiling():
    shorts = [_short(f"s{i}", days_old=60, views=DEFAULT_MAX_VIEWS - 1) for i in range(6)]
    batches = select_library_hygiene_candidates(shorts, TODAY)
    assert sum(len(b) for b in batches) == 6


def test_a_series_of_exactly_the_floor_size_is_excluded():
    shorts = [_short(f"s{i}", days_old=60) for i in range(DEFAULT_MIN_SERIES_SIZE)]
    assert select_library_hygiene_candidates(shorts, TODAY) == []


def test_a_series_larger_than_the_floor_qualifies():
    shorts = [_short(f"s{i}", days_old=60) for i in range(DEFAULT_MIN_SERIES_SIZE + 1)]
    batches = select_library_hygiene_candidates(shorts, TODAY)
    assert sum(len(b) for b in batches) == DEFAULT_MIN_SERIES_SIZE + 1


def test_non_public_privacy_is_excluded():
    shorts = _heavy_series(6, privacy="unlisted", days_old=60)
    assert select_library_hygiene_candidates(shorts, TODAY) == []


def test_unknown_privacy_never_treated_as_public():
    shorts = _heavy_series(6, privacy=None, days_old=60)
    assert select_library_hygiene_candidates(shorts, TODAY) == []


def test_ordering_is_heaviest_series_first():
    heavy = _heavy_series(10, series="Example County")
    light = _heavy_series(6, series="Sullivan")
    batches = select_library_hygiene_candidates(heavy + light, TODAY, batch_size=100)
    flat = [c for b in batches for c in b]
    assert flat[0].series == "Example County"
    assert flat[-1].series == "Sullivan"


def test_batching_caps_each_batch_at_the_batch_size():
    shorts = _heavy_series(12)
    batches = select_library_hygiene_candidates(shorts, TODAY, batch_size=5)
    assert [len(b) for b in batches] == [5, 5, 2]


def test_candidate_log_line_never_calls_any_privacy_api():
    # Purely a formatting check: this module has no import of any publishing
    # adapter, so there is nothing here that COULD change a platform's privacy.
    shorts = _heavy_series(6)
    batch = select_library_hygiene_candidates(shorts, TODAY)[0]
    line = batch[0].log_line()
    assert "LIBRARY_HYGIENE_CANDIDATE" in line
    assert "series=" in line and "views=" in line and "age_days=" in line


# --- library_hygiene_report / main(): the digest-facing store integration ---


def test_report_on_an_empty_store_says_no_records_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gomoveshift.tracking.volume.TrackingStore", lambda: TrackingStore(root=tmp_path)
    )
    assert library_hygiene_report() == "LIBRARY_HYGIENE: no tracking records yet"


def test_report_never_assumes_public_when_privacy_is_untracked(tmp_path, monkeypatch):
    store = TrackingStore(root=tmp_path)
    record = TrackingRecord(
        project="example-2026-06-01",
        location="Example County",
        shorts=[
            ShortRecord(
                index=i,
                posted_at="2026-06-01T00:00:00+00:00",
                posted_at_source="operator_confirmed",
            )
            for i in range(6)
        ],
    )
    store.save(record)
    monkeypatch.setattr("gomoveshift.tracking.volume.TrackingStore", lambda: store)
    got = library_hygiene_report(today=TODAY)
    assert "no candidates" in got
    assert "privacy" in got


def test_cli_main_rejects_an_unknown_subcommand(capsys):
    assert hygiene_main([]) == 2
    assert "usage:" in capsys.readouterr().err


def test_cli_main_prints_the_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "gomoveshift.tracking.volume.TrackingStore", lambda: TrackingStore(root=tmp_path)
    )
    assert hygiene_main(["library-hygiene"]) == 0
    assert "LIBRARY_HYGIENE: no tracking records yet" in capsys.readouterr().out
