"""Nothing anywhere persisted a SERIES of our own repeated fetches --
`ShortRecord.metrics` is one `PlatformMetrics` per platform and the newest
collection overwrites the last, so "how did this post move between Tuesday
and Friday" was unanswerable. `store.append_observation` appends instead of
overwriting, keyed to account + real post id, and replaying the identical
fetch must not double-count.
"""

from __future__ import annotations

from gomoveshift.tracking.observation import MetricObservation, ObservationStatus
from gomoveshift.tracking.schema import PlatformMetrics, ShortRecord, TrackingRecord
from gomoveshift.tracking.store import append_observation


def _obs(fetched_at="2026-09-01T00:00:00+00:00", views=10):
    return MetricObservation(
        platform="youtube",
        status=ObservationStatus.MEASURED,
        post_id="vid1",
        account_id="chan-1",
        post_format="short",
        cumulative=PlatformMetrics(views=views),
        fetched_at=fetched_at,
    )


def test_append_observation_grows_the_series():
    short = ShortRecord(index=0)
    record = TrackingRecord(project="proj", shorts=[short])

    append_observation(record, 0, _obs(fetched_at="2026-09-01T00:00:00+00:00", views=10))
    append_observation(record, 0, _obs(fetched_at="2026-09-02T00:00:00+00:00", views=25))

    series = short.observation_log["youtube"]
    assert len(series) == 2
    assert [o["cumulative"]["views"] for o in series] == [10, 25]


def test_replaying_the_identical_fetch_does_not_double_count():
    short = ShortRecord(index=0)
    record = TrackingRecord(project="proj", shorts=[short])
    obs = _obs(fetched_at="2026-09-01T00:00:00+00:00", views=10)

    first = append_observation(record, 0, obs)
    second = append_observation(record, 0, obs)

    assert first is True
    assert second is False
    assert len(short.observation_log["youtube"]) == 1


def test_series_round_trips_through_to_dict_from_dict():
    short = ShortRecord(index=0)
    record = TrackingRecord(project="proj", shorts=[short])
    append_observation(record, 0, _obs())

    reloaded = TrackingRecord.from_dict(record.to_dict())

    assert reloaded.shorts[0].observation_log["youtube"][0]["cumulative"]["views"] == 10


def test_a_record_saved_before_the_series_existed_reads_back_as_not_recorded():
    """Old JSON has no `observation_log` key at all. It must load as an
    empty series -- "not recorded" -- never as a fabricated recorded-but-
    empty platform key."""
    legacy = {
        "index": 0,
        "file": "short0.mp4",
        "metrics": {"youtube": PlatformMetrics(views=99).to_dict()},
    }
    short = ShortRecord.from_dict(legacy)

    assert short.observation_log == {}
    # the legacy overwrite-style metrics field is untouched by this change
    assert short.metrics["youtube"].views == 99
