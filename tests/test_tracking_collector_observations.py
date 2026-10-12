"""Collection is wired to ``ShortRecord.observation_log`` and
``store.append_observation``, refusals included, instead of the thin
``collect_short_metrics`` projection overwriting ``short.metrics`` in place.
"""

from __future__ import annotations

from gomoveshift.tracking.collector import MetricsCollector
from gomoveshift.tracking.observation import MetricObservation, ObservationStatus
from gomoveshift.tracking.schema import PlatformMetrics, ShortRecord, TrackingRecord


class _MeasuringCollector:
    def is_available(self):
        return True

    def collect_observation(self, video_id, published_at=None):
        return MetricObservation(
            platform="youtube",
            status=ObservationStatus.MEASURED,
            post_id=video_id,
            post_format="short",
            event_at=published_at,
            fetched_at="2026-09-10T00:00:00+00:00",
            cumulative=PlatformMetrics(views=100, likes=5),
        )


class _RefusingCollector:
    def is_available(self):
        return True

    def collect_observation(self, video_id, published_at=None):
        return MetricObservation(
            platform="instagram",
            status=ObservationStatus.PERMISSION_DENIED,
            post_id=video_id,
            post_format="reel",
            event_at=published_at,
            fetched_at="2026-09-10T00:00:00+00:00",
            detail="token is missing the read_insights scope",
        )


def _record_with(short: ShortRecord) -> TrackingRecord:
    return TrackingRecord(project="proj", shorts=[short])


def test_collect_all_appends_a_measured_observation_to_the_log() -> None:
    short = ShortRecord(index=1, platform_ids={"youtube": "yt_1"})
    record = _record_with(short)
    mc = MetricsCollector()
    mc._collectors["youtube"] = _MeasuringCollector()

    mc.collect_all(record)

    series = short.observation_log["youtube"]
    assert len(series) == 1
    assert series[0]["status"] == ObservationStatus.MEASURED
    assert series[0]["post_id"] == "yt_1"
    assert short.metrics["youtube"].views == 100


def test_collect_all_appends_a_refusal_and_never_overwrites_metrics() -> None:
    short = ShortRecord(
        index=2,
        platform_ids={"instagram": "ig_1"},
        metrics={"instagram": PlatformMetrics(views=42)},
    )
    record = _record_with(short)
    mc = MetricsCollector()
    mc._collectors["instagram"] = _RefusingCollector()

    mc.collect_all(record)

    series = short.observation_log["instagram"]
    assert len(series) == 1
    assert series[0]["status"] == ObservationStatus.PERMISSION_DENIED
    # A refusal carries no numbers -- the prior measured metrics survive
    # untouched rather than being clobbered by an empty result.
    assert short.metrics["instagram"].views == 42


def test_collect_all_running_twice_appends_a_second_series_entry() -> None:
    """Two collection runs must not overwrite the previous read -- that is
    the whole point of the series over the old overwrite-in-place metrics."""
    short = ShortRecord(index=3, platform_ids={"youtube": "yt_1"})
    record = _record_with(short)
    mc = MetricsCollector()

    calls = {"n": 0}

    class _Sequenced:
        def is_available(self):
            return True

        def collect_observation(self, video_id, published_at=None):
            calls["n"] += 1
            return MetricObservation(
                platform="youtube",
                status=ObservationStatus.MEASURED,
                post_id=video_id,
                post_format="short",
                fetched_at=f"2026-09-1{calls['n']}T00:00:00+00:00",
                cumulative=PlatformMetrics(views=100 * calls["n"]),
            )

    mc._collectors["youtube"] = _Sequenced()
    mc.collect_all(record)
    mc.collect_all(record)

    assert len(short.observation_log["youtube"]) == 2
    assert short.metrics["youtube"].views == 200


def test_collect_for_short_still_returns_bare_metrics_for_measured_platforms() -> None:
    short = ShortRecord(index=4, platform_ids={"youtube": "yt_1"})
    mc = MetricsCollector()
    mc._collectors["youtube"] = _MeasuringCollector()

    results = mc.collect_for_short(short)

    assert results["youtube"].views == 100


def test_collect_for_short_omits_a_refused_platform() -> None:
    short = ShortRecord(index=5, platform_ids={"instagram": "ig_1"})
    mc = MetricsCollector()
    mc._collectors["instagram"] = _RefusingCollector()

    results = mc.collect_for_short(short)

    assert "instagram" not in results


def test_a_refusal_only_short_is_not_counted_as_collected() -> None:
    """``collect_all``'s return feeds the CLI's "Collected metrics for N
    short(s)" line (``cli/handlers/metrics.py``). A short every platform
    REFUSED about measured nothing, so counting it restates the claim
    ``test_tracking_collected_nothing_is_not_collected`` refuses for the log
    line -- one level up, on the count that line prints.

    The refusal is still recorded: it belongs in ``observation_log``, which
    is what makes it reportable without inflating a measurement count.
    """
    short = ShortRecord(index=1, platform_ids={"instagram": "ig_1"})
    record = _record_with(short)
    mc = MetricsCollector()
    mc._collectors["instagram"] = _RefusingCollector()

    assert mc.collect_all(record) == 0
    # ...and the refusal was still written down.
    assert short.observation_log["instagram"][0]["status"] == (ObservationStatus.PERMISSION_DENIED)
    assert "instagram" not in short.metrics


def test_one_measurement_beside_a_refusal_still_counts() -> None:
    """The count must not swing the other way: a short where one platform
    measured and another refused DID have something measured."""
    short = ShortRecord(index=1, platform_ids={"youtube": "yt_1", "instagram": "ig_1"})
    record = _record_with(short)
    mc = MetricsCollector()
    mc._collectors["youtube"] = _MeasuringCollector()
    mc._collectors["instagram"] = _RefusingCollector()

    assert mc.collect_all(record) == 1
    assert short.metrics["youtube"].views == 100
    assert short.observation_log["instagram"][0]["status"] == (ObservationStatus.PERMISSION_DENIED)
