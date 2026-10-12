"""``interval.latest_interval`` is the first production consumer of
``observation.interval_between``. It reads the last TWO entries of a short's
``observation_log[platform]`` series and turns them into an
``IntervalDelta`` -- never a zero-filled one, and never a raise, even when
the series is short, absent, or corrupted.
"""

from __future__ import annotations

from gomoveshift.tracking.interval import latest_interval
from gomoveshift.tracking.observation import MetricObservation, ObservationStatus
from gomoveshift.tracking.schema import PlatformMetrics, ShortRecord


def _measured(post_id="p1", fetched_at="2026-09-01T00:00:00+00:00", views=100, likes=10):
    return MetricObservation(
        platform="youtube",
        status=ObservationStatus.MEASURED,
        post_id=post_id,
        post_format="short",
        cumulative=PlatformMetrics(views=views, likes=likes),
        fetched_at=fetched_at,
    )


def _refused(
    post_id="p1", fetched_at="2026-09-02T00:00:00+00:00", status=ObservationStatus.NOT_RETURNED
):
    return MetricObservation(
        platform="youtube",
        status=status,
        post_id=post_id,
        post_format="short",
        fetched_at=fetched_at,
        detail="queried; the platform reported no values for this post",
    )


def _short_with_series(*observations) -> ShortRecord:
    short = ShortRecord(index=1)
    short.observation_log["youtube"] = [o.to_dict() for o in observations]
    return short


def test_no_series_at_all_returns_none() -> None:
    short = ShortRecord(index=1)
    assert latest_interval(short, "youtube") is None


def test_single_entry_returns_none() -> None:
    short = _short_with_series(_measured())
    assert latest_interval(short, "youtube") is None


def test_two_measurements_yield_a_measured_delta() -> None:
    earlier = _measured(fetched_at="2026-09-01T00:00:00+00:00", views=100, likes=10)
    later = _measured(fetched_at="2026-09-02T00:00:00+00:00", views=150, likes=12)
    short = _short_with_series(earlier, later)

    delta = latest_interval(short, "youtube")

    assert delta is not None
    assert delta.status == ObservationStatus.MEASURED
    assert (delta.views, delta.likes) == (50, 2)
    assert (delta.start_at, delta.end_at) == (earlier.fetched_at, later.fetched_at)


def test_counter_reset_is_reported_not_masked_as_negative() -> None:
    """A post deleted and reposted: the counter goes backwards. The true
    interval is unknowable, not negative."""
    earlier = _measured(fetched_at="2026-09-01T00:00:00+00:00", views=200)
    later = _measured(fetched_at="2026-09-02T00:00:00+00:00", views=50)
    short = _short_with_series(earlier, later)

    delta = latest_interval(short, "youtube")

    assert delta is not None
    assert delta.status == ObservationStatus.COUNTER_RESET
    assert delta.views is None


def test_a_refusal_as_the_later_entry_carries_its_own_status() -> None:
    earlier = _measured()
    later = _refused(status=ObservationStatus.RATE_LIMITED)
    short = _short_with_series(earlier, later)

    delta = latest_interval(short, "youtube")

    assert delta is not None
    assert delta.status == ObservationStatus.RATE_LIMITED
    assert delta.views is None


def test_unknown_platform_returns_none() -> None:
    short = _short_with_series(_measured(), _measured())
    assert latest_interval(short, "tiktok") is None


def test_corrupted_entry_never_raises() -> None:
    """A MEASURED entry with no ``post_id`` fails ``MetricObservation``
    construction; a caller that only asked for a trend must get ``None``,
    never a ``ValueError``."""
    short = ShortRecord(index=1)
    short.observation_log["youtube"] = [
        {
            "platform": "youtube",
            "status": ObservationStatus.MEASURED,
            "post_id": None,
            "cumulative": PlatformMetrics(views=1).to_dict(),
            "fetched_at": "2026-09-01T00:00:00+00:00",
        },
        _measured(fetched_at="2026-09-02T00:00:00+00:00").to_dict(),
    ]

    assert latest_interval(short, "youtube") is None
