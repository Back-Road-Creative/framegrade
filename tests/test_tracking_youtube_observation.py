"""YouTubeCollector had no observation method — `fetch_video_metrics`
and `collect_short_metrics` return a bare `PlatformMetrics()` on BOTH the
no-client path and the error path, so a caller cannot tell "the API said
nothing" from "we never asked" from "the call failed". This mirrors the
sibling `collect_observation`/`collect_observations_batch` methods the other
three collectors already carry.

No network: `_get_service` is always faked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gomoveshift.tracking.collector import YouTubeCollector
from gomoveshift.tracking.observation import MetricObservation, ObservationStatus

_UPLOAD_HANDLE = "v_inbox_file~v2.7000000000003"


def _fake_youtube(statistics: dict | None, items: list | None = None):
    fake = MagicMock()
    if items is not None:
        fake.videos().list().execute.return_value = {"items": items}
    else:
        fake.videos().list().execute.return_value = {"items": [{"statistics": statistics}]}
    return fake


def test_measured_when_the_api_answers_with_numbers():
    collector = YouTubeCollector()
    with (
        patch.object(collector, "is_available", return_value=True),
        patch.object(
            collector,
            "_get_service",
            return_value=_fake_youtube({"viewCount": "500", "likeCount": "3"}),
        ),
    ):
        obs = collector.collect_observation("vid1", account_id="chan-1")

    assert obs.status == ObservationStatus.MEASURED
    assert obs.platform == "youtube"
    assert obs.post_id == "vid1"
    assert obs.account_id == "chan-1"
    assert obs.cumulative.views == 500
    assert obs.cumulative.likes == 3


def test_not_configured_when_no_credentials_and_no_api_call_is_made():
    collector = YouTubeCollector()
    with (
        patch.object(collector, "is_available", return_value=False),
        patch.object(collector, "_get_service") as get_service,
    ):
        obs = collector.collect_observation("vid1")

    assert obs.status == ObservationStatus.NOT_CONFIGURED
    assert obs.cumulative.views is None
    get_service.assert_not_called()


def test_provider_error_when_the_api_call_raises():
    collector = YouTubeCollector()
    broken = MagicMock()
    broken.videos().list().execute.side_effect = RuntimeError("quota exceeded")
    with (
        patch.object(collector, "is_available", return_value=True),
        patch.object(collector, "_get_service", return_value=broken),
    ):
        obs = collector.collect_observation("vid1")

    assert obs.status == ObservationStatus.PROVIDER_ERROR
    assert "quota exceeded" in obs.detail
    assert obs.cumulative.views is None


def test_not_returned_when_the_platform_has_no_item_for_this_id():
    collector = YouTubeCollector()
    with (
        patch.object(collector, "is_available", return_value=True),
        patch.object(collector, "_get_service", return_value=_fake_youtube(None, items=[])),
    ):
        obs = collector.collect_observation("vid1")

    assert obs.status == ObservationStatus.NOT_RETURNED
    assert obs.cumulative.views is None


def test_upload_handle_is_unqueryable_without_ever_calling_the_api():
    collector = YouTubeCollector()
    with patch.object(collector, "_get_service") as get_service:
        obs = collector.collect_observation(_UPLOAD_HANDLE)

    assert obs.status == ObservationStatus.UNQUERYABLE_ID
    get_service.assert_not_called()


def test_observation_carries_event_and_fetch_times():
    collector = YouTubeCollector()
    with (
        patch.object(collector, "is_available", return_value=True),
        patch.object(collector, "_get_service", return_value=_fake_youtube({"viewCount": "10"})),
    ):
        obs = collector.collect_observation("vid1", published_at="2026-09-01T00:00:00+00:00")

    assert obs.event_at == "2026-09-01T00:00:00+00:00"
    assert obs.fetched_at is not None


def test_existing_fetch_video_metrics_still_returns_plain_metrics():
    """The sibling method must not disturb the existing bare-metrics callers."""
    collector = YouTubeCollector()
    with patch.object(collector, "_get_service", return_value=_fake_youtube({"viewCount": "77"})):
        metrics = collector.fetch_video_metrics("vid1")

    assert metrics.views == 77
    assert not isinstance(metrics, MetricObservation)
