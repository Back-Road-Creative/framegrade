"""An observation says WHY it has no numbers, keeps cumulative counters
apart from interval deltas, distinguishes event/fetch/provider-delay times,
binds itself to an account + public post id, and refuses to blend view counts
whose definitions differ. No network: every HTTP layer here is a fake."""

from __future__ import annotations

import pytest

from gomoveshift.tracking.collector import (
    FacebookCollector,
    InstagramCollector,
    TikTokCollector,
)
from gomoveshift.tracking.observation import (
    MetricObservation,
    ObservationStatus,
    compare_views,
    interval_between,
)
from gomoveshift.tracking.schema import PlatformMetrics

_UPLOAD_HANDLE = "v_inbox_file~v2.7000000000003"


def _measured(platform="tiktok", post_id="111", views=10, **kw):
    return MetricObservation(
        platform=platform,
        status=ObservationStatus.MEASURED,
        post_id=post_id,
        account_id="acct-1",
        post_format=kw.pop("post_format", "short"),
        cumulative=PlatformMetrics(views=views, likes=1, comments=0, shares=0),
        **kw,
    )


# ── 1. Reason, not just absence ──────────────────────────────────────────


def test_the_reasons_an_observation_has_no_numbers_are_distinct():
    reasons = {
        ObservationStatus.NOT_CONFIGURED,
        ObservationStatus.UNQUERYABLE_ID,
        ObservationStatus.NOT_RETURNED,
        ObservationStatus.PERMISSION_DENIED,
        ObservationStatus.RATE_LIMITED,
        ObservationStatus.PROVIDER_ERROR,
        ObservationStatus.MEASURED,
    }
    assert len(reasons) == 7
    assert ObservationStatus.MEASURED not in ObservationStatus.UNANSWERED


def test_an_unanswered_observation_may_not_carry_numbers():
    with pytest.raises(ValueError, match="numbers"):
        MetricObservation(
            platform="tiktok",
            status=ObservationStatus.RATE_LIMITED,
            post_id="111",
            cumulative=PlatformMetrics(views=0),
        )


def test_a_measurement_must_name_the_public_post_it_measured():
    with pytest.raises(ValueError, match="post_id"):
        MetricObservation(platform="tiktok", status=ObservationStatus.MEASURED)


def test_an_upload_handle_is_never_a_public_post_id():
    with pytest.raises(ValueError, match="upload handle"):
        MetricObservation(
            platform="tiktok", status=ObservationStatus.MEASURED, post_id=_UPLOAD_HANDLE
        )


def test_observation_round_trips_through_json_shape():
    obs = _measured(event_at="2026-09-01T10:00:00+00:00", fetched_at="2026-09-03T10:00:00+00:00")
    back = MetricObservation.from_dict(obs.to_dict())
    assert back == obs
    assert back.cumulative.views == 10


# ── 2. Cumulative counters vs interval deltas ────────────────────────────


def test_interval_between_two_cumulative_reads_is_a_delta_not_a_counter():
    earlier = _measured(views=100, fetched_at="2026-09-01T00:00:00+00:00")
    later = _measured(views=130, fetched_at="2026-09-02T00:00:00+00:00")
    delta = interval_between(earlier, later)
    assert delta.status == ObservationStatus.MEASURED
    assert delta.views == 30
    assert delta.start_at == "2026-09-01T00:00:00+00:00"
    assert delta.end_at == "2026-09-02T00:00:00+00:00"
    # The cumulative read is untouched — a lifetime total and a delta are
    # never the same field.
    assert later.cumulative.views == 130


def test_interval_is_unknown_when_either_end_never_reported():
    earlier = _measured(views=100)
    later = MetricObservation(
        platform="tiktok",
        status=ObservationStatus.PERMISSION_DENIED,
        post_id="111",
        detail="scope_not_authorized",
    )
    delta = interval_between(earlier, later)
    assert delta.status == ObservationStatus.PERMISSION_DENIED
    assert delta.views is None


def test_a_counter_that_went_backwards_is_not_a_negative_delta():
    earlier = _measured(views=500)
    later = _measured(views=12)
    delta = interval_between(earlier, later)
    assert delta.status == ObservationStatus.COUNTER_RESET
    assert delta.views is None


def test_an_interval_may_not_span_two_different_posts():
    with pytest.raises(ValueError, match="same post"):
        interval_between(_measured(post_id="111"), _measured(post_id="222"))


# ── 3. Event date, fetch time, provider reporting delay ──────────────────


def test_event_fetch_and_provider_delay_are_three_separate_facts():
    obs = _measured(
        event_at="2026-09-01T00:00:00+00:00",
        fetched_at="2026-09-03T00:00:00+00:00",
        data_through="2026-09-02T00:00:00+00:00",
    )
    assert obs.age_at_fetch_seconds == 2 * 86400
    assert obs.reporting_delay_seconds == 86400
    # A fresh zero and a stale zero are not the same observation.
    assert _measured(fetched_at="2026-09-03T00:00:00+00:00").reporting_delay_seconds is None


# ── 5. Refuse to average incomparable things ─────────────────────────────


def test_view_counts_from_different_platforms_are_never_blended():
    result = compare_views([_measured("youtube", "yt1", 100), _measured("tiktok", "111", 900)])
    assert result.blended_views is None
    assert "definition" in result.declined_reason
    assert set(result.per_platform) == {"youtube", "tiktok"}
    assert result.definitions["youtube"] != result.definitions["tiktok"]


def test_like_platform_unlike_format_is_not_blended():
    result = compare_views(
        [
            _measured(post_id="111", post_format="short"),
            _measured(post_id="222", post_format="long_form"),
        ]
    )
    assert result.blended_views is None
    assert "format" in result.declined_reason


def test_like_platform_and_format_at_unlike_ages_is_not_blended():
    young = _measured(
        post_id="111", event_at="2026-09-02T00:00:00+00:00", fetched_at="2026-09-03T00:00:00+00:00"
    )
    old = _measured(
        post_id="222", event_at="2026-06-01T00:00:00+00:00", fetched_at="2026-09-03T00:00:00+00:00"
    )
    result = compare_views([young, old])
    assert result.blended_views is None
    assert "age" in result.declined_reason


def test_like_formats_at_like_ages_on_one_platform_do_compare():
    a = _measured(
        post_id="111",
        views=100,
        event_at="2026-09-02T00:00:00+00:00",
        fetched_at="2026-09-03T00:00:00+00:00",
    )
    b = _measured(
        post_id="222",
        views=200,
        event_at="2026-09-02T06:00:00+00:00",
        fetched_at="2026-09-03T00:00:00+00:00",
    )
    result = compare_views([a, b])
    assert result.declined_reason == ""
    assert result.blended_views == 150.0


# ── Collectors report the reason instead of an empty dict ────────────────


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, params=None, json=None, headers=None, timeout=None):
        self.calls.append((url, params, json, headers))
        return _Resp(self._responses.pop(0))

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return _Resp(self._responses.pop(0))


class _Resp:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = body.get("_status", status_code) if isinstance(body, dict) else 200

    def json(self):
        return self._body


def _ok(videos):
    return {
        "data": {"videos": videos, "cursor": 0, "has_more": False},
        "error": {"code": "ok"},
    }


def test_tiktok_batch_says_why_each_id_has_no_metrics():
    session = _FakeSession([_ok([{"id": "111", "view_count": 10}])])
    collector = TikTokCollector(access_token="tok", session=session)  # noqa: S106
    obs = collector.collect_observations_batch(
        ["111", "222", _UPLOAD_HANDLE], account_id="tiktok-acct"
    )
    assert set(obs) == {"111", "222", _UPLOAD_HANDLE}
    assert obs["111"].status == ObservationStatus.MEASURED
    assert obs["111"].cumulative.views == 10
    assert obs["111"].account_id == "tiktok-acct"
    assert obs["111"].post_id == "111"
    assert obs["111"].fetched_at
    # Queried and not returned — deleted, private or not on this account.
    assert obs["222"].status == ObservationStatus.NOT_RETURNED
    # Never queried at all: an upload handle is not a public post id.
    assert obs[_UPLOAD_HANDLE].status == ObservationStatus.UNQUERYABLE_ID


def test_tiktok_batch_without_credentials_says_not_configured(monkeypatch):
    monkeypatch.setattr(
        "gomoveshift.tracking.collector.get_valid_tiktok_credentials",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("no creds in test")),
    )
    obs = TikTokCollector(access_token=None).collect_observations_batch(["111"])
    assert obs["111"].status == ObservationStatus.NOT_CONFIGURED


@pytest.mark.parametrize(
    "code,expected",
    [
        ("rate_limit_exceeded", ObservationStatus.RATE_LIMITED),
        ("scope_not_authorized", ObservationStatus.PERMISSION_DENIED),
        ("internal_error", ObservationStatus.PROVIDER_ERROR),
    ],
)
def test_tiktok_error_classes_are_not_all_the_same_silence(code, expected):
    body = {"data": {}, "error": {"code": code, "message": code}}
    collector = TikTokCollector(access_token="tok", session=_FakeSession([body]))  # noqa: S106
    obs = collector.collect_observations_batch(["111"])
    assert obs["111"].status == expected
    assert obs["111"].cumulative.views is None


def test_instagram_scope_denial_is_a_permission_status_not_an_empty_reel():
    creds = {"page_access_token": "t", "ig_user_id": "ig1", "graph_version": "v20.0"}  # noqa: S106
    body = {"error": {"message": "requires instagram_manage_insights"}, "_status": 403}
    collector = InstagramCollector(credentials=creds, session=_FakeSession([body]))
    obs = collector.collect_observation("ig-media-1", published_at="2026-09-01T00:00:00+00:00")
    assert obs.status == ObservationStatus.PERMISSION_DENIED
    assert obs.post_id == "ig-media-1"
    assert obs.account_id == "ig1"
    assert obs.event_at == "2026-09-01T00:00:00+00:00"
    assert obs.cumulative.views is None


def test_facebook_missing_insights_scope_is_a_permission_status():
    creds = {"page_access_token": "t", "page_id": "pg1", "scopes": []}  # noqa: S106
    obs = FacebookCollector(credentials=creds, session=_FakeSession([])).collect_observation("v1")
    assert obs.status == ObservationStatus.PERMISSION_DENIED
    assert obs.detail
