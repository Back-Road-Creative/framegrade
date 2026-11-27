"""Distance to each YouTube Partner tier: pure comparison plus a degrade-to-None fetch."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from gomoveshift.tracking import partner_progress as pp

BEFORE = date(2026, 11, 25)
AFTER = date(2027, 2, 1)


def _ad(today):
    return next(t for t in pp.tiers_for(today) if t.name == "ad_revenue")


def _fan(today):
    return next(t for t in pp.tiers_for(today) if t.name == "fan_funding")


def test_ad_revenue_thresholds_rise_on_new_terms_date():
    assert (_ad(BEFORE).watch_hours, _ad(BEFORE).shorts_views) == (4_000, 10_000_000)
    assert (_ad(AFTER).watch_hours, _ad(AFTER).shorts_views) == (8_000, 20_000_000)
    assert _ad(AFTER).subscribers == 1_000
    assert (_fan(AFTER).subscribers, _fan(AFTER).watch_hours) == (500, 3_000)


def test_watch_hours_or_shorts_views_satisfies_viewing():
    counts = pp.ChannelCounts(
        subscribers=1_200, long_form_watch_hours_365d=4_100.0, shorts_views_90d=10
    )
    assert pp.tier_progress(_ad(BEFORE), counts).met is True
    assert pp.tier_progress(_ad(AFTER), counts).met is False


def test_short_on_subscribers_is_not_met_and_gap_is_stated():
    counts = pp.ChannelCounts(
        subscribers=340, long_form_watch_hours_365d=5_000.0, shorts_views_90d=0
    )
    progress = pp.tier_progress(_ad(BEFORE), counts)
    assert progress.met is False
    assert progress.gaps["subscribers"] == 660
    assert progress.gaps["watch_hours"] == 0


def test_unmeasured_count_is_unknown_never_zero():
    counts = pp.ChannelCounts(
        subscribers=None, long_form_watch_hours_365d=5_000.0, shorts_views_90d=None
    )
    progress = pp.tier_progress(_ad(BEFORE), counts)
    assert progress.met is None
    line = progress.line(counts)
    assert "subs=unmeasured/1000" in line
    assert "shorts_views_90d=unmeasured/10000000" in line
    assert "approximate=yes" in line


def test_known_shortfall_wins_over_an_unmeasured_count():
    counts = pp.ChannelCounts(
        subscribers=100, long_form_watch_hours_365d=None, shorts_views_90d=None
    )
    assert pp.tier_progress(_ad(BEFORE), counts).met is False


def test_fan_funding_needs_three_uploads():
    counts = pp.ChannelCounts(
        subscribers=600,
        long_form_watch_hours_365d=3_500.0,
        shorts_views_90d=0,
        public_uploads_90d=2,
    )
    assert pp.tier_progress(_fan(BEFORE), counts).met is False
    counts = pp.ChannelCounts(
        subscribers=600,
        long_form_watch_hours_365d=3_500.0,
        shorts_views_90d=0,
        public_uploads_90d=3,
    )
    assert pp.tier_progress(_fan(BEFORE), counts).met is True


def test_deadline_line_only_before_new_terms():
    line = pp.deadline_line(BEFORE)
    assert "days_left=68" in line and "2027-02-01" in line
    assert pp.deadline_line(AFTER) is None


def test_counts_from_rows_splits_long_form_and_shorts():
    rows = [
        ["VIDEO_ON_DEMAND", 900, 6_000.0],
        ["SHORTS", 12_000, 300.0],
        ["LIVE_STREAM", 10, 60.0],
        ["STORY", 5, 1.0],
    ]
    assert pp.counts_from_rows(rows) == {"long_form_minutes": 6_060.0, "shorts_views": 12_000}
    assert pp.counts_from_rows([]) == {"long_form_minutes": None, "shorts_views": None}


def test_counts_from_rows_matches_content_types_in_any_case():
    # The dimensions page spells VIDEO_ON_DEMAND; a live answer reported as
    # videoOnDemand / shorts must count the same, not as 0 hours.
    rows = [
        ["videoOnDemand", 900, 6_000.0],
        ["shorts", 12_000, 300.0],
        ["liveStream", 10, 60.0],
        ["story", 5, 1.0],
    ]
    assert pp.counts_from_rows(rows) == {"long_form_minutes": 6_060.0, "shorts_views": 12_000}
    assert pp.counts_from_rows([["video_on_demand", 1, 60.0]])["long_form_minutes"] == 60.0


def test_counts_from_rows_with_no_known_type_is_unmeasured_and_logged(caplog):
    rows = [["PODCAST_EPISODE", 50, 600.0], ["SOMETHING_NEW", 7, 10.0]]
    with caplog.at_level("ERROR", logger=pp.__name__):
        assert pp.counts_from_rows(rows) == {"long_form_minutes": None, "shorts_views": None}
    assert "PARTNER_PROGRESS_UNMAPPED rows=2" in caplog.text
    assert "PODCAST_EPISODE" in caplog.text


def test_counts_from_rows_ignores_and_warns_on_one_unknown_type(caplog):
    rows = [["VIDEO_ON_DEMAND", 1, 120.0], ["NEW_KIND", 9, 999.0]]
    with caplog.at_level("WARNING", logger=pp.__name__):
        assert pp.counts_from_rows(rows) == {"long_form_minutes": 120.0, "shorts_views": 0}
    assert "NEW_KIND" in caplog.text


def test_counts_from_rows_known_types_without_long_form_are_a_real_zero():
    assert pp.counts_from_rows([["STORY", 3, 2.0], ["UNSPECIFIED", 1, 1.0]]) == {
        "long_form_minutes": 0.0,
        "shorts_views": 0,
    }


def _short(yt_id, posted):
    return SimpleNamespace(platform_ids={"youtube": yt_id} if yt_id else {}, posted_at=posted)


def test_public_uploads_counts_last_90_days_and_unknown_when_none_dated():
    records = [
        SimpleNamespace(
            shorts=[
                _short("a", "2026-11-01T10:00:00"),
                _short("b", "2026-06-01"),
                _short(None, "2026-11-02"),
            ],
            long_forms=[SimpleNamespace(video_id="L", published_date="2026-10-10")],
        )
    ]
    assert pp.count_public_uploads(records, BEFORE) == 2
    assert pp.count_public_uploads([SimpleNamespace(shorts=[], long_forms=[])], BEFORE) is None


class _Exec:
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error

    def execute(self):
        if self.error:
            raise self.error
        return self.value


def test_fetch_degrades_each_count_to_none_and_never_raises():
    noted = []
    collector = SimpleNamespace(
        _get_service=lambda: (_ for _ in ()).throw(RuntimeError("no token")),
        _analytics_client=lambda: None,
        _note_analytics_exception=noted.append,
    )
    counts = pp.fetch_channel_counts(collector, [], BEFORE)
    assert counts == pp.ChannelCounts()


def test_fetch_reads_subscribers_and_windows():
    calls = []

    class Reports:
        def query(self, **kw):
            calls.append(kw)
            return _Exec({"rows": [["VIDEO_ON_DEMAND", 100, 120.0], ["SHORTS", 4_000, 30.0]]})

    analytics = SimpleNamespace(reports=lambda: Reports())
    channels = SimpleNamespace(
        list=lambda **kw: _Exec(
            {"items": [{"statistics": {"subscriberCount": "412", "hiddenSubscriberCount": False}}]}
        )
    )
    collector = SimpleNamespace(
        _get_service=lambda: SimpleNamespace(channels=lambda: channels),
        _analytics_client=lambda: analytics,
        _note_analytics_exception=lambda exc: None,
    )
    counts = pp.fetch_channel_counts(collector, [], BEFORE)
    assert counts.subscribers == 412
    assert counts.long_form_watch_hours_365d == 2.0
    assert counts.shorts_views_90d == 4_000
    assert {c["startDate"] for c in calls} == {"2025-11-25", "2026-08-27"}
    assert all(c["dimensions"] == "creatorContentType" for c in calls)
