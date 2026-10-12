"""Console SEO tab: the anomalies half.

`tracking/anomaly.py` (per-post dip/spike detection) and
`tracking/suppression.py` (per-platform feed-suppression classification) have
existed for some time and are each reachable only from a CLI/cron
report -- an operator who does not tail that log has never seen either. This
module is the second surface for both, on the exact same never-raise,
never-invent-a-number contract `performance_view` already keeps for the tab.

Three things this file pins that a careless reader could get wrong:

* `AnomalyFinding`/`AnomalyRefusal` do not carry a `shadow` field -- it is
  stamped only into their `log_line()` string. The console adds
  `shadow: True` itself on the way into the template, and it must never be
  dropped between the kernel call and the render.
* `tracking/suppression.py`'s classifiers return a bare `{day: status}` map
  with no statistic attached. The console may translate that into "latest
  status, N consecutive days" -- and nothing richer; inventing an n or a
  p-value for a status that carries none would be exactly the fabrication
  `performance_view` was built to refuse.
* Neither kernel's policy constants (alpha, reference cap, share thresholds)
  are re-declared here -- a change to `anomaly.TOTAL_ALPHA` or
  `anomaly.REGISTERED_TESTS` must be visible in what the console shows
  without editing this module.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jinja2
import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import insights
from gomoveshift.console.app import create_app
from gomoveshift.tracking import anomaly
from gomoveshift.tracking.channel_daily import TRAFFIC_SOURCE_DIMENSION
from gomoveshift.tracking.schema import (
    DailyMetricPoint,
    PlatformMetrics,
    ShortRecord,
    TrackingRecord,
)

PLATFORM = "facebook"
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)

_TEMPLATES_DIR = (
    insights.Path(__file__).resolve().parent.parent / "gomoveshift" / "console" / "templates"
)

#: `seo.html`'s pre-existing Performance section reads
#: `performance.*` unconditionally -- unlike `anomalies`, `app.py` always
#: supplies it today, so the template rightly does not guard it. A direct
#: render here still needs *something* in that shape, or every test in this
#: file would be pinning an unrelated section's crash, not this section's
#: behavior; Jinja resolves dotted lookups on a plain dict via `__getitem__`
#: when there is no matching attribute, so this stand-in needs no class.
_MIN_PERFORMANCE = {
    "unknowns_badge": "",
    "unknowns": [],
    "hint": "",
    "rows": [],
    "totals": None,
    "data_as_of": None,
    "oldest_sample": None,
    "top_performers_cap": 10,
    "total_content_with_performance": None,
}


def _render_seo(**context) -> str:
    """Renders `seo.html` directly against Jinja, the way `_base.html`'s own
    ``if request else ''`` guards already allow -- the per-shape tests below
    need only the template; the live route is covered separately by
    `test_the_live_seo_route_shows_the_anomalies_section`."""
    context.setdefault("performance", _MIN_PERFORMANCE)
    # autoescape=True matches `Jinja2Templates`' own default (the real
    # production renderer, `app.py`'s), not just an S701 lint silencer.
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=True)
    template = env.get_template("seo.html")
    return template.render(request=None, **context)


def _short(index: int, day7: int, posted: datetime, platform: str = PLATFORM) -> ShortRecord:
    n = day7 // 7
    daily = [DailyMetricPoint(day=f"2026-01-{d:02d}", views=n) for d in range(1, 11)]
    return ShortRecord(
        index=index,
        posted_at=posted.isoformat(),
        posted_at_source="platform_confirmed",
        platform_ids={platform: f"vid{index}"},
        metrics={platform: PlatformMetrics(views=n * 10, daily=daily)},
    )


def _records(values_with_offsets, platform: str = PLATFORM):
    shorts = [
        _short(i, v, BASE + timedelta(days=offset), platform=platform)
        for i, (v, offset) in enumerate(values_with_offsets)
    ]
    return [TrackingRecord(project="proj", shorts=shorts)]


def _varied_references(n=100, base=1400, spread=70):
    # Multiples of 7 only: day7 round-trips through `// 7`, and a
    # non-multiple base/spread collapses the intended spread.
    return [(base + (i % 5) * spread, i) for i in range(n)]


def _dip_records():
    values = _varied_references()
    values.append((100, 100))
    return _records(values)


def _point_store(monkeypatch, records):
    monkeypatch.setattr(
        insights, "TrackingStore", lambda *a, **k: SimpleNamespace(load_all_records=lambda: records)
    )


def _no_suppression(monkeypatch):
    """Isolates a test from any real `channel_daily` store on disk."""
    monkeypatch.setattr(insights, "load_channel_daily_rows", lambda brand: [])


def _suppressed_rows(days):
    """Zero-total-views rows for each of `days` -- the one suppression shape
    that doesn't need 28 days of warm-up history to classify as SUPPRESSED
    (see `classify_youtube_shorts_feed`'s zero-total-day rule)."""
    return [
        {"day": day, "dimension": TRAFFIC_SOURCE_DIMENSION, "value": "SHORTS", "views": 0}
        for day in days
    ]


def test_a_significant_dip_renders_as_a_labeled_hypothesis_not_a_fact(monkeypatch):
    _point_store(monkeypatch, _dip_records())
    _no_suppression(monkeypatch)

    view = insights.anomaly_view()

    assert view["has_data"] is True
    assert len(view["findings"]) == 1
    finding = view["findings"][0]
    assert finding["kind"] == "dip"
    assert finding["deviation"] < 0
    assert finding["shadow"] is True
    assert finding["platform"] == PLATFORM
    assert finding["n"] == 100

    html = _render_seo(anomalies=view)
    assert "Anomalies &amp; suppression signals (hypotheses, not proof)" in html
    assert "Unvalidated" in html
    assert "dip (hypothesis)" in html


def test_the_live_seo_route_shows_the_anomalies_section(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(tmp_path / "place_corrections.yaml"))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    _point_store(monkeypatch, _dip_records())
    _no_suppression(monkeypatch)

    response = TestClient(create_app()).get("/seo")

    assert response.status_code == 200
    assert "Anomalies &amp; suppression signals (hypotheses, not proof)" in response.text
    assert "dip (hypothesis)" in response.text


def test_a_refusal_shows_its_reason_and_required_n(monkeypatch):
    # 78 references (< 79 required at allocated_alpha=0.025), then a target --
    # the kernel refuses before any statistic exists, see
    # `tracking/anomaly.py::detect_post_anomaly`.
    offsets = list(range(78))
    values = [(1000 + (i % 3) * 10, offset) for i, offset in enumerate(offsets)]
    values.append((100, 78))
    _point_store(monkeypatch, _records(values))
    _no_suppression(monkeypatch)

    view = insights.anomaly_view()

    assert view["has_data"] is True
    assert view["findings"] == []
    assert len(view["refusals"]) == 1
    refusal = view["refusals"][0]
    assert refusal["reason"] == "insufficient_resolution"
    assert refusal["required"] == 79
    assert refusal["shadow"] is True

    html = _render_seo(anomalies=view)
    assert "insufficient_resolution" in html
    assert "needs 79" in html


def test_no_tracking_data_yields_no_anomalies_section_not_a_crash(monkeypatch):
    _point_store(monkeypatch, [])
    _no_suppression(monkeypatch)

    view = insights.anomaly_view()

    assert view["has_data"] is False
    assert view["findings"] == []
    assert view["refusals"] == []
    html = _render_seo(anomalies=view)
    assert "Anomalies &amp; suppression signals" not in html

    # An unreadable store degrades the exact same way -- no exception escapes
    # the tab, and no section renders for it either.
    def _boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(insights, "TrackingStore", _boom)
    view2 = insights.anomaly_view()
    assert view2 == view
    html2 = _render_seo(anomalies=view2)
    assert "Anomalies &amp; suppression signals" not in html2


def test_shadow_flag_is_never_dropped_before_the_template(monkeypatch):
    offsets = list(range(78))
    refusal_values = [(1000 + (i % 3) * 10, offset) for i, offset in enumerate(offsets)]
    refusal_values.append((100, 78))
    records = _records(_varied_references())
    records[0].shorts.append(_short(100, 100, BASE + timedelta(days=100)))  # dip
    records += _records(refusal_values, platform="youtube")
    monkeypatch.setattr(
        insights, "TrackingStore", lambda *a, **k: SimpleNamespace(load_all_records=lambda: records)
    )
    monkeypatch.setattr(
        insights,
        "load_channel_daily_rows",
        lambda brand: _suppressed_rows(["2026-01-01", "2026-01-02"]),
    )

    view = insights.anomaly_view()

    assert view["has_data"] is True
    assert view["findings"], "expected at least one finding to check shadow on"
    assert all(f["shadow"] is True for f in view["findings"])
    assert view["refusals"], "expected at least one refusal to check shadow on"
    assert all(r["shadow"] is True for r in view["refusals"])
    assert view["suppression"], "expected a suppression row to check shadow on"
    assert all(s["shadow"] is True for s in view["suppression"])
    assert view["suppression"][0]["status"] == "SUPPRESSED"
    assert view["suppression"][0]["consecutive_days"] == 2


def test_policy_constants_are_read_not_reimplemented_by_the_console(monkeypatch):
    _point_store(monkeypatch, _dip_records())
    _no_suppression(monkeypatch)

    default_view = insights.anomaly_view()
    default_alpha = default_view["findings"][0]["allocated_alpha"]
    assert default_alpha == pytest.approx(anomaly.TOTAL_ALPHA / anomaly.REGISTERED_TESTS)

    # Change the kernel's own policy constants and confirm the console's
    # output moves with them -- a hardcoded alpha in `insights.py` would fail
    # this half of the test even though the assertions above still pass.
    monkeypatch.setattr(anomaly, "TOTAL_ALPHA", 0.10)
    monkeypatch.setattr(anomaly, "REGISTERED_TESTS", 1)
    patched_view = insights.anomaly_view()
    assert patched_view["findings"][0]["allocated_alpha"] == pytest.approx(0.10)
    assert patched_view["findings"][0]["allocated_alpha"] != default_alpha
