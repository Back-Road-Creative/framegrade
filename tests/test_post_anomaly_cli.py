"""``post-anomaly`` CLI: read-only, always exits 0, shadow=true on every finding line."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from seo_manager.analytics.anomaly_stats import Refusal

from gomoveshift.cli.handlers.post_anomaly import cmd_post_anomaly
from gomoveshift.tracking.schema import (
    DailyMetricPoint,
    PlatformMetrics,
    ShortRecord,
    TrackingRecord,
)

RUN_MARKER_RE = re.compile(
    r"^POST_ANOMALY_RUN ts=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) window=(\d+) "
    r"outcome=(ok|refused|no-candidate|error)$"
)

PLATFORM = "facebook"
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _short(index: int, day7: int, posted) -> ShortRecord:
    n = day7 // 7
    daily = [DailyMetricPoint(day=f"2026-01-{d:02d}", views=n) for d in range(1, 11)]
    return ShortRecord(
        index=index,
        posted_at=posted.isoformat(),
        # A posting time must say which moment it is: ShortRecord refuses a
        # posted_at with no provenance on construction, so the fixture states
        # one rather than leaving the comparison undefined.
        posted_at_source="platform_confirmed",
        platform_ids={PLATFORM: f"vid{index}"},
        metrics={PLATFORM: PlatformMetrics(views=n * 10, daily=daily)},
    )


def _dip_records():
    values = [(1000 + (i % 5) * 50, i) for i in range(100)]
    values.append((100, 100))
    shorts = [_short(i, v, BASE + timedelta(days=offset)) for i, (v, offset) in enumerate(values)]
    return [TrackingRecord(project="proj", shorts=shorts)]


def _run(monkeypatch, records, capsys):
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.post_anomaly.TrackingStore",
        lambda *a: SimpleNamespace(load_all_records=lambda: records),
    )
    rc = cmd_post_anomaly(SimpleNamespace(), None)
    return rc, capsys.readouterr().out


def _run_markers(out):
    return [
        RUN_MARKER_RE.match(ln) for ln in out.splitlines() if ln.startswith("POST_ANOMALY_RUN ")
    ]


def test_empty_store_prints_insufficient_data_and_exits_zero(monkeypatch, capsys):
    rc, out = _run(monkeypatch, [], capsys)
    assert rc == 0
    assert "insufficient data" in out.lower()
    assert "POST_ANOMALY " not in out

    markers = _run_markers(out)
    assert len(markers) == 1
    assert markers[0].group(3) == "no-candidate"


def test_a_dip_fixture_prints_one_post_anomaly_line(monkeypatch, capsys):
    rc, out = _run(monkeypatch, _dip_records(), capsys)
    lines = [ln for ln in out.splitlines() if ln.startswith("POST_ANOMALY ")]
    assert rc == 0
    assert len(lines) == 1
    assert "kind=dip" in lines[0] and "shadow=true" in lines[0]

    markers = _run_markers(out)
    assert len(markers) == 1
    assert markers[0].group(3) == "ok"


def test_no_candidate_found_still_emits_ok_outcome(monkeypatch, capsys):
    """A run that completes with zero findings is still a fresh, successful run."""
    values = [(1000 + (i % 5) * 50, i) for i in range(100)]
    values.append((1000, 100))
    shorts = [_short(i, v, BASE + timedelta(days=offset)) for i, (v, offset) in enumerate(values)]
    rc, out = _run(monkeypatch, [TrackingRecord(project="proj", shorts=shorts)], capsys)
    assert rc == 0
    assert "no candidate" in out.lower()

    markers = _run_markers(out)
    assert len(markers) == 1
    assert markers[0].group(3) == "ok"


def test_infeasible_epoch_emits_refused_outcome(monkeypatch, capsys):
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.post_anomaly.TrackingStore",
        lambda *a: SimpleNamespace(load_all_records=lambda: _dip_records()),
    )
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.post_anomaly.check_feasibility",
        lambda: Refusal(reason="reference_cap_too_small", detail="cap below required"),
    )
    rc = cmd_post_anomaly(SimpleNamespace(), None)
    out = _capsys_out(capsys)
    assert rc == 0

    markers = _run_markers(out)
    assert len(markers) == 1
    assert markers[0].group(3) == "refused"


def test_error_path_still_emits_a_marker_before_raising(monkeypatch, capsys):
    def _boom():
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.post_anomaly.TrackingStore",
        lambda *a: SimpleNamespace(load_all_records=_boom),
    )
    with pytest.raises(RuntimeError, match="kaboom"):
        cmd_post_anomaly(SimpleNamespace(), None)
    out = _capsys_out(capsys)

    markers = _run_markers(out)
    assert len(markers) == 1
    assert markers[0].group(3) == "error"


def _capsys_out(capsys):
    return capsys.readouterr().out
