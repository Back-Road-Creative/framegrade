"""Degraded shorts OST derivation policy.

Operator override: an operator ``--per-lf-location``
override labels its LF only — a GPS-BEARING short never reads it; the clip's
own GPS vote is burned, or the short hard-fails.
Degraded derivation: a GPS-bearing short whose own OST derivation degrades
(quorum miss / exception) WARNs (a per-short WARN plus one per-day summary
WARN) and then hard-fails — never silently inherits the parent-LF/drive-wide
text.
Region-gate skips for GPS-bearing shorts (the gate failing OPEN) are
counted into the same per-day summary WARN.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import gomoveshift.pipeline.day_pipeline as dp
from gomoveshift.pipeline.gps_enrichment import _make_short_ost_text_fn

# Moving trace (~22 m/s, stays in County Kerry): a real short is cut from moving
# footage; a parked constant-coordinate fixture would trip the stop-free gate.
_PTS = [(float(i), 52.14 + i * 0.0002, -10.27) for i in range(8)]
_PARENT_AB = "Dingle to Gortmore via R559"


def _run_shorts(monkeypatch, tmp_path, **overrides):
    """Drive ``dp._render_day_shorts`` with heavy I/O stubbed; returns the
    burned OST texts. Every short is GPS-BEARING (trim is identity)."""
    burned: list[str] = []

    def _render(**kw):
        burned.append(kw["location"])
        return True

    n = overrides.pop("n_shorts", 1)
    monkeypatch.setattr(dp, "_render_short_ffmpeg", _render)
    monkeypatch.setattr(dp, "trim_gps_to_clip_timeline", lambda pts, ranges: pts)
    monkeypatch.setattr(
        dp, "_resolve_short_ost_text", lambda c, p, r, d, drive_location=None: (c, 0)
    )
    monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda b: [])
    monkeypatch.setattr(dp, "assign_shorts_to_lfs_by_source_window", lambda **k: {1: range(n)})
    kwargs = dict(
        day_dir=tmp_path,
        date="2026-01-01",
        day_batch=SimpleNamespace(episodes=[]),
        short_ranges=[
            SimpleNamespace(rank=i + 1, start_sec=i * 40.0, end_sec=i * 40.0 + 38.0)
            for i in range(n)
        ],
        allow_short_count=True,
        gps_points_by_date={"2026-01-01": _PTS},
        lf_plans=[SimpleNamespace(index=1)],
        per_lf_ost_text={1: _PARENT_AB},
        per_lf_locations={1: "County Kerry"},
        location="Drive Wide",
        short_ost_text_fn=None,
        master_source_for_extract=Path("/dev/null"),
        # Fresh --ost-audit-confirmed bypasses the shorts pre-burn OST gate.
        state={"ost_audit_confirmed": True, "ost_audit_confirmed_this_run": True},
    )
    kwargs.update(overrides)
    dp._render_day_shorts(**kwargs)
    return burned


def test_gps_bearing_short_burns_its_clip_vote_not_the_operator_lf_label(monkeypatch, tmp_path):
    """The parent LF's operator override names the whole LF, never
    this clip — the clip's own GPS vote is consulted and burned instead."""
    calls: list[int] = []

    def _derive(pts, short_range):
        calls.append(short_range.rank)
        return "GPS Derived Label"

    burned = _run_shorts(
        monkeypatch,
        tmp_path,
        per_lf_overrides={1: "Operator Label, County Kerry"},
        short_ost_text_fn=_derive,
    )
    assert burned == ["GPS Derived Label"]
    assert calls == [1]  # the clip vote runs even under an operator override


def test_degraded_derivation_warns_then_hard_fails(monkeypatch, tmp_path, caplog):
    """Derivation returning None WARNs (naming rank, parent LF,
    reason, and the text that would have shipped) and then HARD-FAILS —
    a GPS-bearing short never silently inherits the parent A→B text."""
    with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.day_pipeline"):
        with pytest.raises(RuntimeError, match="rank 1"):
            _run_shorts(monkeypatch, tmp_path, short_ost_text_fn=lambda p, s: None)
    msgs = [r.getMessage() for r in caplog.records if "SHORT-OST-DEGRADED" in r.getMessage()]
    assert len(msgs) == 1, "degraded GPS-bearing short must WARN before the hard fail"
    assert "rank 1" in msgs[0]
    assert "parent LF 1" in msgs[0]
    assert _PARENT_AB in msgs[0]


def test_per_day_summary_counts_region_gate_skips(monkeypatch, tmp_path, caplog):
    """One per-day summary WARN folds region-gate skips (gate failing
    open on degraded region derivation) — independent of OST derivation,
    which here succeeds (a genuine OST degrade hard-fails, so it can
    no longer coexist with a multi-short summary WARN)."""
    with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.day_pipeline"):
        _run_shorts(
            monkeypatch,
            tmp_path,
            n_shorts=2,
            per_lf_ost_text={1: "County Kerry"},  # clean county, not an A→B arc
            short_ost_text_fn=lambda p, s: "Derived, Kerry",
            short_regions_fn=lambda p, s: [],
        )
    summaries = [
        r.getMessage() for r in caplog.records if "SHORTS-DEGRADE-SUMMARY" in r.getMessage()
    ]
    assert len(summaries) == 1
    assert "0 of 2" in summaries[0]  # OST degrades
    assert "region gate skipped for 2 of 2" in summaries[0]


def test_region_derivation_error_hard_blocks(monkeypatch, tmp_path):
    """An EXCEPTION from short_regions_fn is a genuine error, not a GPS-less
    clip. Even a CLEAN county label (which fails OPEN on an empty vote) must
    HARD-BLOCK when the region derivation itself errors — the clip's location is
    unverifiable, so it can't ship."""

    def _boom(pts, short_range):
        raise ValueError("region derivation exploded")

    with pytest.raises(RuntimeError, match="region gate ERROR"):
        _run_shorts(
            monkeypatch,
            tmp_path,
            per_lf_ost_text={1: "County Kerry"},  # clean county — would fail OPEN on empty vote
            short_ost_text_fn=lambda p, s: None,
            short_regions_fn=_boom,
        )


def test_inherited_arc_under_empty_region_vote_hard_blocks(monkeypatch, tmp_path):
    """Decouple the region-gate skip from the quorum miss that ALSO caused
    parent-text inheritance. A GPS-bearing short that inherits the parent LF's
    "A to B via Road" arc while its OWN clip vote is EMPTY must HARD-BLOCK
    upstream of the encode — nothing corroborates the arc against this clip —
    never silently fail open and ship it."""
    with pytest.raises(RuntimeError, match="inherited .*arc|arc.*empty"):
        _run_shorts(
            monkeypatch,
            tmp_path,
            short_ost_text_fn=lambda p, s: None,  # derivation degrades → inherits _PARENT_AB
            short_regions_fn=lambda p, s: [],  # empty clip vote (same quorum miss)
        )


def test_no_summary_when_derivation_healthy(monkeypatch, tmp_path, caplog):
    """A healthy derivation stays silent — no per-short WARN, no summary."""
    with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.day_pipeline"):
        burned = _run_shorts(monkeypatch, tmp_path, short_ost_text_fn=lambda p, s: "Derived, Kerry")
    assert burned == ["Derived, Kerry"]
    assert not [r for r in caplog.records if "SHORTS-DEGRADE" in r.getMessage()]


def test_ost_closure_quorum_miss_warns(caplog):
    """The closure's quorum-miss degrade (dominant_location → None) WARNs
    with the short's rank instead of returning None silently."""
    fn = _make_short_ost_text_fn(object(), poi_source=lambda _b: [])
    with patch("gomoveshift.pipeline.gps_enrichment.dominant_location", return_value=None):
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.gps_enrichment"):
            assert fn(_PTS, SimpleNamespace(rank=3)) is None
    assert any(
        "rank 3" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records
    ), "quorum-miss degrade must WARN with the short's rank"


def test_ost_closure_exception_warns(caplog):
    """The closure's exception degrade logs WARN (was DEBUG-only)."""
    fn = _make_short_ost_text_fn(object(), poi_source=lambda _b: [])
    with patch(
        "gomoveshift.pipeline.gps_enrichment.dominant_location",
        side_effect=RuntimeError("geocoder down"),
    ):
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.gps_enrichment"):
            assert fn(_PTS, SimpleNamespace(rank=4)) is None
    assert any("rank 4" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_ost_closure_coerces_tuple_points_before_vote():
    """The closure coerces positional ``(t, lat, lon)`` tuples → dicts
    BEFORE ``dominant_location`` votes, so the GPS quality filter reads
    lat/lon/time by key instead of no-opping on the bare-tuple shape — mirrors
    ``build_resolved_location``. Coercion preserves the trace's time/lat/lon."""
    fn = _make_short_ost_text_fn(object(), poi_source=lambda _b: [])
    captured: dict = {}

    def _capture(points, geocoder, *args, **kwargs):
        captured["points"] = list(points)
        return None

    with patch("gomoveshift.pipeline.gps_enrichment.dominant_location", side_effect=_capture):
        fn(_PTS, SimpleNamespace(rank=5))

    assert captured.get("points"), "dominant_location must receive the short's points"
    assert all(isinstance(p, dict) for p in captured["points"]), (
        "tuple GPS points must be coerced to dicts before the shorts-OST vote"
    )
    assert captured["points"][0] == {"time": 0.0, "lat": 52.14, "lon": -10.27}
