"""A short's burned place must come from ITS OWN clip GPS, not the
parent LF's drive-wide label — and the audit sidecar must record WHICH source
won (``place_source``) so a reviewer/CSV consumer can see when a short fell
back to the inherited label instead of its own clip-derived place.

Before this change:
- ``_render_day_shorts`` preferred a GPS-derived per-clip label
  (``short_ost_text_fn`` over ``_trimmed_points``, the clip's own on-screen GPS
  window) over the parent LF's ``per_lf_ost_text``/``per_lf_locations``/
  drive-wide ``location``, and tracked that provenance internally as
  ``_ost_from_clip_vote``, but discarded it: the audit entry it appended
  carried no source field, so nothing downstream (``format_shorts_ost_audit``)
  could tell a clip-derived place from an inherited one.
- Per-clip GPS data already existed via ``trim_gps_to_clip_timeline`` (the
  clip's on-screen GPS fixes) fed into ``_make_short_ost_text_fn`` /
  ``dominant_location`` (``gomoveshift/pipeline/gps_enrichment.py``), which
  votes county/state through the SAME cached geocoder as the LF path
  (``_build_shared_geocoder``) — no new network client.

This module pins the fix: the discarded ``_ost_from_clip_vote`` boolean is
now persisted as ``place_source`` (``"clip_gps"`` vs ``"lf_label"``) on both
the raw audit entry and the sidecar JSON.

Further, ``place_source: "lf_label"`` was still a SILENT fallback — a
GPS-bearing short whose own derivation degraded (geocoder quorum miss /
exception, never "no GPS") shipped the inherited parent-LF/drive-wide text
with only a WARN log (a caption reading "To X"). This
module now also pins the hard gate: that specific fallback (GPS present,
derivation actually attempted and degraded — an operator override exempts
nothing) raises UPSTREAM of the 9:16 encode instead of shipping. A genuinely GPS-less short
(nothing of its own to vote) is unaffected — that fallback is legitimate.
On a day WITH GPS, a clip with no fix of its own is not GPS-less: it hard-fails
too, since no LF label, destination or county may stand in.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import gomoveshift.pipeline.day_pipeline as dp

_PARENT_AB = "Dingle to Gortmore via R559"
# Moving trace (~22 m/s, stays in County Kerry): a real short is cut from
# moving footage; a parked constant-coordinate fixture trips the
# stop-free gate (mirrors tests/test_shorts_ost_degrade_policy.py).
_PTS = [(float(i), 52.14 + i * 0.0002, -10.27) for i in range(80)]


def _run_shorts(monkeypatch, tmp_path, **overrides):
    """Drive ``dp._render_day_shorts`` with heavy I/O stubbed; returns the
    raw audit entries (not just burned text) so ``place_source`` is visible."""
    monkeypatch.setattr(dp, "_render_short_ffmpeg", lambda **kw: True)
    monkeypatch.setattr(dp, "trim_gps_to_clip_timeline", overrides.pop("_trim", lambda p, r: p))
    monkeypatch.setattr(
        dp, "_resolve_short_ost_text", lambda c, p, r, d, drive_location=None: (c, 0)
    )
    monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda b: [])
    n = overrides.pop("n_shorts", 1)
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
        state={"ost_audit_confirmed": True, "ost_audit_confirmed_this_run": True},
    )
    kwargs.update(overrides)
    _, _, entries = dp._render_day_shorts(**kwargs)
    return entries


def test_clip_derived_short_records_place_source_clip_gps(monkeypatch, tmp_path):
    """A GPS-bearing short whose own clip vote resolves a label is
    ``place_source: "clip_gps"`` — and the burned text is the clip's own
    town, never the parent LF's ``"A to B"`` label."""
    entries = _run_shorts(monkeypatch, tmp_path, short_ost_text_fn=lambda p, s: "Annascaul")
    assert entries[0]["ost_text"] == "Annascaul"
    assert entries[0]["ost_text"] != _PARENT_AB
    assert entries[0]["place_source"] == "clip_gps"


def test_inherited_label_with_gps_present_hard_fails(monkeypatch, tmp_path):
    """A GPS-bearing short whose own derivation degrades (quorum miss)
    is NOT allowed to silently ship the inherited parent LF label — the
    run must hard-fail upstream of the 9:16 encode, naming the short."""
    with pytest.raises(RuntimeError, match="rank 1"):
        _run_shorts(monkeypatch, tmp_path, short_ost_text_fn=lambda p, s: None)


def test_operator_override_with_gps_present_is_ignored(monkeypatch, tmp_path):
    """An operator ``--per-lf-location`` override names the whole LF, never one
    clip — a GPS-bearing short under an
    override burns its own clip vote and records ``"clip_gps"``."""
    entries = _run_shorts(
        monkeypatch,
        tmp_path,
        per_lf_overrides={1: "Operator Label, County Kerry"},
        short_ost_text_fn=lambda p, s: "Annascaul",
    )
    assert entries[0]["ost_text"] == "Annascaul"
    assert entries[0]["place_source"] == "clip_gps"


def test_gps_less_short_falls_back_with_place_source_set(monkeypatch, tmp_path):
    """A short on a GPS-LESS day (no usable fix anywhere) has nothing of its
    own to vote — it must fall back to the parent LF anchor AND still carry
    an explicit ``place_source`` (never an absent/None key)."""
    entries = _run_shorts(
        monkeypatch,
        tmp_path,
        gps_points_by_date={"2026-01-01": []},
        _trim=lambda p, r: [],
        short_ost_text_fn=lambda p, s: "Should Never Be Called",
    )
    assert entries[0]["place_source"] == "lf_label"
    assert "place_source" in entries[0]


def test_clip_without_a_fix_on_a_gps_day_hard_fails(monkeypatch, tmp_path):
    """On a day WITH GPS, a clip whose own window holds no fix cannot verify
    its place and nothing may stand in for it: the render hard-fails upstream
    of the 9:16 encode, naming the short."""
    with pytest.raises(RuntimeError, match=r"rank 1.*no GPS fix of its own"):
        _run_shorts(
            monkeypatch,
            tmp_path,
            _trim=lambda p, r: [],
            short_ost_text_fn=lambda p, s: "Should Never Be Called",
        )


def test_two_shorts_same_lf_different_towns_get_distinct_clip_gps_places(monkeypatch, tmp_path):
    """Two shorts cut from the SAME LF, whose clip midpoints geocode
    (mocked) to different towns, get DIFFERENT places — neither is the
    LF-level ``"To X"`` arc label."""

    def _derive(pts, short_range):
        return "Annascaul" if short_range.rank == 1 else "Camp"

    entries = _run_shorts(monkeypatch, tmp_path, n_shorts=2, short_ost_text_fn=_derive)
    assert entries[0]["ost_text"] == "Annascaul"
    assert entries[1]["ost_text"] == "Camp"
    assert entries[0]["ost_text"] != entries[1]["ost_text"]
    for e in entries:
        assert e["place_source"] == "clip_gps"
        assert e["ost_text"] != _PARENT_AB


def test_format_shorts_ost_audit_persists_place_source():
    """The sidecar JSON serializer carries ``place_source`` through, defaulting
    an absent/legacy entry to ``"lf_label"`` (the pre-fix, inherited-only
    behaviour) rather than silently dropping the field."""
    from gomoveshift.pipeline.ost_audit_sidecar import format_shorts_ost_audit

    payload = format_shorts_ost_audit(
        "2026-06-01",
        [
            {
                "rank": 1,
                "parent_lf_index": 1,
                "ost_text": "Annascaul",
                "start_sec": 0.0,
                "end_sec": 30.0,
                "regions": ["County Kerry"],
                "place_source": "clip_gps",
            },
            {
                "rank": 2,
                "parent_lf_index": 1,
                "ost_text": "County Kerry",
                "start_sec": 30.0,
                "end_sec": 60.0,
                "regions": [],
                # legacy entry: no place_source supplied
            },
        ],
    )
    assert payload["shorts"][0]["place_source"] == "clip_gps"
    assert payload["shorts"][1]["place_source"] == "lf_label"
