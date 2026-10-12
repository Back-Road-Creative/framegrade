"""A shorts->LF mapping failure must not silently swap an authoritative
per-LF override for the drive-wide label.

``_render_day_shorts`` builds ``_short_idx_to_lf_idx`` inside a try/except. When
the mapping raises, the pre-fix code swallowed it, leaving the map empty so every
short's ``_parent_lf_idx`` became ``None``: GPS-less shorts skipped the anchor
cascade and burned the drive-wide ``location``, and the per-short contract
assertion + region gate + degrade WARN were all guarded off. This mirrors
``phase_bc_day_first``'s hard-fail for the same >1-LF ambiguity: re-raise on a
multi-LF day (ambiguous attribution), and map to the sole LF's anchors on a
single-LF day (unambiguous, safe).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import gomoveshift.pipeline.day_pipeline as dp

_PTS = [SimpleNamespace(lat=52.1, lon=-9.5, time=0.0)]
_PARENT_AB = "Killarney to Kenmare via N71, County Kerry"


def _boom(**_kw):
    raise ValueError("source-window mapping blew up")


def _run(monkeypatch, tmp_path, *, lf_plans, per_lf_overrides=None, n_shorts=1):
    """Drive ``dp._render_day_shorts`` with heavy I/O stubbed AND the short->LF
    mapping forced to raise. Returns the burned OST texts (every short is
    GPS-bearing: trim is identity)."""
    burned: list[str] = []

    def _render(**kw):
        burned.append(kw["location"])
        return True

    monkeypatch.setattr(dp, "_render_short_ffmpeg", _render)
    monkeypatch.setattr(dp, "trim_gps_to_clip_timeline", lambda pts, ranges: pts)
    monkeypatch.setattr(dp, "_gate_short_ost_text", lambda c, p, r, d, drive_location=None: c)
    monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda b: [])
    # The mapping step raises — the behaviour under test is what happens next.
    monkeypatch.setattr(dp, "assign_shorts_to_lfs_by_source_window", _boom)

    dp._render_day_shorts(
        day_dir=tmp_path,
        date="2026-01-01",
        day_batch=SimpleNamespace(episodes=[]),
        short_ranges=[
            SimpleNamespace(rank=i + 1, start_sec=i * 40.0, end_sec=i * 40.0 + 38.0)
            for i in range(n_shorts)
        ],
        allow_short_count=True,
        gps_points_by_date={"2026-01-01": _PTS},
        lf_plans=lf_plans,
        per_lf_ost_text={p.index: _PARENT_AB for p in lf_plans},
        per_lf_locations={p.index: "County Kerry" for p in lf_plans},
        per_lf_overrides=per_lf_overrides,
        location="Drive Wide",
        short_ost_text_fn=None,
        master_source_for_extract=Path("/dev/null"),
        # Fresh --ost-audit-confirmed bypasses the shorts pre-burn OST gate.
        state={"ost_audit_confirmed": True, "ost_audit_confirmed_this_run": True},
    )
    return burned


def test_multi_lf_mapping_exception_reraises(monkeypatch, tmp_path):
    """>1 LF plan: an ambiguous short->LF mapping failure must RE-RAISE (mirror of
    phase_bc_day_first's equal-split hard-fail), never swallow -> drive-wide burn."""
    with pytest.raises(RuntimeError, match="ambiguous"):
        _run(
            monkeypatch,
            tmp_path,
            lf_plans=[SimpleNamespace(index=1), SimpleNamespace(index=2)],
        )


def test_single_lf_mapping_exception_maps_to_sole_lf(monkeypatch, tmp_path):
    """Exactly one LF: attribution is unambiguous, so a mapping failure maps the
    short to that LF and primes from that LF's own text — never the drive-wide
    ``location`` fallback, and never the operator LF label, which a
    GPS-bearing short does not read."""
    burned = _run(
        monkeypatch,
        tmp_path,
        lf_plans=[SimpleNamespace(index=1)],
        per_lf_overrides={1: "Operator Label, County Kerry"},
    )
    assert len(burned) == 1
    assert burned[0].startswith("Killarney to Kenmare")
    assert "Drive Wide" not in burned
    assert "Operator Label" not in burned[0]
