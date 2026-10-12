"""Per-short clip-GPS regions persist into the shorts OST sidecar and feed the
CSV region invariant.

Previously the per-short CSV region invariant checked a label against itself:
``SocialPost.sampled_regions`` was the SAME resolved label the title/caption/
tags derive from, so the gate passed by construction. These pins prove each
short's own clip-GPS vote regions (an independent source) are recorded in the
day sidecar and flow into ``sampled_regions``, and that a GPS-bearing short
whose sidecar label went missing HARD-FAILS upstream of the CSV instead of
silently inheriting the parent-LF label (a genuinely GPS-less short still
falls back).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.validation.content_gates import _make_short_regions_fn

# Real cached coords (mirrors tests/test_short_location_invariant.py).
_FRIPP = (32.3095, -80.6332)  # Beaufort County, SC
_BEAUFORT = {"county": "Beaufort County", "state": "South Carolina", "hamlet": "Fort Fremont"}
_COLLETON = {"county": "Colleton County", "state": "South Carolina", "town": "Edisto Beach"}


class _Geo:
    """Mock geocoder keyed by nearest cached point."""

    def reverse_geocode_detailed(self, lat, lon):
        return dict(_BEAUFORT) if abs(lat - _FRIPP[0]) < 0.2 else dict(_COLLETON)


def _clip_pts(latlon, n):
    # Moving trace (lat advances ~22 m/s, staying within the same county): a real
    # short is cut from moving footage; a parked fixture would trip the
    # stop-free gate. The tiny delta keeps the county vote unchanged.
    return [(float(i), latlon[0] + i * 0.0002, latlon[1]) for i in range(n)]


def _mk_sidecar(tmp_path: Path, date: str, shorts: list) -> list:
    """Write a day-first shorts OST sidecar; return parallel short paths."""
    stages = tmp_path / "_stages"
    shorts_dir = stages / f"day_{date}" / "shorts"
    shorts_dir.mkdir(parents=True)
    (stages / f"shorts_ost_audit_{date}.json").write_text(
        json.dumps({"date": date, "shorts": shorts})
    )
    return [str(shorts_dir / f"short_{s['rank']:02d}.mp4") for s in shorts]


# ── sidecar schema carries per-short regions ────────────────────────────────


def test_format_shorts_ost_audit_persists_regions():
    from gomoveshift.pipeline.ost_audit_sidecar import format_shorts_ost_audit

    payload = format_shorts_ost_audit(
        "2026-06-01",
        [
            {
                "rank": 1,
                "parent_lf_index": 1,
                "ost_text": "Conor Pass, County Kerry",
                "start_sec": 0.0,
                "end_sec": 30.0,
                "regions": ["County Kerry, Munster"],
            }
        ],
    )
    assert payload["shorts"][0]["regions"] == ["County Kerry, Munster"]


def test_render_day_shorts_persists_clip_regions(monkeypatch, tmp_path):
    """``_render_day_shorts`` records each short's clip-GPS vote regions."""
    import gomoveshift.pipeline.day_pipeline as dp

    monkeypatch.setattr(dp, "_render_short_ffmpeg", lambda *a, **k: True)
    monkeypatch.setattr(dp, "trim_gps_to_clip_timeline", lambda pts, ranges: pts)
    monkeypatch.setattr(
        dp, "_resolve_short_ost_text", lambda c, p, r, d, drive_location=None: (c, 0)
    )
    monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda b: [])
    monkeypatch.setattr(dp, "assign_shorts_to_lfs_by_source_window", lambda **k: {})
    monkeypatch.setattr(dp, "shorts_gate_or_raise", lambda *a, **k: None)

    short_range = SimpleNamespace(rank=1, start_sec=0.0, end_sec=38.0)
    _, _, entries = dp._render_day_shorts(
        day_dir=tmp_path,
        date="2026-01-01",
        day_batch=SimpleNamespace(episodes=[]),
        short_ranges=[short_range],
        allow_short_count=True,
        gps_points_by_date={"2026-01-01": _clip_pts(_FRIPP, 8)},
        lf_plans=[],
        per_lf_ost_text=None,
        per_lf_locations=None,
        location="Beaufort County, South Carolina",
        short_ost_text_fn=None,  # burned text falls back to the passing drive-wide label
        master_source_for_extract=Path("/dev/null"),
        short_regions_fn=_make_short_regions_fn(_Geo()),
    )
    assert entries[0]["regions"] == ["Beaufort County, South Carolina"]


# ── regions (not the label) flow into sampled_regions ───────────────────────


def test_sidecar_regions_flow_into_sampled_regions(tmp_path):
    """Drive path: persisted clip regions reach ``sampled_regions`` via
    ``generate_all(per_short_regions=...)``, independent of the parent label."""
    from gomoveshift.pipeline.drive_phases.phase_d import _build_social_posts_per_lf
    from gomoveshift.pipeline.social_generator import SocialGenerator

    short_paths = _mk_sidecar(
        tmp_path,
        "2026-06-01",
        [
            {
                "rank": 1,
                "parent_lf_index": 1,
                "ost_text": "Conor Pass, County Kerry",
                "start_sec": 0.0,
                "end_sec": 30.0,
                "regions": ["County Kerry, Munster"],
                "place_source": "clip_gps",
            }
        ],
    )
    out = _build_social_posts_per_lf(
        social_gen_cls=SocialGenerator,
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
        shorts_per_lf={1: short_paths},
        per_lf_locations_override={1: "County Cork, Munster"},  # DIFFERENT parent label
        project_base="proj",
        output_base=tmp_path / "out",
    )
    assert out[1][0].sampled_regions == ["County Kerry, Munster"]


# ── missing label + clip regions → hard fail upstream of the CSV ────────────


def test_missing_label_with_clip_regions_hard_fails(tmp_path):
    """A GPS-bearing short whose sidecar label is missing must HARD-FAIL."""
    from gomoveshift.pipeline.drive_phases.phase_d import _build_social_posts_per_lf
    from gomoveshift.pipeline.social_generator import SocialGenerator

    short_paths = _mk_sidecar(
        tmp_path,
        "2026-06-01",
        [
            {
                "rank": 1,
                "parent_lf_index": 1,
                "ost_text": "",  # LABEL MISSING
                "start_sec": 0.0,
                "end_sec": 30.0,
                "regions": ["County Kerry, Munster"],  # but the clip HAS GPS votes
            }
        ],
    )
    with pytest.raises(RuntimeError, match="(?i)region"):
        _build_social_posts_per_lf(
            social_gen_cls=SocialGenerator,
            drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
            shorts_per_lf={1: short_paths},
            per_lf_locations_override={1: "County Cork, Munster"},
            project_base="proj",
            output_base=tmp_path / "out",
        )


def test_gps_less_missing_label_falls_back_to_parent_no_raise(tmp_path):
    """A GPS-less short (no clip regions) with a missing label keeps the legacy
    silent parent-LF fallback — the hard fail is scoped to GPS-bearing shorts."""
    from gomoveshift.pipeline.drive_phases.phase_d import _build_social_posts_per_lf
    from gomoveshift.pipeline.social_generator import SocialGenerator

    short_paths = _mk_sidecar(
        tmp_path,
        "2026-06-01",
        [
            {
                "rank": 1,
                "parent_lf_index": 1,
                "ost_text": "",  # no label
                "start_sec": 0.0,
                "end_sec": 30.0,
                "regions": [],  # GPS-less: no clip regions
            }
        ],
    )
    out = _build_social_posts_per_lf(
        social_gen_cls=SocialGenerator,
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
        shorts_per_lf={1: short_paths},
        per_lf_locations_override={1: "County Kerry, Ireland"},
        project_base="proj",
        output_base=tmp_path / "out",
    )
    assert "County Kerry" in out[1][0].youtube_description
