"""A short names ONLY the place its own clip was filmed.

A short is ~30 s cut from a long drive. The operator ``--per-lf-location``
label names the whole long-form ("Gettysburg Battlefield, PA"); a clip from
the drive there is not at Gettysburg. So a GPS-bearing short burns its own
clip-GPS place or the run hard-fails upstream of the encode -- the operator
label never overrides the clip, and the region gate has no operator exemption
(Gettysburg short 1 shipped the operator label through that exemption).
Phase D re-checks the sidecar so a short rendered before this rule can never
be planned for publish.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gomoveshift.pipeline.day_pipeline as dp
from gomoveshift.pipeline.drive_phases.phase_d import _build_social_posts_per_lf
from gomoveshift.pipeline.social_generator import SocialGenerator
from gomoveshift.pipeline.validation.content_gates import check_short_location_invariant

_OPERATOR_LF_LABEL = "Gettysburg Battlefield, PA"
_PARENT_AB = "Dingle to Gortmore via R559"
# Moving trace (~22 m/s, one county): a parked fixture trips the stop-free gate.
_PTS = [(float(i), 52.14 + i * 0.0002, -10.27) for i in range(80)]


def _run_shorts(monkeypatch, tmp_path, **overrides):
    monkeypatch.setattr(dp, "_render_short_ffmpeg", lambda **kw: True)
    monkeypatch.setattr(dp, "trim_gps_to_clip_timeline", lambda p, r: p)
    monkeypatch.setattr(
        dp,
        "_resolve_short_ost_text",
        overrides.pop("_resolve", lambda c, p, r, d, drive_location=None: (c, 0)),
    )
    monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda b: [])
    monkeypatch.setattr(dp, "assign_shorts_to_lfs_by_source_window", lambda **k: {1: range(1)})
    kwargs = dict(
        day_dir=tmp_path,
        date="2026-01-01",
        day_batch=SimpleNamespace(episodes=[]),
        short_ranges=[SimpleNamespace(rank=1, start_sec=0.0, end_sec=38.0)],
        allow_short_count=True,
        gps_points_by_date={"2026-01-01": _PTS},
        lf_plans=[SimpleNamespace(index=1)],
        per_lf_ost_text={1: _PARENT_AB},
        per_lf_locations={1: "County Kerry"},
        per_lf_overrides={1: _OPERATOR_LF_LABEL},
        location="Drive Wide",
        master_source_for_extract=Path("/dev/null"),
        state={"ost_audit_confirmed": True, "ost_audit_confirmed_this_run": True},
    )
    kwargs.update(overrides)
    _, _, entries = dp._render_day_shorts(**kwargs)
    return entries


def test_operator_lf_label_never_replaces_the_clips_own_gps_place(monkeypatch, tmp_path):
    entries = _run_shorts(monkeypatch, tmp_path, short_ost_text_fn=lambda p, s: "Annascaul")
    assert entries[0]["ost_text"] == "Annascaul"
    assert entries[0]["place_source"] == "clip_gps"


def test_operator_lf_label_does_not_rescue_a_failed_clip_vote(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="rank 1"):
        _run_shorts(monkeypatch, tmp_path, short_ost_text_fn=lambda p, s: None)


def test_clip_place_rejected_by_the_text_cascade_hard_fails(monkeypatch, tmp_path):
    # The clip voted "Annascaul" but the 9:16 cascade fell back to the parent
    # LF text: that text was never checked against this clip's GPS.
    with pytest.raises(RuntimeError, match="rank 1"):
        _run_shorts(
            monkeypatch,
            tmp_path,
            per_lf_overrides={},
            short_ost_text_fn=lambda p, s: "Annascaul",
            _resolve=lambda c, p, r, d, drive_location=None: (p, 1),
        )


def test_region_gate_has_no_operator_exemption():
    assert "operator_override" not in inspect.signature(check_short_location_invariant).parameters


def _mk_sidecar(tmp_path: Path, place_source=None) -> list:
    stages = tmp_path / "_stages"
    shorts_dir = stages / "day_2026-06-01" / "shorts"
    shorts_dir.mkdir(parents=True)
    entry = {
        "rank": 1,
        "parent_lf_index": 1,
        "ost_text": _OPERATOR_LF_LABEL,
        "start_sec": 0.0,
        "end_sec": 30.0,
        "regions": ["Adams County, Pennsylvania"],
    }
    if place_source is not None:
        entry["place_source"] = place_source
    (stages / "shorts_ost_audit_2026-06-01.json").write_text(
        json.dumps({"date": "2026-06-01", "shorts": [entry]})
    )
    return [str(shorts_dir / "short_01.mp4")]


def _plan(tmp_path, short_paths):
    return _build_social_posts_per_lf(
        social_gen_cls=SocialGenerator,
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
        shorts_per_lf={1: short_paths},
        per_lf_locations_override={1: _OPERATOR_LF_LABEL},
        project_base="proj",
        output_base=tmp_path / "out",
    )


@pytest.mark.parametrize("place_source", ["lf_label", None])
def test_phase_d_refuses_a_gps_short_labelled_by_anything_but_its_clip(tmp_path, place_source):
    with pytest.raises(RuntimeError, match="place its clip was filmed"):
        _plan(tmp_path, _mk_sidecar(tmp_path, place_source))


def test_phase_d_plans_a_gps_short_labelled_by_its_clip(tmp_path):
    out = _plan(tmp_path, _mk_sidecar(tmp_path, "clip_gps"))
    assert out[1][0].sampled_regions == ["Adams County, Pennsylvania"]
