"""A superlong part's burned text is checked over that part's OWN
retained ranges with the long-form rule (route_claims).

The day-level ``location`` title and every chapter-derived lower-third are
checked per part, before any part extracts or burns; every refusal is raised
together. Synthetic places only ("Harrow", "Pellmere", "Ostry", "Ashby County",
"Westmark" and "Vell" are invented); no production trace or coordinate.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from _state_helpers import make_state
from gomoveshift.pipeline import day_pipeline
from gomoveshift.pipeline.color_filters import cadence_kwargs
from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError
from gomoveshift.pipeline.route_claims import (
    TIMELINE,
    RouteClaimRefusedError,
    make_superlong_route_fn,
)
from gomoveshift.pipeline.superlong_planner import SuperlongPartPlan

from test_retained_gps_claims import _BandGeocoder
from test_short_route_claims import _VellGeocoder

DATE = "2026-01-01"
# Two 600 s parts on the day master: part 1 = [0, 600), part 2 = [600, 1200).
PLANS = [
    SuperlongPartPlan(part_index=1, part_count=2, ranges=[(0.0, 600.0)], kept_duration_sec=600.0),
    SuperlongPartPlan(
        part_index=2, part_count=2, ranges=[(600.0, 1200.0)], kept_duration_sec=600.0
    ),
]
HARROW, PELLMERE, OSTRY = 9.5, 10.2, 10.7  # _BandGeocoder latitude bands


def _points(lat_of_t=lambda t: HARROW, *, skip=()):
    ts = [t for t in range(1200) if not any(a <= t < b for a, b in skip)]
    return [(float(t), lat_of_t(t), 20.0 + t * 0.0002) for t in ts]


def _chapters(*entering):
    return [{"timestamp": "0:00", "title": "Start - Harrow"}] + [
        {"timestamp": f"{m}:00", "title": f"Entering {place}"} for m, place in entering
    ]


def _run(
    tmp_path,
    points,
    *,
    text="Harrow",
    chapters=None,
    geo=None,
    route_fn=True,
    confirmed=True,
    state=None,
):
    chapters = chapters or {}
    meta_locations, burns, extracts = {}, [], []

    def _meta(**kw):
        meta_locations[kw["part"].part_index] = kw["location"]
        return {
            "title": f"{kw['location']} part {kw['part'].part_index}",
            "description": "d",
            "chapters": chapters.get(kw["part"].part_index, []),
            "video_duration": kw["part"].kept_duration_sec,
            "has_gps": True,
            "location": kw["location"],
        }

    def _ost(input_path, output_path, location, layout, **kw):
        burns.append({"location": location, "lower_thirds": kw.get("lower_thirds")})
        Path(output_path).write_bytes(b"ost")
        return output_path

    def _concat(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        extracts.append(output_path)
        Path(output_path).write_bytes(b"concat")
        return Path(output_path)

    master = tmp_path / "day_master.mp4"
    master.write_bytes(b"m")
    day_dir = tmp_path / f"day_{DATE}"
    day_dir.mkdir(exist_ok=True)
    fn = make_superlong_route_fn(geo or _BandGeocoder()) if route_fn else None
    with (
        patch("gomoveshift.pipeline.superlong_planner.plan_superlong_parts", return_value=PLANS),
        patch.object(day_pipeline, "check_superlong_output_space", lambda *a: None),
        patch.object(day_pipeline, "_build_superlong_part_metadata", side_effect=_meta),
        patch.object(day_pipeline, "apply_ost", side_effect=_ost),
        patch.object(day_pipeline, "extract_and_concat_segments", side_effect=_concat),
    ):
        try:
            result = day_pipeline._produce_superlong_parts(
                date=DATE,
                stop_mask=np.zeros(1200, dtype=bool),
                episodes=[SimpleNamespace(concat_path=Path("/fake/ep.mp4"))],
                file_boundaries=[(0.0, 1200.0)],
                master_path=master,
                day_dir=day_dir,
                config=SimpleNamespace(drive=SimpleNamespace(brand_logo_path=None)),
                location=text,
                ost_crossfade_sec=0.5,
                state=state if state is not None else make_state(ost_audit_confirmed=confirmed),
                day_gps_points=points,
                superlong_route_fn=fn,
            )
            error = None
        except (RouteClaimRefusedError, OstAuditPendingError) as exc:
            result, error = None, exc
    return SimpleNamespace(
        result=result, error=error, meta=meta_locations, burns=burns, extracts=extracts
    )


def _pellmere_in_part2(t):
    return PELLMERE if 700 <= t < 790 else HARROW  # 90 s >= MIN_PLACE_S, part 2 only


# SL4
def test_sl4_text_every_part_supports_burns_unchanged_with_no_note(tmp_path):
    run = _run(tmp_path, _points())
    assert run.error is None
    assert [b["location"] for b in run.burns] == ["Harrow", "Harrow"]
    claims = [r["route_claim"] for r in run.result["records"]]
    assert [c["label"] for c in claims] == ["Harrow", "Harrow"]
    assert [c["reason"] for c in claims] == ["", ""]


# SL1 + SL2 (broaden), and the "check only the first part" / "whole-day" mutations
def test_sl2_a_part_with_an_omitted_place_is_broadened_on_that_part_only(tmp_path):
    run = _run(tmp_path, _points(_pellmere_in_part2))
    assert run.error is None
    assert run.meta == {1: "Harrow", 2: "Ashby County, Westmark"}
    assert [b["location"] for b in run.burns] == ["Harrow", "Ashby County, Westmark"]
    rec2 = run.result["records"][1]
    assert rec2["location"] == "Ashby County, Westmark"  # the thumbnail's title source
    assert rec2["route_claim"]["part_index"] == 2
    assert "omits 'Pellmere'" in rec2["route_claim"]["reason"]


def test_sl2_a_part_with_no_containing_region_is_refused_naming_part_and_place(tmp_path):
    run = _run(
        tmp_path, _points(lambda t: OSTRY if 700 <= t < 790 else HARROW), geo=_VellGeocoder()
    )
    assert run.error is not None
    msg = str(run.error)
    assert "part 2" in msg and "omits 'Ostry'" in msg
    assert "part 1" not in msg
    assert run.burns == [] and run.extracts == []


# SL3
def test_sl3_a_lower_third_naming_a_place_off_its_parts_ranges_is_refused(tmp_path):
    # Pellmere is on the DAY's route (part 2), not on part 1's ranges.
    run = _run(
        tmp_path,
        _points(_pellmere_in_part2),
        text="Harrow to Pellmere",
        chapters={1: _chapters((5, "Pellmere, Ashby County"))},
    )
    assert run.error is not None
    msg = str(run.error)
    assert "part 1" in msg and "place_not_on_route" in msg and "'Pellmere'" in msg
    assert run.burns == []


def test_sl3_a_lower_third_on_its_parts_ranges_burns(tmp_path):
    run = _run(
        tmp_path,
        _points(_pellmere_in_part2),
        chapters={2: _chapters((2, "Pellmere, Ashby County"))},
    )
    assert run.error is None
    assert run.burns[1]["lower_thirds"] == [("Pellmere, Ashby County", 120.0)]


# SL5
def test_sl5_every_refusal_is_raised_together_before_any_part_encodes(tmp_path):
    run = _run(
        tmp_path,
        _points(lambda t: OSTRY if 700 <= t < 790 else HARROW),
        chapters={1: _chapters((5, "Pellmere, Ashby County"))},
        geo=_VellGeocoder(),
    )
    msg = str(run.error)
    assert "2 superlong part(s)" in msg
    assert "part 1" in msg and "part 2" in msg
    assert run.extracts == [] and run.burns == []


# SL6
def test_sl6_each_part_carries_its_own_sampled_record(tmp_path):
    run = _run(tmp_path, _points(skip=[(100, 103)]))
    c1, c2 = (r["route_claim"] for r in run.result["records"])
    for c in (c1, c2):
        assert c["timeline"] == TIMELINE
        assert c["containment_proven"] is False
        assert c["retained_seconds"] == 600.0
    assert c1["gaps"] == [] and c2["gaps"] == []  # 4 s interval: not over GAP_S
    assert c1["coverage_pct"] == 100.0


def test_sl6_a_gap_over_gap_s_inside_a_part_refuses_its_region_fallback(tmp_path):
    run = _run(tmp_path, _points(_pellmere_in_part2, skip=[(900, 912)]))
    msg = str(run.error)
    assert "part 2" in msg and "GPS gap" in msg and "omits 'Pellmere'" in msg
    assert run.burns == []


def test_a_gps_bearing_superlong_day_without_a_route_check_refuses(tmp_path):
    run = _run(tmp_path, _points(), route_fn=False)
    assert "no route check" in str(run.error)
    assert run.burns == []


def test_the_route_record_reaches_the_g_sl_audit(tmp_path):
    run = _run(tmp_path, _points(_pellmere_in_part2), confirmed=False)
    assert isinstance(run.error, OstAuditPendingError) and run.burns == []
    audit = json.loads((tmp_path / f"superlong_ost_audit_{DATE}.json").read_text())
    assert [p["ost_text"] for p in audit["parts"]] == ["Harrow", "Ashby County, Westmark"]


def _forwards(module, caller, callee, kwarg):
    """True when ``caller``'s call to ``callee`` passes ``kwarg=`` by name."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == caller:
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) == callee:
                    return any(k.arg == kwarg for k in node.keywords)
    raise AssertionError(f"{caller} never calls {callee}")


def test_every_hop_forwards_the_superlong_route_check():
    # A hop that drops the kwarg would hand _produce_superlong_parts None,
    # which fails closed on a GPS day — so each hop is pinned by name.
    from gomoveshift.pipeline import day_pipeline, media_api
    from gomoveshift.pipeline.drive_phases import phase_bc_day_first

    k = "superlong_route_fn"
    assert _forwards(day_pipeline, "run_day_pipeline", "_produce_superlong_parts", k)
    assert _forwards(day_pipeline, "run_all_days", "RunSpec", k)
    assert _forwards(media_api, "render_day", "run_day_pipeline", k)
    assert _forwards(phase_bc_day_first, "run", "run_all_days", k)
    import inspect

    assert "make_superlong_route_fn(geocoder)" in inspect.getsource(phase_bc_day_first)


def test_montage_cannot_burn_place_text_of_its_own():
    # M1: the montage's only place text is the LF's checked OST, which
    # apply_ost overlays afterwards. The montage takes no text input and
    # draws none, so it cannot name a place the route check never saw.
    import ast
    import inspect

    from gomoveshift.pipeline import montage_builder

    for fn in (montage_builder.prepend_montage, montage_builder.build_montage):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"title", "date_str", "location", "text", "ost_text"}, fn
    assert "drawtext=" not in inspect.getsource(montage_builder)
    tree = ast.parse(inspect.getsource(day_pipeline))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "prepend_montage"
    ]
    assert calls
    for call in calls:
        # The only splat allowed is the cadence opt-in, which yields output_fps alone.
        splats = [k.value for k in call.keywords if k.arg is None]
        assert all(
            isinstance(v, ast.Call) and getattr(v.func, "id", None) == "cadence_kwargs"
            for v in splats
        )
        assert {k.arg for k in call.keywords if k.arg is not None} <= {
            "raw_lf_path",
            "peaks",
            "output_path",
            "work_dir",
            "master_path",
            "file_boundaries",
        }
    on = SimpleNamespace(drive=SimpleNamespace(output_cadence="source"))
    assert set(cadence_kwargs(on)) == {"output_fps"}
