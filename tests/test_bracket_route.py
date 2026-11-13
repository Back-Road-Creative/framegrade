"""``--bracket-gpx BEFORE AFTER``: a GPS-less drive's on-screen label COMPUTED from the
fixes recorded immediately before and after the footage.

The specificity ladder (``title_builder.award_tier``) has always named a bracket-route
tier ("state only — bracket route alone"), but nothing fed it: the only evidence source
was the LF's own GPS model, so a drive with no fix at all could only carry a claim-free
placeless label. This flag is that feed. The label is never typed: it is the narrowest
region (county → state → country) that BOTH bracketing fixes geocode to, the two fixes
must be in order and close in time, the footage must sit between them, and the ladder
gate re-checks that the burned text is exactly the computed label.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.cli.parser import _build_parser, _validate_per_lf_location_args
from gomoveshift.pipeline import bracket_route as br
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _build_ladder_evidence,
    _degrade_labels_to_awarded_tier,
)
from gomoveshift.pipeline.validation.content_gates import (
    award_tier_for_evidence,
    check_specificity_ladder,
)

T0 = datetime(2025, 9, 16, 9, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _gpx(path: Path, points) -> Path:
    """A minimal GPX track: ``points`` = [(lat, lon, datetime), ...]."""
    body = "".join(
        f'<trkpt lat="{lat}" lon="{lon}"><time>{_iso(t)}</time></trkpt>' for lat, lon, t in points
    )
    path.write_text(
        '<?xml version="1.0"?><gpx version="1.1" creator="test"><trk><trkseg>'
        f"{body}</trkseg></trk></gpx>"
    )
    return path


def _pair(tmp_path: Path, *, before_end=T0, after_start=None, before_pts=None, after_pts=None):
    after_start = after_start or (T0 + timedelta(hours=3))
    before = _gpx(
        tmp_path / "before.gpx",
        before_pts
        if before_pts is not None
        else [(52.0, -9.5, before_end - timedelta(minutes=5)), (52.01, -9.51, before_end)],
    )
    after = _gpx(
        tmp_path / "after.gpx",
        after_pts
        if after_pts is not None
        else [(52.1, -9.3, after_start), (52.11, -9.31, after_start + timedelta(minutes=5))],
    )
    return before, after


KERRY = {"county": "County Kerry", "state": None, "country": "Ireland"}
CORK = {"county": "County Cork", "state": None, "country": "Ireland"}
FRANKLIN_PA = {"county": "Franklin County", "state": "Pennsylvania", "country": "United States"}
ADAMS_PA = {"county": "Adams County", "state": "Pennsylvania", "country": "United States"}
MONTGOMERY_VA = {"county": "Montgomery County", "state": "Virginia", "country": "United States"}


def _geocoder(*places):
    """A stub ``reverse_geocode_detailed``: hands out ``places`` in call order."""
    queue = list(places)
    return lambda lat, lon: queue.pop(0)


# -- argparse ---------------------------------------------------------------------------


def _parser():
    with patch("importlib.metadata.version", return_value="0"):
        return _build_parser()


def _parse(*extra):
    return _parser().parse_args(["drive", "/src", *extra])


def test_bracket_gpx_takes_exactly_two_existing_files(tmp_path):
    before, after = _pair(tmp_path)
    args = _parse("--bracket-gpx", str(before), str(after))
    assert args.bracket_gpx == [str(before), str(after)]
    with pytest.raises(SystemExit):
        _parse("--bracket-gpx", str(before))
    with pytest.raises(SystemExit):
        _parse("--bracket-gpx", str(before), str(tmp_path / "nope.gpx"))


def test_bracket_gpx_refuses_a_text_file(tmp_path):
    before, after = _pair(tmp_path)
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    with pytest.raises(SystemExit):
        _parse("--bracket-gpx", str(before), str(txt))


@pytest.mark.parametrize(
    "flag",
    [
        ("--placeless-label", "Somewhere Between"),
        ("--location", "County Kerry, Ireland"),
        ("--per-lf-location", "1=Kenmare to Cork, County Kerry, Ireland"),
        ("--day-gpx", "/day.gpx"),
    ],
)
def test_bracket_gpx_is_refused_beside_a_typed_or_traced_place(tmp_path, flag):
    before, after = _pair(tmp_path)
    args = _parse("--bracket-gpx", str(before), str(after), *flag)
    with pytest.raises(argparse.ArgumentTypeError, match="--bracket-gpx"):
        _validate_per_lf_location_args(args)


def test_bracket_gpx_alone_passes_validation(tmp_path):
    before, after = _pair(tmp_path)
    _validate_per_lf_location_args(_parse("--bracket-gpx", str(before), str(after)))


# -- the common region ------------------------------------------------------------------


def test_same_county_without_a_state_carries_the_country():
    assert br.common_region(KERRY, KERRY) == ("county", "County Kerry, Ireland")


def test_same_county_with_a_state_is_county_comma_state():
    assert br.common_region(FRANKLIN_PA, FRANKLIN_PA) == (
        "county",
        "Franklin County, Pennsylvania",
    )


def test_different_counties_in_one_state_fall_to_the_state():
    assert br.common_region(FRANKLIN_PA, ADAMS_PA) == ("state", "Pennsylvania")


def test_different_states_in_one_country_name_the_leg():
    # Was ("country", "United States") until 2026-11-13: both ends of the leg are measured.
    assert br.common_region(FRANKLIN_PA, MONTGOMERY_VA) == ("states", "Pennsylvania to Virginia")
    assert br.common_region(FRANKLIN_PA, MONTGOMERY_VA, ("country",)) == (
        "country",
        "United States",
    )


def test_different_counties_with_no_state_fall_to_the_country():
    assert br.common_region(KERRY, CORK) == ("country", "Ireland")


def test_no_common_region_is_refused():
    with pytest.raises(br.BracketRouteRefusedError, match="no common region"):
        br.common_region(KERRY, FRANKLIN_PA)


def test_unresolved_slots_never_match_each_other():
    with pytest.raises(br.BracketRouteRefusedError):
        br.common_region({"county": None, "state": None, "country": None}, KERRY)


# -- resolving the bracket from two files ----------------------------------------------


def test_resolve_uses_the_last_fix_before_and_the_first_fix_after(tmp_path):
    before, after = _pair(tmp_path)
    route = br.resolve_bracket_route(
        before, after, geocode_fn=_geocoder(KERRY, KERRY), footage_dates=["2025-09-16"]
    )
    assert (route.before.lat, route.before.lon) == (52.01, -9.51)
    assert (route.after.lat, route.after.lon) == (52.1, -9.3)
    assert route.before.ts == T0.timestamp()
    assert route.after.ts == (T0 + timedelta(hours=3)).timestamp()
    assert route.gap_h == pytest.approx(3.0)
    assert (route.level, route.label) == ("county", "County Kerry, Ireland")
    assert route.before.source == str(before)
    assert route.after.source == str(after)


def test_record_round_trips_and_is_json_scalar_only(tmp_path):
    import json

    before, after = _pair(tmp_path)
    route = br.resolve_bracket_route(
        before, after, geocode_fn=_geocoder(FRANKLIN_PA, ADAMS_PA), footage_dates=["2025-09-16"]
    )
    rec = route.to_record()
    json.dumps(rec)
    assert rec["label"] == "Pennsylvania"
    assert rec["level"] == "state"
    assert rec["before_place"] == FRANKLIN_PA
    assert rec["after_place"] == ADAMS_PA
    assert rec["before"]["time"] == _iso(T0)
    assert rec["after"]["time"] == _iso(T0 + timedelta(hours=3))
    assert br.BracketRoute.from_record(rec) == route


def test_before_must_precede_after(tmp_path):
    before, after = _pair(tmp_path, before_end=T0 + timedelta(hours=5))
    with pytest.raises(br.BracketRouteRefusedError, match="not before"):
        br.resolve_bracket_route(
            before, after, geocode_fn=_geocoder(KERRY, KERRY), footage_dates=["2025-09-16"]
        )


def test_a_file_with_no_timed_fix_is_refused(tmp_path):
    before, after = _pair(tmp_path, after_pts=[])
    with pytest.raises(br.BracketRouteRefusedError, match="no timed GPS fix"):
        br.resolve_bracket_route(
            before, after, geocode_fn=_geocoder(KERRY, KERRY), footage_dates=["2025-09-16"]
        )


def test_the_gap_caps_are_per_level():
    # A county claim needs a tight bracket; a whole country survives a longer one.
    assert br.BRACKET_MAX_GAP_H_BY_LEVEL == {
        "county": 36.0,
        "state": 36.0,
        "states": 36.0,
        "country": 48.0,
    }
    assert br.BRACKET_MAX_GAP_H == 48.0


def test_a_gap_over_every_cap_is_refused(tmp_path):
    before, after = _pair(tmp_path, after_start=T0 + timedelta(hours=br.BRACKET_MAX_GAP_H + 1))
    with pytest.raises(br.BracketRouteRefusedError, match="apart"):
        br.resolve_bracket_route(
            before, after, geocode_fn=_geocoder(KERRY, KERRY), footage_dates=["2025-09-16"]
        )


def test_a_county_within_its_cap_keeps_the_county(tmp_path):
    before, after = _pair(tmp_path, after_start=T0 + timedelta(hours=30))
    route = br.resolve_bracket_route(
        before, after, geocode_fn=_geocoder(KERRY, KERRY), footage_dates=["2025-09-16"]
    )
    assert (route.level, route.label) == ("county", "County Kerry, Ireland")


def test_a_county_past_its_cap_names_only_the_country(tmp_path):
    # 40 h: both fixes in one county, but that is too long to prove the county.
    before, after = _pair(tmp_path, after_start=T0 + timedelta(hours=40))
    route = br.resolve_bracket_route(
        before, after, geocode_fn=_geocoder(KERRY, KERRY), footage_dates=["2025-09-16"]
    )
    assert (route.level, route.label) == ("country", "Ireland")
    assert br.BRACKET_LEVEL_TIER[route.level] == 4


def test_a_state_past_its_cap_names_only_the_country(tmp_path):
    before, after = _pair(tmp_path, after_start=T0 + timedelta(hours=40))
    route = br.resolve_bracket_route(
        before, after, geocode_fn=_geocoder(FRANKLIN_PA, ADAMS_PA), footage_dates=["2025-09-16"]
    )
    assert (route.level, route.label) == ("country", "United States")


def test_a_long_gap_across_countries_is_refused(tmp_path):
    before, after = _pair(tmp_path, after_start=T0 + timedelta(hours=40))
    with pytest.raises(br.BracketRouteRefusedError, match="common region"):
        br.resolve_bracket_route(
            before,
            after,
            geocode_fn=_geocoder(KERRY, FRANKLIN_PA),
            footage_dates=["2025-09-16"],
        )


def test_footage_outside_the_bracket_is_refused(tmp_path):
    before, after = _pair(tmp_path)
    with pytest.raises(br.BracketRouteRefusedError, match="footage date"):
        br.resolve_bracket_route(
            before, after, geocode_fn=_geocoder(KERRY, KERRY), footage_dates=["2025-09-20"]
        )


def test_footage_on_the_local_day_either_side_of_a_utc_midnight_is_inside(tmp_path):
    before, after = _pair(
        tmp_path,
        before_end=datetime(2025, 9, 16, 23, 30, tzinfo=timezone.utc),
        after_start=datetime(2025, 9, 17, 2, 0, tzinfo=timezone.utc),
    )
    route = br.resolve_bracket_route(
        before, after, geocode_fn=_geocoder(KERRY, KERRY), footage_dates=["2025-09-17"]
    )
    assert route.label == "County Kerry, Ireland"


def test_a_geocoder_miss_is_refused_not_guessed(tmp_path):
    before, after = _pair(tmp_path)
    with pytest.raises(br.BracketRouteRefusedError, match="geocode"):
        br.resolve_bracket_route(
            before, after, geocode_fn=_geocoder(None, KERRY), footage_dates=["2025-09-16"]
        )


def test_a_gopro_file_is_extracted_to_gpx_first(tmp_path, monkeypatch):
    before, after = _pair(tmp_path)
    mp4 = tmp_path / "GX040061.MP4"
    mp4.write_bytes(b"\0")
    calls = []

    def fake_extract(video_path, output_dir, media_offset_s=0.0):
        calls.append((Path(video_path), Path(output_dir)))
        return before

    monkeypatch.setattr(br, "extract_gopro_gpx", fake_extract)
    route = br.resolve_bracket_route(
        mp4,
        after,
        geocode_fn=_geocoder(KERRY, KERRY),
        footage_dates=["2025-09-16"],
        workdir=tmp_path / "work",
    )
    assert calls == [(mp4, tmp_path / "work")]
    assert route.before.source == str(mp4)


def test_a_gopro_file_with_no_telemetry_is_refused(tmp_path, monkeypatch):
    before, after = _pair(tmp_path)
    mp4 = tmp_path / "GX040061.MP4"
    mp4.write_bytes(b"\0")
    monkeypatch.setattr(br, "extract_gopro_gpx", lambda *a, **k: None)
    with pytest.raises(br.BracketRouteRefusedError, match="no GPS telemetry"):
        br.resolve_bracket_route(
            mp4, after, geocode_fn=_geocoder(KERRY, KERRY), footage_dates=["2025-09-16"]
        )


# -- the ladder: evidence, award, gate --------------------------------------------------


def _record(label="County Kerry, Ireland", level="county"):
    return {
        "label": label,
        "level": level,
        "gap_h": 3.0,
        "before": {"lat": 52.01, "lon": -9.51, "time": _iso(T0), "source": "/b.gpx"},
        "after": {"lat": 52.1, "lon": -9.3, "time": _iso(T0 + timedelta(hours=3)), "source": "/a.gpx"},
        "before_place": KERRY,
        "after_place": KERRY,
    }  # fmt: skip


def test_evidence_marks_an_lf_burning_the_bracket_label():
    ev = _build_ladder_evidence(
        {1: "County Kerry, Ireland", 2: "Kenmare to Cork"}, {}, {}, bracket_route=_record()
    )
    assert ev[1]["has_bracket_route"] is True
    assert ev[1]["tier_reason"] == br.BRACKET_TIER_REASON
    assert ev[1]["bracket"] == _record()
    assert "bracket" not in ev[2] and "has_bracket_route" not in ev[2]


def test_a_bracket_award_is_the_level_s_tier():
    assert award_tier_for_evidence({"gps_points": None, "bracket": _record(level="county")}) == (
        None,
        3,
    )
    assert award_tier_for_evidence({"gps_points": None, "bracket": _record(level="state")}) == (
        None,
        4,
    )
    assert award_tier_for_evidence({"gps_points": None, "bracket": _record(level="country")}) == (
        None,
        4,
    )


def test_the_gate_passes_the_computed_label_and_refuses_any_other_text():
    ev = {"gps_points": None, "has_bracket_route": True, "bracket": _record()}
    assert check_specificity_ladder({"location": "County Kerry, Ireland"}, ev).passed
    res = check_specificity_ladder({"location": "Kenmare to Cork, County Kerry, Ireland"}, ev)
    assert not res.passed
    assert "computed bracket label" in res.message


def test_a_model_less_bracket_lf_is_awarded_and_recorded_but_never_demoted():
    state = SimpleNamespace(
        per_lf_gps_coverage_pct=None, per_lf_awarded_tier=None, per_lf_tier_evidence=None
    )
    ev = {1: {"gps_points": None, "has_bracket_route": True, "bracket": _record()}}
    demoted = _degrade_labels_to_awarded_tier({1: "County Kerry, Ireland"}, {}, ev, state=state)
    assert demoted == {}
    assert state.per_lf_awarded_tier == {1: 3}
    rec = state.per_lf_tier_evidence[1]
    assert rec["bracket"]["label"] == "County Kerry, Ireland"
    assert rec["has_bracket_route"] is True
    assert "gps_points" not in rec


# -- OST system review: the computed label passes without a person -----------------------


def test_system_review_agrees_with_the_bracket_label():
    from gomoveshift.pipeline import ost_system_review as osr

    state = {"bracket_route": _record(), "ost_system_review": {}}
    osr.record_system_reviews(
        state, {"2025-09-16": {1: "County Kerry, Ireland", 2: "Kenmare to Cork"}}, {}
    )
    reviews = state["ost_system_review"]["2025-09-16"]["reviews"]
    assert set(reviews) == {"1"}
    assert reviews["1"]["verdict"] == osr.AGREE
    assert "bracket" in reviews["1"]["reason"]
    assert "County Kerry, Ireland" in reviews["1"]["reason"]
    assert _iso(T0) in reviews["1"]["reason"]


def test_shorts_inheriting_the_bracket_label_pass_the_surface_review():
    from gomoveshift.pipeline import ost_system_review as osr

    state = {"bracket_route": _record(), osr.OPT_IN_FIELD: True}
    entries = [
        {
            "rank": 1,
            "start_sec": 0.0,
            "end_sec": 30.0,
            "ost_text": "County Kerry, Ireland",
            "place_source": osr.INHERITED_SOURCE,
        }
    ]
    osr.record_surface_reviews(state, "shorts", "2025-09-16", entries, {})
    ok, why = osr.surface_review_passes(state, "shorts", "2025-09-16", entries)
    assert ok, why


# -- enrichment: one computed label on every LF -----------------------------------------


def test_enrichment_burns_the_bracket_label_on_every_lf(tmp_path, monkeypatch):
    from _state_helpers import make_state

    from gomoveshift.pipeline import gps_enrichment as ge

    before, after = _pair(tmp_path)
    lfs = [SimpleNamespace(index=1, segments=[]), SimpleNamespace(index=2, segments=[])]
    monkeypatch.setattr(ge, "_partition_lf_plans_by_date", lambda *a, **k: {"2025-09-16": lfs})
    geocoder = SimpleNamespace(reverse_geocode_detailed=_geocoder(KERRY, KERRY))
    monkeypatch.setattr(ge, "_build_shared_geocoder", lambda **k: geocoder)
    state = make_state(
        bracket_gpx=[str(before), str(after)],
        episode_dates=["2025-09-16"],
        output_base=str(tmp_path / "out"),
    )
    out = ge._enrich_per_lf_gps(
        state=state,
        real_plan=SimpleNamespace(regular_lfs=lfs),
        day_batches=[],
        episode_files=[],
        per_day_locations={},
        drive_wide_location="",
    )
    label = "County Kerry, Ireland"
    assert out.per_lf_overrides == {1: label, 2: label}
    assert state.per_lf_overrides == out.per_lf_overrides
    assert out.per_lf_ost_text_by_date["2025-09-16"] == {1: label, 2: label}
    assert state.bracket_route["label"] == label
    assert state.location == label


def test_enrichment_refuses_a_bracket_without_a_geocoder(tmp_path, monkeypatch):
    from _state_helpers import make_state

    from gomoveshift.pipeline import gps_enrichment as ge

    before, after = _pair(tmp_path)
    lfs = [SimpleNamespace(index=1, segments=[])]
    monkeypatch.setattr(ge, "_partition_lf_plans_by_date", lambda *a, **k: {"2025-09-16": lfs})
    monkeypatch.setattr(ge, "_build_shared_geocoder", lambda **k: None)
    state = make_state(bracket_gpx=[str(before), str(after)], episode_dates=["2025-09-16"])
    with pytest.raises(RuntimeError, match="bracket"):
        ge._enrich_per_lf_gps(
            state=state,
            real_plan=SimpleNamespace(regular_lfs=lfs),
            day_batches=[],
            episode_files=[],
            per_day_locations={},
            drive_wide_location="",
        )


# -- flag plumbing ----------------------------------------------------------------------


def test_bracket_gpx_is_a_registered_cli_wins_sentinel():
    from gomoveshift.pipeline.drive_state_model import PipelineState
    from gomoveshift.pipeline.flag_registry import FLAG_REGISTRY, Semantic

    rule = next(r for r in FLAG_REGISTRY if r.cli_dest == "bracket_gpx")
    assert rule.state_field == "bracket_gpx"
    assert rule.semantic is Semantic.SENTINEL_CLI_WINS
    fields = {f.name for f in dataclasses.fields(PipelineState)}
    assert {"bracket_gpx", "bracket_route"} <= fields


def test_the_drive_handler_hands_bracket_gpx_to_the_orchestrator():
    import inspect

    from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

    assert "bracket_gpx" in inspect.signature(DrivePipelineOrchestrator.run).parameters
    src = Path("gomoveshift/cli/handlers/drive.py").read_text()
    assert 'bracket_gpx=getattr(args, "bracket_gpx", None)' in src


# -- the backlog launcher -----------------------------------------------------------------


def test_a_bracket_row_launches_with_both_files(tmp_path):
    from test_health_backlog_launch import Rig

    rig = Rig(tmp_path)
    before, after = _pair(tmp_path)
    row = rig.rows[0]
    # The pinned Waypost BacklogRow carries the pair itself (waypost#109).
    rig.rows[0] = dataclasses.replace(row, bracket_gpx=(str(before), str(after)))
    res = rig.run()
    assert res["project"] == row.project, res
    argv, _cwd, _env = rig.launches[0]
    flat = " ".join(map(str, argv))
    assert f"--bracket-gpx {before} {after}" in flat
    assert "--placeless-label" not in flat


def test_a_bracket_row_with_a_missing_file_is_skipped_before_launch(tmp_path):
    from test_health_backlog_launch import Rig

    rig = Rig(tmp_path)
    before, _after = _pair(tmp_path)
    row = rig.rows[0]
    rig.rows[0] = dataclasses.replace(row, bracket_gpx=(str(before), str(tmp_path / "gone.gpx")))
    res = rig.run()
    assert res["project"] == "day-b"
    assert "bracket_gpx" in res["skipped"][0]["reason"]
    assert "gone.gpx" in res["skipped"][0]["reason"]
