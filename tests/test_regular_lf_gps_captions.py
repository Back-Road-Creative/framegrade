"""Opt-in ``--gps-captions events`` on regular long-form videos.

Off is today's path (apply_ost gets no lower-thirds); on derives cards from the
LF's own chapters, suppresses any the LF's retained route does not support, spaces
the rest, and refuses BEFORE any encode when the LF has no route evidence.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from _state_helpers import make_state

from gomoveshift.pipeline import day_pipeline
from gomoveshift.pipeline.route_claims import (
    RouteClaimRefusedError,
    RouteEvidence,
    RoutePlace,
    _Bin,
)

CHAPTERS = [
    {"timestamp": "0:00", "title": "Start - Alphaville"},
    {"timestamp": "5:00", "title": "Entering Alphaville, SC"},
    {"timestamp": "5:20", "title": "Entering Deltaton, SC"},  # inside Alphaville's gap
    {"timestamp": "10:00", "title": "Entering Ghostburg, SC"},  # not on the route
    {"timestamp": "15:00", "title": "Entering Charlieville, SC"},
    {"timestamp": "19:45", "title": "Finish - Charlieville"},
]


def _route_record() -> dict:
    places = [
        RoutePlace(n, "Test County", "SC", 120.0, True, f"e-{n}")
        for n in ("Alphaville", "Deltaton", "Charlieville")
    ]
    bins = [_Bin(30.0, "Test County", "SC", p.name) for p in places for _ in range(4)]
    ev = RouteEvidence(360.0, [], places, 0.0, bins)
    return {"record": ev.record(), "region_claims": []}


def _cfg(captions: str) -> SimpleNamespace:
    return SimpleNamespace(
        drive=SimpleNamespace(
            brand_logo_path=None,
            shorts_target_count=9,
            shorts_min_gap_sec=60.0,
            skip_audio=False,
            montage_max_duration_sec=15.0,
            gps_captions=captions,
            regular_lf_lower_third_duration_sec=8.0,
            regular_lf_lower_third_min_gap_sec=45.0,
        )
    )


def _run(tmp_path, captions: str, *, with_route: bool = True):
    ep = tmp_path / "ep.mp4"
    ep.write_bytes(b"\x00")
    for n in ("master", "seg", "wm"):
        (tmp_path / f"{n}.mp4").write_bytes(b"\x00")
    lf_plan = SimpleNamespace(
        index=1,
        segments=[SimpleNamespace(source_file_index=0, start=0.0, end=60.0)],
        montage_peaks=[],
    )
    day_batch = SimpleNamespace(
        date="2026-02-20",
        episodes=[SimpleNamespace(index=1, concat_path=ep, total_duration=300.0)],
    )
    state = make_state(
        ost_audit_confirmed=True,
        per_lf_route_record={1: _route_record()} if with_route else None,
    )
    calls: list[dict] = []

    def fake_apply_ost(**kwargs):
        calls.append(kwargs)
        return kwargs["output_path"]

    with (
        patch.object(
            day_pipeline, "encode_day_master", return_value=tmp_path / "master.mp4"
        ) as enc,
        patch.object(
            day_pipeline, "extract_and_concat_segments", return_value=tmp_path / "seg.mp4"
        ),
        patch.object(day_pipeline, "prepend_montage", return_value=tmp_path / "wm.mp4"),
        patch.object(day_pipeline, "apply_ost", side_effect=fake_apply_ost),
        patch.object(day_pipeline, "plan_shorts", return_value=[]),
        patch.object(day_pipeline, "compute_day_file_boundaries", return_value=[(0.0, 300.0)]),
        patch.object(day_pipeline, "probe_media", return_value=SimpleNamespace(duration=60.0)),
    ):
        try:
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=None,
                output_dir=tmp_path / "out",
                location="Alphaville",
                config=_cfg(captions),
                lf_plans=[lf_plan],
                state=state,
                lf_chapters_fn=lambda lf, loc: CHAPTERS,
            )
        finally:
            run_calls = [c for c in calls if c.get("layout") == "regular_lf"]
            _run.encode_calls = enc.call_count
    return run_calls, state


def test_off_is_todays_call(tmp_path):
    calls, state = _run(tmp_path, "off")
    assert len(calls) == 1
    assert calls[0].get("lower_thirds") is None
    assert "lower_third_duration_sec" not in calls[0]
    assert "lower_third_min_gap_sec" not in calls[0]
    assert state.get("regular_lf_captions") is None


def test_events_derive_space_suppress_and_persist(tmp_path):
    calls, state = _run(tmp_path, "events")
    # Body times shift by the 15 s montage lead; Deltaton (320 s) is inside the
    # 45 s gap after Alphaville's card; Ghostburg is not on the retained route.
    assert calls[0]["lower_thirds"] == [("Alphaville, SC", 315.0), ("Charlieville, SC", 915.0)]
    assert calls[0]["lower_third_duration_sec"] == 8.0
    assert calls[0]["lower_third_min_gap_sec"] == 45.0
    rec = state.get("regular_lf_captions")["2026-02-20"]["1"]
    assert rec["lower_thirds"] == [
        {"label": "Alphaville, SC", "time_sec": 315.0},
        {"label": "Charlieville, SC", "time_sec": 915.0},
    ]
    assert [s["label"] for s in rec["lower_thirds_suppressed"]] == ["Ghostburg, SC"]
    assert "place_not_on_route" in rec["lower_thirds_suppressed"][0]["reason"]


def test_each_card_is_checked_against_the_lfs_route(tmp_path):
    seen = []
    from gomoveshift.pipeline import route_claims

    orig = route_claims.lower_third_issues

    def spy(cards, ev):
        seen.append([c[0] for c in cards])
        return orig(cards, ev)

    with patch.object(route_claims, "lower_third_issues", side_effect=spy):
        _run(tmp_path, "events")
    assert [c[0] for c in seen] == [
        "Alphaville, SC",
        "Deltaton, SC",
        "Ghostburg, SC",
        "Charlieville, SC",
    ]


def test_no_route_evidence_refuses_before_encode(tmp_path):
    with pytest.raises(RouteClaimRefusedError, match="no retained-route evidence"):
        _run(tmp_path, "events", with_route=False)
    assert _run.encode_calls == 0


def test_config_validates_mode():
    from gomoveshift.config import DriveConfig

    assert DriveConfig().gps_captions == "off"
    assert DriveConfig().regular_lf_lower_third_duration_sec == 8.0
    assert DriveConfig().regular_lf_lower_third_min_gap_sec == 45.0
    with pytest.raises(ValueError, match="gps_captions"):
        DriveConfig(gps_captions="always")


def test_cli_flag_parses():
    from unittest.mock import patch as _p

    with _p("importlib.metadata.version", return_value="0"):
        from gomoveshift.cli.parser import _build_parser

        parser = _build_parser()
    sub = next(
        a for a in parser._actions if hasattr(a, "choices") and a.choices and "drive" in a.choices
    )
    flag = next(a for a in sub.choices["drive"]._actions if a.dest == "gps_captions")
    assert flag.choices == ["off", "events"] and flag.default is None
