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


def _probe_for(wm_duration: float):
    """Concat body is 300 s; the montage-prefixed file is ``wm_duration`` long."""

    def probe(path, *a, **k):
        return SimpleNamespace(duration=wm_duration if "wm" in str(path) else 300.0)

    return probe


def _run(tmp_path, captions: str, *, with_route: bool = True, cfg=None, wm_duration=315.0):
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
        kwargs["output_path"].write_bytes(b"\x00")
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
        patch.object(day_pipeline, "probe_media", side_effect=_probe_for(wm_duration)),
    ):
        try:
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=None,
                output_dir=tmp_path / "out",
                location="Alphaville",
                config=cfg or _cfg(captions),
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
    # Body times shift by the 15 s montage lead (315 s file minus 300 s body); Deltaton (320 s) is inside the
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


# -- montage lead: chapter times are LF-body-relative ---------------------------------
# ChapterGenerator maps GPS through ``included_segments`` against
# ``lf.total_duration`` (the segments' sum, models.py "Sum of segment durations +
# transitions"), and nothing adds the montage; ost_applier burns on the file with the
# montage prepended. So a chapter at body time T is visible at T + the REAL lead.


@pytest.mark.parametrize(
    "wm_duration, expected_alpha",
    [
        (315.0, 315.0),  # planned 15 s montage
        (308.0, 308.0),  # montage came out shorter than planned: card follows the footage
        (300.0, 300.0),  # no montage prepended (no peaks): lead 0
    ],
)
def test_card_lands_where_the_viewer_sees_the_chapters_footage(
    tmp_path, wm_duration, expected_alpha
):
    calls, state = _run(tmp_path, "events", wm_duration=wm_duration)
    body_t = 300.0  # "5:00 Entering Alphaville" in the body-relative chapters
    lead = wm_duration - 300.0
    assert calls[0]["lower_thirds"][0] == ("Alphaville, SC", body_t + lead)
    assert calls[0]["lower_thirds"][0][1] == expected_alpha
    rec = state.get("regular_lf_captions")["2026-02-20"]["1"]
    assert rec["lower_thirds"][0]["time_sec"] == expected_alpha


# -- resume keeps the choice -----------------------------------------------------------


def _drive_cfg(captions: str = "off"):
    from gomoveshift.config import Config

    cfg = Config()
    cfg.drive.gps_captions = captions
    return cfg


def _state_with(cli: str | None, cfg):
    from gomoveshift.pipeline.drive_state_model import PipelineState
    from gomoveshift.pipeline.flag_registry import apply_cli_to_state

    state = PipelineState()
    apply_cli_to_state(state, {"gps_captions": cli}, cfg)
    return state


def test_choice_is_persisted_in_state():
    cfg = _drive_cfg()
    state = _state_with("events", cfg)
    assert state.gps_captions == "events"
    assert cfg.drive.gps_captions == "events"
    assert state.to_dict()["gps_captions"] == "events"


def test_resume_without_flag_keeps_captions_and_fingerprint(tmp_path):
    from gomoveshift.pipeline.flag_registry import merge_on_resume

    first_cfg = _drive_cfg()
    saved = _state_with("events", first_cfg).to_dict()
    cfg1 = _cfg("events")
    calls1, _ = _run(tmp_path, "events", cfg=cfg1)
    assert len(calls1) == 1

    resume_cfg = _drive_cfg()  # a fresh process: config default is off
    resumed = _state_with(None, resume_cfg)
    merge_on_resume(resumed, saved, resume_cfg)
    assert resume_cfg.drive.gps_captions == "events"
    assert resumed.gps_captions == "events"

    # Same on-disk run dir: the burned final is fresh iff the fingerprint is unchanged.
    calls2, _ = _run(tmp_path, "events", cfg=_cfg(resume_cfg.drive.gps_captions))
    assert calls2 == []


def test_cli_flag_differing_from_saved_wins_and_is_persisted(caplog):
    import logging

    from gomoveshift.pipeline.flag_registry import merge_on_resume

    saved = _state_with("events", _drive_cfg()).to_dict()
    cfg = _drive_cfg()
    state = _state_with("off", cfg)
    with caplog.at_level(logging.INFO):
        merge_on_resume(state, saved, cfg)
    assert state.gps_captions == "off"
    assert cfg.drive.gps_captions == "off"
    assert state.to_dict()["gps_captions"] == "off"
    assert any(
        "gps_captions" in r.getMessage() and r.levelno == logging.INFO for r in caplog.records
    )
