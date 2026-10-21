"""Phase D really attaches Waypost's drive-description entries.

The older Phase D tests build entries with a stand-in inside ``generate``, which
would hide a deleted adapter call; these use a generator that sets none. Synthetic
places only ("Harrow", "Pellmere", "Ostry" are invented).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from seo_manager.analytics.final_text_surfaces import compose_description

from gomoveshift.pipeline.drive_phases import phase_d_metadata as pdm
from gomoveshift.pipeline.drive_text_adapter import (
    DriveFacts,
    DriveTextRefusedError,
    attach_drive_text,
    build_drive_data,
)
from gomoveshift.pipeline.title_decision import decide_title

from test_drive_text_adapter import INDEX
from test_final_text_record import SUFFIX, TITLE, _plan_lf, _state

_NO_REDUNDANCY = patch(
    "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
    return_value=SimpleNamespace(passed=True, message=""),
)


def _bare_generator(state):
    """A metadata generator that sets NO ``final_text_description``."""
    loc = state.per_lf_location_evidence[1]
    gen = MagicMock()
    gen.generate.side_effect = lambda **kw: {
        "title": (kw.get("preselected_title_decision") or {}).get("chosen_title", TITLE),
        "description": "d",
        "title_decision": kw.get("preselected_title_decision")
        or decide_title("longform", TITLE, loc, suffix=SUFFIX),
    }
    return gen


def _build(tmp_path, label):
    state = _state()
    lf = _plan_lf()
    lf.total_duration = 600.0
    with _NO_REDUNDANCY:
        per_lf = pdm._build_metadata_per_lf(
            meta_gen=_bare_generator(state),
            drive_plan=SimpleNamespace(regular_lfs=[lf]),
            regular_lf_paths=["/v/lf.mp4"],
            state=state,
            location=label,
            publish_day_full="Wednesday",
            project_base="p",
            output_base=tmp_path,
        )
    return state, lf, per_lf[1]


def test_build_metadata_gives_the_review_waypost_entries(tmp_path):
    _, _, meta = _build(tmp_path, "Harrow")
    assert "drive_text_refusal" not in meta
    assert "Location: Harrow" in compose_description(meta["final_text_description"])
    assert meta["final_text_review"]["description_layout"] == "gms-drive-1"


def test_chapters_rebuild_the_reviewed_entries(tmp_path):
    state, lf, meta = _build(tmp_path, "Harrow")
    state.per_lf_gpx_path = {1: "lf01.gpx"}
    before = compose_description(meta["final_text_description"])
    chapters = [
        {"timestamp": "0:00", "title": "Harrow"},
        {"timestamp": "2:00", "title": "Pellmere"},
        {"timestamp": "4:00", "title": "Harrow"},
    ]
    with (
        patch.object(pdm, "_lf_chapters", return_value=(chapters, "0:00 Harrow")),
        patch.object(pdm, "_below_chapter_floor", return_value=False),
        _NO_REDUNDANCY,
    ):
        pdm._generate_chapters_all_lfs(
            config=MagicMock(),
            drive_plan=SimpleNamespace(regular_lfs=[lf]),
            state=state,
            metadata_per_lf={1: meta},
            location="Harrow",
            project_base="p",
            output_base=tmp_path,
            stages_dir=tmp_path,
        )
    after = compose_description(meta["final_text_description"])
    assert after != before
    assert "2:00" in after and "Pellmere" in after
    assert meta["final_text_review"]["verdict"] == "agree"


def test_refused_attach_clears_earlier_entries_and_names_the_reason():
    route_entry = {"record": {"places": []}, "region_claims": []}
    meta = {"final_text_description": ["stale-entry"]}
    attach_drive_text(route_entry, meta, "Nowhere", final_seconds=600.0)
    assert meta["final_text_description"] == []
    assert meta["drive_text_refusal"] == ["name_unbacked:place:nowhere"]


def test_half_arc_is_dropped_not_rendered():
    for half in ({"journey_start": "Harrow"}, {"journey_end": "Pellmere"}):
        data = build_drive_data(
            DriveFacts(place="Harrow", final_seconds=600.0, passing=["Ostry"], **half), INDEX
        )
        assert data.journey_start is None and data.journey_end is None
        assert not data.passing


def test_unbacked_arc_end_still_refuses():
    try:
        build_drive_data(
            DriveFacts(
                place="Harrow", final_seconds=600.0, journey_start="Harrow", journey_end="Nowhere"
            ),
            INDEX,
        )
    except DriveTextRefusedError as exc:
        assert exc.reasons == ["name_unbacked:journey.end:nowhere"]
    else:
        raise AssertionError("expected refusal")
