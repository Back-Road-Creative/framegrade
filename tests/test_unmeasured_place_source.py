"""A GPS-less drive's shorts pass the place gate only on re-checked evidence.

``place_source`` is ``bracket_route`` only when the short's text is exactly the
label of a valid ``state.bracket_route`` record, and ``placeless_label`` only
when the text claims nothing. The gate re-checks that evidence instead of
trusting the recorded string; ``lf_label`` stays unverified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gomoveshift.pipeline.drive_phases.phase_d_scheduling import (
    _unverified_reasons,
    assert_all_places_verified,
    place_evidence,
    unverified_place_shorts,
)
from gomoveshift.pipeline.drive_text_adapter import attach_drive_text

DATE = "2026-03-05"
BRACKET = {
    "label": "United States",
    "level": "country",
    "gap_h": 17.5,
    "before": {"time": "2026-03-04T23:28:05Z"},
    "after": {"time": "2026-03-05T17:00:12Z"},
}
EVIDENCE = {"bracket_route": BRACKET, "placeless_label": "Somewhere On The Road"}


def _bracket(**over):
    return {**BRACKET, **over}


# -- _unverified_reasons ---------------------------------------------------------


def test_bracket_source_with_matching_record_and_no_regions_is_verified():
    assert _unverified_reasons("United States", [], "bracket_route", EVIDENCE) == []


def test_bracket_source_with_differing_text_is_unverified():
    assert _unverified_reasons("Canada", [], "bracket_route", EVIDENCE)


def test_bracket_source_without_a_record_is_unverified():
    assert _unverified_reasons("United States", [], "bracket_route", {})
    assert _unverified_reasons("United States", [], "bracket_route", None)


def test_bracket_source_past_its_gap_cap_is_unverified():
    ev = {"bracket_route": _bracket(gap_h=49.0)}
    assert _unverified_reasons("United States", [], "bracket_route", ev)
    ev = {"bracket_route": _bracket(level="county", gap_h=40.0)}
    assert _unverified_reasons("United States", [], "bracket_route", ev)


def test_bracket_source_with_clip_regions_is_unverified():
    assert _unverified_reasons("United States", ["Ohio"], "bracket_route", EVIDENCE)


def test_placeless_source_with_claim_free_label_is_verified():
    assert _unverified_reasons("Somewhere On The Road", [], "placeless_label", EVIDENCE) == []


def test_placeless_source_with_a_claiming_word_is_unverified():
    ev = {"placeless_label": "Valleys"}
    assert _unverified_reasons("Valleys", [], "placeless_label", ev)


def test_placeless_source_with_text_other_than_the_label_is_unverified():
    assert _unverified_reasons("Somewhere Between", [], "placeless_label", EVIDENCE)


def test_lf_label_stays_unverified_even_with_evidence():
    assert _unverified_reasons("United States", [], "lf_label", EVIDENCE)


def test_clip_gps_unchanged():
    assert _unverified_reasons("Wythe County", ["Wythe County"], "clip_gps") == []


# -- the gate end to end -----------------------------------------------------------


def _short(stages: Path, rank: int) -> Path:
    return stages / f"day_{DATE}" / "shorts" / f"short_{rank:02d}.mp4"


def _sidecar(stages: Path, source: str, text: str = "United States") -> None:
    stages.mkdir(parents=True, exist_ok=True)
    (stages / f"shorts_ost_audit_{DATE}.json").write_text(
        json.dumps(
            {
                "shorts": [
                    {
                        "rank": 1,
                        "ost_text": text,
                        "regions": [],
                        "start_sec": 0.0,
                        "end_sec": 30.0,
                        "place_source": source,
                    }
                ]
            }
        )
    )


def test_gate_passes_a_bracket_short_given_the_evidence(tmp_path):
    _sidecar(tmp_path, "bracket_route")
    shorts = {1: [_short(tmp_path, 1)]}
    assert unverified_place_shorts(shorts, EVIDENCE) == []
    assert assert_all_places_verified(shorts, evidence=EVIDENCE) == []


def test_gate_refuses_the_same_short_without_evidence(tmp_path):
    _sidecar(tmp_path, "bracket_route")
    assert unverified_place_shorts({1: [_short(tmp_path, 1)]})


def test_gate_refuses_lf_label_with_evidence(tmp_path):
    _sidecar(tmp_path, "lf_label")
    assert unverified_place_shorts({1: [_short(tmp_path, 1)]}, EVIDENCE)


def test_place_evidence_reads_state_object_and_dict():
    class S:
        bracket_route = BRACKET
        placeless_label = None

    assert place_evidence(S())["bracket_route"] == BRACKET
    assert place_evidence({"placeless_label": "x"})["placeless_label"] == "x"
    assert place_evidence(None) == {"bracket_route": None, "placeless_label": None}


# -- day_pipeline records the source -----------------------------------------------


def test_recorded_source_follows_the_evidence():
    from gomoveshift.pipeline.ost_system_review import unmeasured_place_source

    assert unmeasured_place_source(EVIDENCE, "United States") == "bracket_route"
    assert unmeasured_place_source(EVIDENCE, "Somewhere On The Road") == "placeless_label"
    assert unmeasured_place_source(EVIDENCE, "Canada") is None
    assert unmeasured_place_source({}, "United States") is None
    assert unmeasured_place_source({"placeless_label": "Valleys"}, "Valleys") is None
    assert unmeasured_place_source({"bracket_route": _bracket(gap_h=99)}, "United States") is None


# -- attach_drive_text ---------------------------------------------------------------


def test_attach_backs_the_bracket_label_with_a_valid_record():
    meta = attach_drive_text(None, {}, "United States", final_seconds=600.0, bracket_route=BRACKET)
    assert "drive_text_refusal" not in meta
    assert meta["final_text_description"]


def test_attach_still_refuses_any_other_unbacked_name():
    meta = attach_drive_text(None, {}, "Canada", final_seconds=600.0, bracket_route=BRACKET)
    assert meta["drive_text_refusal"] == ["name_unbacked:place:canada"]
    assert meta["final_text_description"] == []


def test_attach_refuses_the_bracket_label_without_or_past_cap_record():
    for rec in (None, _bracket(gap_h=49.0)):
        meta = attach_drive_text(None, {}, "United States", final_seconds=600.0, bracket_route=rec)
        assert meta["drive_text_refusal"] == ["name_unbacked:place:united states"]


def test_reattach_keeps_the_bracket_record():
    from gomoveshift.pipeline.drive_text_adapter import reattach_with_chapters

    meta = attach_drive_text(None, {}, "United States", final_seconds=600.0, bracket_route=BRACKET)
    meta = reattach_with_chapters(None, meta, [])
    assert "drive_text_refusal" not in meta
