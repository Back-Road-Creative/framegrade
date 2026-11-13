"""A GPS-less long-form titled with the drive's bracket leg is backed by the bracket.

2026-11-13, nctopa-2026-03-05-rec0063: the bracket label "Virginia to Pennsylvania"
(level ``states``) reads as an "A to B" endpoint claim, so Phase D refused the title
"Virginia to Pennsylvania in 4K — Part 1 of 3" with "no location evidence record" --
footage with no fix has no such record. The bracket record measured both ends of that
leg; it backs exactly its own label and nothing else.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.location_evidence import TitleEvidenceError, refuse_unsupported_title

ROANOKE_VA = {"county": "Roanoke County", "state": "Virginia", "country": "United States"}
FRANKLIN_PA = {"county": "Franklin County", "state": "Pennsylvania", "country": "United States"}


def _bracket(gap_h=17.535, label="Virginia to Pennsylvania", level="states"):
    return {
        "label": label,
        "level": level,
        "gap_h": gap_h,
        "before_place": ROANOKE_VA,
        "after_place": FRANKLIN_PA,
    }


@pytest.mark.parametrize(
    "title",
    [
        "Virginia to Pennsylvania in 4K — Part 1 of 3",
        "Virginia to Pennsylvania",
        "Virginia to Pennsylvania | 4K Drive",
    ],
)
def test_the_bracket_leg_title_is_backed_by_the_bracket_record(title):
    refuse_unsupported_title(title, None, lf_index=1, bracket=_bracket())


def test_without_a_bracket_the_title_is_still_refused():
    with pytest.raises(TitleEvidenceError, match="no location evidence record"):
        refuse_unsupported_title("Virginia to Pennsylvania in 4K", None, lf_index=1)


def test_a_bracket_past_its_gap_cap_backs_nothing():
    with pytest.raises(TitleEvidenceError):
        refuse_unsupported_title(
            "Virginia to Pennsylvania in 4K", None, lf_index=1, bracket=_bracket(gap_h=40.0)
        )


@pytest.mark.parametrize(
    "title",
    [
        "Pennsylvania to Virginia in 4K",  # against the measured travel order
        "Virginia to Pennsylvania Avenue in 4K",  # a longer place name, not the leg
        "Roanoke to Chambersburg in 4K",  # towns the bracket never measured
    ],
)
def test_the_bracket_backs_only_its_own_label(title):
    with pytest.raises(TitleEvidenceError):
        refuse_unsupported_title(title, None, lf_index=1, bracket=_bracket())


def test_a_non_leg_bracket_level_backs_no_endpoint_claim():
    with pytest.raises(TitleEvidenceError):
        refuse_unsupported_title(
            "Virginia to Pennsylvania in 4K",
            None,
            lf_index=1,
            bracket=_bracket(label="Virginia to Pennsylvania", level="state"),
        )


def test_phase_d_ships_the_bracket_leg_title_of_a_gps_less_lf(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from _state_helpers import make_state
    from gomoveshift.pipeline.drive_phases.phase_d import _build_metadata_per_lf

    title = "Virginia to Pennsylvania in 4K — Part 1 of 3"
    gen = MagicMock()
    gen.generate.side_effect = lambda **kw: {"title": title, "description": "d"}
    with patch(
        "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
        return_value=SimpleNamespace(passed=True, message=""),
    ):
        out = _build_metadata_per_lf(
            meta_gen=gen,
            drive_plan=SimpleNamespace(
                regular_lfs=[SimpleNamespace(index=1, total_duration=1800.0, segments=[])]
            ),
            regular_lf_paths=["/v/lf.mp4"],
            state=make_state(per_lf_location_evidence={}, bracket_route=_bracket()),
            location="x",
            publish_day_full="Wednesday",
            project_base="p",
            output_base=tmp_path,
        )
    assert out[1]["title"] == title
