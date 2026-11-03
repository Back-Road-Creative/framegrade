"""``--placeless-label``: a made-up on-screen label that names NO place.

A drive with no GPS fix at all cannot honestly name a place, and every place
gate rightly refuses an operator label it cannot check against a trace. This
flag is the one path that is not a bypass: the label is held to a closed
vocabulary of words that cannot name a place, the LF is awarded the ladder's
tier 5 (no location claim), and the gate re-checks the vocabulary itself.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.cli.parser import _build_parser, _validate_per_lf_location_args
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _assert_pre_encode_location_invariant,
    _build_ladder_evidence,
    _degrade_labels_to_awarded_tier,
)
from gomoveshift.pipeline.validation.content_gates import (
    PLACELESS_LABEL_WORDS,
    _GENERIC_LABEL_TOKENS,
    _location_claim_surfaces,
    check_specificity_ladder,
)

GOOD = ["Somewhere Between", "The Unmarked Miles", "Back Country Night"]


def _parser():
    with patch("importlib.metadata.version", return_value="0"):
        return _build_parser()


def _parse(*extra):
    return _parser().parse_args(["drive", "/src", *extra])


# -- argparse: the vocabulary is enforced by construction --------------------------


@pytest.mark.parametrize("label", GOOD)
def test_accepts_placeless_labels(label):
    assert _parse("--placeless-label", label).placeless_label == label


def test_absent_flag_is_none():
    assert _parse().placeless_label is None


@pytest.mark.parametrize(
    ("label", "needle"),
    [
        ("Somewhere Near Kenmare", "near"),
        ("Lock Haven Backroads", "lock"),
        ("Route 9 Somewhere", "digit"),
        ("Somewhere to Nowhere", "to"),
        ("Somewhere, Between", "letters"),
        ("Back-Country Night", "letters"),
        ("   ", "blank"),
        ("A" * 41, "40"),
    ],
)
def test_rejects_labels_that_could_name_a_place(label, needle, capsys):
    with pytest.raises(SystemExit):
        _parse("--placeless-label", label)
    assert needle in capsys.readouterr().err.lower()


def test_rejection_names_every_offending_word(capsys):
    with pytest.raises(SystemExit):
        _parse("--placeless-label", "Somewhere Near Kenmare")
    err = capsys.readouterr().err.lower()
    assert "near" in err and "kenmare" in err


def test_vocabulary_excludes_to_via_and_proper_nouns():
    assert "to" not in PLACELESS_LABEL_WORDS
    assert "via" not in PLACELESS_LABEL_WORDS
    assert (_GENERIC_LABEL_TOKENS - {"via"}) <= PLACELESS_LABEL_WORDS
    assert all(w == w.lower() and re.fullmatch(r"[a-z]+", w) for w in PLACELESS_LABEL_WORDS)


# -- cross-flag: a tracked / named drive names its places ---------------------------


@pytest.mark.parametrize(
    "other",
    [
        ("--location", "County Kerry, Ireland"),
        ("--per-lf-location", "1=Alpha to Beta via Route 9, Example County"),
        ("--day-gpx", "/tmp/day.gpx"),
    ],
)
def test_placeless_label_is_exclusive_with_place_naming_flags(other):
    args = _parse("--placeless-label", "Somewhere Between", *other)
    with pytest.raises(argparse.ArgumentTypeError, match="placeless"):
        _validate_per_lf_location_args(args)


def test_placeless_label_alone_passes_cross_flag_validation():
    _validate_per_lf_location_args(_parse("--placeless-label", "Somewhere Between"))


# -- the gate -----------------------------------------------------------------------


def test_gate_passes_a_placeless_label_with_placeless_evidence():
    res = check_specificity_ladder(
        {"location": "Somewhere Between"}, {"gps_points": None, "placeless": True}
    )
    assert res.passed, res.message


def test_gate_fails_a_place_word_even_with_the_placeless_flag():
    res = check_specificity_ladder(
        {"location": "Somewhere Near Kenmare"}, {"gps_points": None, "placeless": True}
    )
    assert not res.passed
    assert "near" in res.message.lower()


def test_gate_still_fails_closed_without_the_placeless_flag():
    res = check_specificity_ladder({"location": "Somewhere Between"}, {"gps_points": None})
    assert not res.passed
    assert "no GPS trace" in res.message


def test_pre_encode_invariant_accepts_a_gps_dead_placeless_lf():
    ev = _build_ladder_evidence(
        {1: "Somewhere Between"}, {}, {}, placeless_label="Somewhere Between"
    )
    _assert_pre_encode_location_invariant({1: "Somewhere Between"}, {}, evidence_by_lf=ev)


# -- evidence + tier ----------------------------------------------------------------


def test_ladder_evidence_marks_only_the_placeless_lf():
    ev = _build_ladder_evidence(
        {1: "Somewhere Between", 2: "Elsewhere"}, {}, {}, placeless_label="Somewhere Between"
    )
    assert ev[1]["placeless"] is True
    assert ev[1]["tier_reason"] == "placeless_operator_label"
    assert "placeless" not in ev[2]


def test_ladder_evidence_unchanged_without_a_placeless_label():
    ev = _build_ladder_evidence({1: "Somewhere Between"}, {}, {})
    assert "placeless" not in ev[1]


def test_gps_dead_placeless_lf_is_awarded_tier_5_with_a_recorded_reason():
    ev = _build_ladder_evidence(
        {1: "Somewhere Between"}, {}, {}, placeless_label="Somewhere Between"
    )
    state = SimpleNamespace(
        per_lf_gps_coverage_pct=None, per_lf_awarded_tier=None, per_lf_tier_evidence=None
    )
    # No resolved model (GPS-dead): the LF is still evaluated, never demoted.
    demoted = _degrade_labels_to_awarded_tier({1: "Somewhere Between"}, {}, ev, state=state)
    assert demoted == {}
    assert state.per_lf_awarded_tier == {1: 5}
    assert state.per_lf_tier_evidence[1]["tier_reason"] == "placeless_operator_label"
    assert state.per_lf_gps_coverage_pct in (None, {})


def test_sidecar_records_fix_density_null_with_the_placeless_reason():
    from gomoveshift.pipeline.lf_provenance_sidecar import build_lf_provenance_payload
    from _state_helpers import make_state

    state = make_state(
        per_lf_ost_text={1: "Somewhere Between"},
        per_lf_awarded_tier={1: 5},
        per_lf_tier_evidence={
            1: {"placeless": True, "tier_reason": "placeless_operator_label", "gps_point_count": 0}
        },
    )
    lf = build_lf_provenance_payload(state)["lfs"][0]
    assert lf["fix_density_pct"] is None
    assert lf["awarded_tier"] == "5"
    assert "placeless" in str(lf["null_reasons"]["fix_density_pct"]).lower()


# -- Phase D: nothing the drive publishes claims a place ----------------------------

_SAFE_BOILERPLATE = {
    "pov", "driving", "drive", "road", "trip", "video", "relaxing", "virtual", "travel",
    "dashcam", "with", "me", "route", "long", "in", "k", "exploring", "through",
}  # fmt: skip


def test_phase_d_title_description_and_hashtags_claim_no_place(tmp_path):
    from gomoveshift.pipeline.drive_phases.phase_d_metadata import _build_metadata_per_lf
    from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
    from _state_helpers import make_state

    label = "Somewhere Between"
    gen = LongFormMetadataGenerator(seo_storage_path=Path(tempfile.mkdtemp()))
    lf = SimpleNamespace(index=1, total_duration=1800.0, segments=[])
    state = make_state(
        region="ireland",
        per_lf_locations={1: label},
        per_lf_ost_text={1: label},
        per_lf_titles={1: label},
    )
    meta = _build_metadata_per_lf(
        meta_gen=gen,
        drive_plan=SimpleNamespace(regular_lfs=[lf]),
        regular_lf_paths=["/v/lf.mp4"],
        state=state,
        location=label,
        publish_day_full="Wednesday",
        project_base="p",
        output_base=tmp_path,
    )[1]
    assert label in meta["title"]
    surfaces = [text for _name, text in _location_claim_surfaces(meta)]
    assert surfaces, "the location sweep must see the description's location line"
    words = {w for text in [meta["title"], *surfaces] for w in re.findall(r"[a-z]+", text.lower())}
    stray = words - PLACELESS_LABEL_WORDS - _SAFE_BOILERPLATE
    assert not stray, f"title/description/hashtags carry non-placeless words: {sorted(stray)}"
    assert "ireland" not in (meta["title"] + meta["description"]).lower()


# -- Phase BC: one label fanned out through the per-LF override path -----------------


def test_enrichment_burns_the_placeless_label_on_every_lf(monkeypatch):
    from gomoveshift.pipeline import gps_enrichment as ge
    from _state_helpers import make_state

    lfs = [SimpleNamespace(index=1, segments=[]), SimpleNamespace(index=2, segments=[])]
    monkeypatch.setattr(ge, "_partition_lf_plans_by_date", lambda *a, **k: {"2026-01-01": lfs})
    monkeypatch.setattr(ge, "_build_shared_geocoder", lambda **k: None)
    state = make_state(placeless_label="Somewhere Between")
    out = ge._enrich_per_lf_gps(
        state=state,
        real_plan=SimpleNamespace(regular_lfs=lfs),
        day_batches=[],
        episode_files=[],
        per_day_locations={},
        drive_wide_location="Somewhere Between",
    )
    assert out.per_lf_overrides == {1: "Somewhere Between", 2: "Somewhere Between"}
    assert state.per_lf_overrides == out.per_lf_overrides
    assert out.per_lf_ost_text_by_date["2026-01-01"] == {
        1: "Somewhere Between",
        2: "Somewhere Between",
    }
