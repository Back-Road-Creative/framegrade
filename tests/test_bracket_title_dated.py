"""A country or state bracket label is dated by construction in the TITLE input.

The burned-in OST text stays the bare bracket label; only the title input carries the
LF's measured drive date, so same-country drives never collide on a bare title.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.drive_phases import phase_d_metadata
from gomoveshift.pipeline.exceptions import PhaseBCExitContractError
from gomoveshift.pipeline.title_builder import (
    bracket_dated_title_inputs,
    ensure_unique_titles,
    strip_drive_date,
)

from _state_helpers import make_state

_ORPHAN = "ireland (sep 16, 2025) in 4k"


def _record(level="country", label="Ireland"):
    return {"level": level, "label": label, "gap_h": 41.5}


def _run(tmp_path, ost_text, record, dates, published=frozenset()):
    state = make_state(
        per_lf_ost_text=ost_text,
        per_lf_recording_date=dates,
        bracket_route=record,
        source_dir="/footage/Ireland-Kerry",
    )
    gen = MagicMock()
    gen.generate.side_effect = lambda **kw: {"title": kw["title_override"], "description": "d"}
    plan = SimpleNamespace(
        regular_lfs=[SimpleNamespace(index=i, total_duration=60.0, segments=[]) for i in ost_text]
    )
    with (
        patch(
            "gomoveshift.pipeline.longform_metadata.known_channel_titles",
            return_value=set(published),
        ),
        patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=True, message=""),
        ),
    ):
        phase_d_metadata._build_metadata_per_lf(
            meta_gen=gen,
            drive_plan=plan,
            regular_lf_paths=["/v/lf.mp4"] * len(ost_text),
            state=state,
            location="x",
            publish_day_full="Wed",
            project_base="p",
            output_base=tmp_path,
        )
    return state, gen


def _titles(gen):
    return [c.kwargs["title_override"] for c in gen.generate.call_args_list]


@pytest.fixture(autouse=True)
def _no_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gomoveshift.pipeline.longform_metadata.live_catalog_title_ids", lambda: {}, raising=False
    )


def test_country_bracket_title_carries_the_drive_date(tmp_path):
    out = bracket_dated_title_inputs({1: "Ireland"}, _record(), {1: "2025-09-16"})
    assert out == {1: "Ireland (Sep 16, 2025)"}


def test_state_bracket_title_carries_the_drive_date():
    out = bracket_dated_title_inputs({1: "Ohio"}, _record("state", "Ohio"), {1: "2025-09-16"})
    assert out == {1: "Ohio (Sep 16, 2025)"}


def test_county_bracket_title_is_unchanged():
    rec = _record("county", "County Kerry, Ireland")
    out = bracket_dated_title_inputs({1: "County Kerry, Ireland"}, rec, {1: "2025-09-16"})
    assert out == {1: "County Kerry, Ireland"}


def test_states_leg_bracket_title_is_unchanged():
    rec = _record("states", "Ohio to Indiana")
    assert bracket_dated_title_inputs({1: "Ohio to Indiana"}, rec, {1: "2025-09-16"}) == {
        1: "Ohio to Indiana"
    }


def test_lf_whose_text_is_not_the_bracket_label_is_unchanged():
    out = bracket_dated_title_inputs({1: "County Cork, Ireland"}, _record(), {1: "2025-09-16"})
    assert out == {1: "County Cork, Ireland"}


def test_no_measured_date_leaves_the_label_undated():
    for dates in ({}, {1: ""}, {1: None}, {1: "not-a-date"}):
        assert bracket_dated_title_inputs({1: "Ireland"}, _record(), dates) == {1: "Ireland"}


def test_no_bracket_record_or_invalid_record_leaves_every_title():
    for rec in (None, {}, {"level": "country"}, {"label": "Ireland"}, "x"):
        assert bracket_dated_title_inputs({1: "Ireland"}, rec, {1: "2025-09-16"}) == {1: "Ireland"}


def test_a_label_already_dated_is_not_dated_twice():
    out = bracket_dated_title_inputs(
        {1: "Ireland (Sep 16, 2025)"}, _record(label="Ireland (Sep 16, 2025)"), {1: "2025-09-17"}
    )
    assert out == {1: "Ireland (Sep 17, 2025)"}
    assert strip_drive_date(out[1]) == "Ireland"


def test_each_lf_gets_its_own_measured_date():
    out = bracket_dated_title_inputs(
        {1: "Ireland", 2: "Ireland"}, _record(), {1: "2025-09-16", 2: "2025-09-17"}
    )
    assert out == {1: "Ireland (Sep 16, 2025)", 2: "Ireland (Sep 17, 2025)"}


def test_ost_label_stays_undated_for_a_country_bracket(tmp_path):
    state, gen = _run(tmp_path, {1: "Ireland"}, _record(), {1: "2025-09-16"})
    assert _titles(gen) == ["Ireland (Sep 16, 2025)"]
    assert state.per_lf_ost_text == {1: "Ireland"}


def test_county_bracket_title_in_the_metadata_loop_is_unchanged(tmp_path):
    rec = _record("county", "County Kerry, Ireland")
    _state, gen = _run(tmp_path, {1: "County Kerry, Ireland"}, rec, {1: "2025-09-16"})
    assert _titles(gen) == ["County Kerry, Ireland"]


def test_country_bracket_without_a_measured_date_stays_undated_in_the_loop(tmp_path):
    _state, gen = _run(tmp_path, {1: "Ireland"}, _record(), {})
    assert _titles(gen) == ["Ireland"]


def test_dated_bracket_title_still_refuses_an_exact_live_duplicate(tmp_path):
    with pytest.raises(PhaseBCExitContractError):
        _run(tmp_path, {1: "Ireland"}, _record(), {1: "2025-09-16"}, published={_ORPHAN})


def test_dated_input_still_passes_the_uniqueness_gate_when_nothing_collides():
    dated = bracket_dated_title_inputs({1: "Ireland"}, _record(), {1: "2025-09-16"})
    assert ensure_unique_titles(dated, published_titles={"ireland in 4k"}) == dated


def test_a_label_with_stray_whitespace_is_dated_without_a_double_space():
    out = bracket_dated_title_inputs({1: "Ireland "}, _record(), {1: "2025-09-16"})
    assert out == {1: "Ireland (Sep 16, 2025)"}


def test_the_backstop_strips_a_label_before_dating_so_a_dated_live_title_still_refuses():
    live = "ireland (sep 16, 2025) in 4k"
    with pytest.raises(PhaseBCExitContractError):
        ensure_unique_titles(
            {1: "Ireland "},
            published_titles={"ireland in 4k", live},
            published_title_ids={"ireland in 4k": "V1", live: "V2"},
            drive_project="mine",
            lf_dates={1: "2025-09-16"},
            video_projects={"V1": "other", "V2": "other"},
        )
