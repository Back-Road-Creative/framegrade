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


# ── split full_drive day: BC confirms exactly what Phase D ships ────────────

_REC = {"level": "country", "label": "Ireland", "gap_h": 41.5}


def _bc_titles(ost, dates, plans, published=frozenset(), project="ireland-kerry"):
    """Phase BC's post-render site: the title kit's ``unique_titles`` with the LF parts."""
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import _lf_parts_of

    return ensure_unique_titles(
        dict(ost),
        bracket_route=dict(_REC),
        published_titles=set(published),
        published_title_ids={},
        drive_project=project,
        lf_dates=dates,
        video_projects={},
        lf_parts=_lf_parts_of(plans),
    )


def _split_run(tmp_path, ost, dates, parts, published=frozenset(), record=None):
    state = make_state(
        per_lf_ost_text=dict(ost),
        per_lf_recording_date=dict(dates),
        bracket_route=record if record is not None else dict(_REC),
        source_dir="/footage/Ireland-Kerry",
    )
    return _split_run_with_state(tmp_path, state, ost, parts, published)


def _split_run_with_state(tmp_path, state, ost, parts=None, published=frozenset()):
    parts = parts or {}
    gen = MagicMock()

    def _generate(**kw):
        out = {"title": kw["title_override"], "description": "d"}
        decision = kw.get("preselected_title_decision")
        if decision:
            out.update(title=decision["chosen_title"], title_decision=decision)
        return out

    gen.generate.side_effect = _generate
    plans = [
        SimpleNamespace(
            index=i,
            total_duration=60.0,
            segments=[],
            part_index=parts.get(i, (None, None))[0],
            part_count=parts.get(i, (None, None))[1],
        )
        for i in ost
    ]
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
            drive_plan=SimpleNamespace(regular_lfs=plans),
            regular_lf_paths=["/v/lf.mp4"] * len(ost),
            state=state,
            location="x",
            publish_day_full="Wed",
            project_base="p",
            output_base=tmp_path,
        )
    return plans, gen


def test_split_bracket_day_bc_confirms_the_dated_title_phase_d_ships(tmp_path):
    ost = {1: "Ireland", 2: "Ireland"}
    dates = {1: "2025-09-16", 2: "2025-09-16"}
    plans, gen = _split_run(tmp_path, ost, dates, {1: (1, 2), 2: (2, 2)})
    bc = _bc_titles(ost, dates, plans)
    assert bc == {
        1: "Ireland (Sep 16, 2025) — Part 1 of 2",
        2: "Ireland (Sep 16, 2025) — Part 2 of 2",
    }
    assert _titles(gen) == list(bc.values())


def test_split_day_title_is_dated_all_the_way_through_the_real_generator():
    from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

    meta = LongFormMetadataGenerator().generate(
        location="Ireland",
        video_duration=3600.0,
        project_name="p",
        recording_date="2025-09-16",
        title_override="Ireland (Sep 16, 2025) — Part 1 of 2",
        has_gps=False,
        title_part=(1, 2),
    )
    assert meta["title"] == "Ireland (Sep 16, 2025) in 4K — Part 1 of 2"


def test_two_split_days_on_different_dates_get_distinct_titles(tmp_path):
    ost = {1: "Ireland", 2: "Ireland", 3: "Ireland", 4: "Ireland"}
    dates = {1: "2025-09-16", 2: "2025-09-16", 3: "2025-09-17", 4: "2025-09-17"}
    parts = {1: (1, 2), 2: (2, 2), 3: (1, 2), 4: (2, 2)}
    plans, gen = _split_run(tmp_path, ost, dates, parts)
    shipped = _titles(gen)
    assert len(set(shipped)) == 4
    assert shipped == list(_bc_titles(ost, dates, plans).values())
    assert shipped[0].startswith("Ireland (Sep 16, 2025)")
    assert shipped[2].startswith("Ireland (Sep 17, 2025)")


def test_a_non_bracket_split_day_title_is_the_bare_label_plus_the_part_suffix(tmp_path):
    ost = {1: "County Kerry, Ireland", 2: "County Kerry, Ireland"}
    dates = {1: "2025-09-16", 2: "2025-09-16"}
    plans, gen = _split_run(tmp_path, ost, dates, {1: (1, 2), 2: (2, 2)}, record={})
    assert _titles(gen) == [
        "County Kerry, Ireland — Part 1 of 2",
        "County Kerry, Ireland — Part 2 of 2",
    ]


def test_a_non_bracket_unsplit_drive_titles_are_unchanged(tmp_path):
    ost = {1: "County Kerry, Ireland", 2: "County Clare, Ireland"}
    dates = {1: "2025-09-16", 2: "2025-09-17"}
    _, gen = _split_run(tmp_path, ost, dates, {}, record={})
    assert _titles(gen) == ["County Kerry, Ireland", "County Clare, Ireland"]


def test_a_split_label_already_dated_is_not_dated_twice(tmp_path):
    out = ensure_unique_titles(
        {1: "Ireland (Sep 16, 2025)"},
        bracket_route=dict(_REC),
        lf_dates={1: "2025-09-17"},
        lf_parts={1: (1, 2)},
    )
    assert out == {1: "Ireland (Sep 16, 2025) — Part 1 of 2"}


def test_the_part_suffix_is_applied_once(tmp_path):
    out = ensure_unique_titles({1: "Kerry — Part 1 of 2"}, lf_parts={1: (1, 2)})
    assert out == {1: "Kerry — Part 1 of 2"}


def test_a_pre_render_decision_lf_is_counted_in_phase_ds_uniqueness_like_bc(tmp_path):
    """Phase D no longer leaves a pre-render-decision LF out of its dedup map."""
    from gomoveshift.pipeline.title_decision import route_fingerprint

    ost = {1: "Ireland", 2: "Ireland"}
    dates = {1: "2025-09-16", 2: "2025-09-16"}
    state = make_state(
        per_lf_ost_text=dict(ost),
        per_lf_recording_date=dict(dates),
        bracket_route=dict(_REC),
        source_dir="/footage/Ireland-Kerry",
    )
    record = {"segments": []}
    state.per_lf_route_record = {1: record}
    state.per_lf_pre_render_title_decisions = {
        1: {
            "input_label": "decided label",
            "chosen_title": "decided label",
            "ost_text": "Ireland",
            "route_fingerprint": route_fingerprint(record),
        }
    }
    plans, gen = _split_run_with_state(tmp_path, state, ost)
    assert _titles(gen) == ["decided label", "Ireland (Sep 16, 2025) (Part 2 of 2)"]
    assert _bc_titles(ost, dates, plans)[2] == "Ireland (Sep 16, 2025) (Part 2 of 2)"
