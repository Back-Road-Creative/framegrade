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


def _bc_titles(tmp_path, dates=("2025-09-16",), split=True):
    """What REAL Phase BC (pre-seed, post-render and all) leaves as the per-LF titles."""
    import test_phase_bc_day_first as bc_tests

    harness = bc_tests.TestBCOwnTitleSitesOnASplitDayMatchPhaseD()
    state, _seen = harness._run_bc(tmp_path / "bc", dates, split)
    return {int(k): v for k, v in state.per_lf_titles.items()}


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


@pytest.mark.overpass_answers_empty
def test_split_bracket_day_bc_confirms_the_dated_title_phase_d_ships(tmp_path):
    ost = {1: "Ireland", 2: "Ireland"}
    dates = {1: "2025-09-16", 2: "2025-09-16"}
    plans, gen = _split_run(tmp_path, ost, dates, {1: (1, 2), 2: (2, 2)})
    bc = _bc_titles(tmp_path)
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


@pytest.mark.overpass_answers_empty
def test_two_split_days_on_different_dates_get_distinct_titles(tmp_path):
    ost = {1: "Ireland", 2: "Ireland", 3: "Ireland", 4: "Ireland"}
    dates = {1: "2025-09-16", 2: "2025-09-16", 3: "2025-09-17", 4: "2025-09-17"}
    parts = {1: (1, 2), 2: (2, 2), 3: (1, 2), 4: (2, 2)}
    plans, gen = _split_run(tmp_path, ost, dates, parts)
    shipped = _titles(gen)
    assert len(set(shipped)) == 4
    assert shipped == list(_bc_titles(tmp_path, ("2025-09-16", "2025-09-17")).values())
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


@pytest.mark.overpass_answers_empty
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
    bc = _bc_titles(tmp_path, split=False)
    record = {"segments": []}
    state.per_lf_route_record = {1: record}
    state.per_lf_pre_render_title_decisions = {
        1: {
            "input_label": bc[1],
            "chosen_title": bc[1],
            "ost_text": "Ireland",
            "route_fingerprint": route_fingerprint(record),
        }
    }
    plans, gen = _split_run_with_state(tmp_path, state, ost)
    assert _titles(gen) == [bc[1], "Ireland (Sep 16, 2025) (Part 2 of 2)"]
    assert _titles(gen) == [bc[1], bc[2]]


# ── a confirm flag is one-shot; a title-only suffix change never restales a day ──

_DAY = "2025-09-20"
_COUNTY = "County Cork, Ireland"
_PARTS = {1: (1, 2), 2: (2, 2)}
_LEGACY_SPLIT = {1: f"{_COUNTY} (Part 1 of 2)", 2: f"{_COUNTY} (Part 2 of 2)"}


def _gate_state(titles, ost=None, stored=None, cli=None, review=None, date=_DAY):
    ost = ost or {1: _COUNTY, 2: _COUNTY}
    return {
        "per_lf_ost_text": dict(ost),
        "per_lf_titles": dict(titles),
        "per_lf_recording_date": {k: date for k in ost},
        "per_lf_locations": {},
        "ost_audit_passed": {date: True} if stored else {},
        "ost_audit_hash": {date: stored} if stored else {},
        "ost_audit_confirmed_this_run": cli,
        "ost_system_review_this_run": review,
    }


def _audit_hash(titles, ost=None, date=_DAY):
    from gomoveshift.pipeline.ost_audit_sidecar import compute_ost_audit_hash, format_ost_audit

    state = _gate_state(titles, ost, date=date)
    return compute_ost_audit_hash(format_ost_audit(date, state, sorted(titles)))


# The hash the DEPLOYED code stored for the day confirmed under ``_LEGACY_SPLIT`` titles. Frozen
# from origin/main (02861d3f) running ``compute_ost_audit_hash(format_ost_audit(...))`` over that
# day's state; no drive ever stored the normalised formula this branch added.
_DEPLOYED_SPLIT_HASH = "3d42cc7f7fde5e5eb9bb113106a15de010177a8735a4f75c6bbaef9c7087548e"  # pragma: allowlist secret (sha256 of an OST audit payload)


def _gate(tmp_path, state, date=_DAY, indices=(1, 2)):
    from gomoveshift.pipeline.ost_audit_sidecar import lf_gate_or_raise

    lf_gate_or_raise(tmp_path, date, list(indices), state)


def test_the_resume_recipe_never_carries_the_one_shot_confirm_flag(tmp_path):
    from gomoveshift.health.resume import read_recipe, write_recipe

    argv = ["drive", "/src", "--ost-system-review", "--ost-audit-confirmed", "--fresh"]
    path = write_recipe(tmp_path, argv, cwd=str(tmp_path), sha="x", now_iso="2026-11-23T00:00:00")
    replayed = read_recipe(path)["argv"]
    assert "--ost-audit-confirmed" not in replayed
    assert "--fresh" not in replayed
    assert "--ost-system-review" in replayed  # only the one-shot confirm is dropped


def test_the_recipe_file_itself_never_stores_the_one_shot_confirm_flag(tmp_path):
    import json

    from gomoveshift.health.resume import write_recipe

    argv = ["drive", "/src", "--ost-audit-confirmed"]
    path = write_recipe(tmp_path, argv, cwd=str(tmp_path), sha="x", now_iso="2026-11-23T00:00:00")
    assert "--ost-audit-confirmed" not in json.loads(path.read_text(encoding="utf-8"))["argv"]


def test_a_recipe_already_on_disk_with_the_confirm_flag_replays_without_it(tmp_path):
    import json

    from gomoveshift.health.resume import RECIPE_NAME, read_recipe

    path = tmp_path / RECIPE_NAME
    argv = ["py", "-m", "gomoveshift", "drive", "/src", "--ost-audit-confirmed", "--no-review"]
    path.write_text(json.dumps({"argv": argv, "cwd": str(tmp_path)}), encoding="utf-8")
    assert read_recipe(path)["argv"] == [a for a in argv if a != "--ost-audit-confirmed"]


def test_a_recipe_relaunch_cannot_autopass_a_day_whose_burn_text_changed(tmp_path):
    from gomoveshift.health.resume import read_recipe, write_recipe
    from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

    argv = ["drive", "/src", "--ost-system-review", "--ost-audit-confirmed"]
    path = write_recipe(tmp_path, argv, cwd=str(tmp_path), sha="x", now_iso="2026-11-23T00:00:00")
    relaunch_confirmed = "--ost-audit-confirmed" in read_recipe(path)["argv"]
    old = {1: _COUNTY, 2: _COUNTY}
    new = {1: "County Kerry, Ireland", 2: "County Kerry, Ireland"}
    state = _gate_state(
        new, ost=new, stored=_audit_hash(old), cli=True if relaunch_confirmed else None
    )
    with pytest.raises(OstAuditPendingError):
        _gate(tmp_path, state)
    assert state["ost_audit_hash"][_DAY] == _audit_hash(old)  # nothing was re-stamped


def test_an_operator_confirm_binds_to_the_hash_it_confirmed_only(tmp_path):
    from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

    old = {1: _COUNTY, 2: _COUNTY}
    new = {1: "County Kerry, Ireland", 2: "County Kerry, Ireland"}
    state = _gate_state(new, ost=new, stored=_audit_hash(old), cli=True)
    _gate(tmp_path, state)  # the operator's own run confirms what they reviewed
    assert state["ost_audit_hash"][_DAY] == _audit_hash(new, new)
    later = {1: "County Clare, Ireland", 2: "County Clare, Ireland"}
    again = _gate_state(later, ost=later, stored=state["ost_audit_hash"][_DAY])
    with pytest.raises(OstAuditPendingError):  # a later burn-text change stales it again
        _gate(tmp_path, again)


def test_the_stale_message_names_the_day_and_the_review_step_not_the_flag(tmp_path):
    from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

    new = {1: "County Kerry, Ireland", 2: "County Kerry, Ireland"}
    state = _gate_state(new, ost=new, stored=_audit_hash({1: _COUNTY, 2: _COUNTY}))
    with pytest.raises(OstAuditPendingError) as exc:
        _gate(tmp_path, state)
    msg = str(exc.value)
    assert _DAY in msg and "ost_audit_2025-09-20.json" in msg
    assert "changed since it was confirmed" in msg
    assert "re-run with --ost-audit-confirmed" not in msg


def test_a_split_day_title_only_rerun_does_not_restale_a_confirmed_day(tmp_path):
    from gomoveshift.pipeline import ost_system_review as osr
    from gomoveshift.pipeline.route_claims import route_claim_outcome

    new = ensure_unique_titles({1: _COUNTY, 2: _COUNTY}, lf_parts=_PARTS)
    assert new != _LEGACY_SPLIT
    state = _gate_state(new, stored=_DEPLOYED_SPLIT_HASH, review=True)
    outcomes = {i: route_claim_outcome(i, _COUNTY, None, chosen=_COUNTY, reason="") for i in (1, 2)}
    osr.record_system_reviews(state, {_DAY: dict(state["per_lf_ost_text"])}, outcomes)
    _gate(tmp_path, state)  # no raise: the burn text is unchanged


def test_the_frozen_deployed_hash_is_the_legacy_form_the_gate_derives_for_the_new_titles():
    from gomoveshift.pipeline.ost_audit_sidecar import (
        format_ost_audit,
        legacy_ost_audit_hashes,
    )

    new = ensure_unique_titles({1: _COUNTY, 2: _COUNTY}, lf_parts=_PARTS)
    payload = format_ost_audit(_DAY, _gate_state(new), [1, 2])
    assert _DEPLOYED_SPLIT_HASH in legacy_ost_audit_hashes(payload)


def test_the_deployed_split_hash_never_covers_a_changed_burn_or_base_label(tmp_path):
    from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

    changed = {1: "County Kerry, Ireland", 2: "County Kerry, Ireland"}
    with pytest.raises(OstAuditPendingError):
        _gate(
            tmp_path,
            _gate_state(
                ensure_unique_titles(dict(changed), lf_parts=_PARTS),
                ost=changed,
                stored=_DEPLOYED_SPLIT_HASH,
            ),
        )
    relabelled = ensure_unique_titles({1: "Cork", 2: "Cork"}, lf_parts=_PARTS)
    with pytest.raises(OstAuditPendingError):  # same burn text, a different base label
        _gate(tmp_path, _gate_state(relabelled, stored=_DEPLOYED_SPLIT_HASH))


def test_a_split_day_hash_of_the_new_titles_is_not_covered_by_the_deployed_form_of_another_day():
    from gomoveshift.pipeline.ost_audit_sidecar import (
        format_ost_audit,
        legacy_ost_audit_hashes,
    )

    three = ensure_unique_titles({1: _COUNTY, 2: _COUNTY}, lf_parts={1: (1, 3), 2: (2, 3)})
    payload = format_ost_audit(_DAY, _gate_state(three), [1, 2])
    assert _DEPLOYED_SPLIT_HASH not in legacy_ost_audit_hashes(payload)


def test_an_in_flight_bracket_drive_is_not_reopened_by_the_added_drive_date(tmp_path):
    undated = _audit_hash({1: "Ireland"}, ost={1: "Ireland"}, date="2025-09-16")
    state = _gate_state(
        {1: "Ireland (Sep 16, 2025)"}, ost={1: "Ireland"}, stored=undated, date="2025-09-16"
    )
    _gate(tmp_path, state, date="2025-09-16", indices=(1,))


def test_a_stored_hash_of_the_dated_title_still_covers_it(tmp_path):
    """kerry-to-cork stored the hash of its already-dated title: not restaled."""
    from gomoveshift.pipeline.ost_audit_sidecar import compute_ost_audit_hash, format_ost_audit

    state = {
        "per_lf_ost_text": {1: "Ireland"},
        "per_lf_titles": {1: "Ireland (Sep 16, 2025)"},
        "per_lf_recording_date": {1: "2025-09-16"},
        "per_lf_locations": {1: "Ireland"},
    }
    stored = compute_ost_audit_hash(format_ost_audit("2025-09-16", state, [1]), normalise=False)
    assert stored.startswith("6343391fddc6")
    state["ost_audit_passed"] = {"2025-09-16": True}
    state["ost_audit_hash"] = {"2025-09-16": stored}
    _gate(tmp_path, state, date="2025-09-16", indices=(1,))


@pytest.mark.parametrize(
    "changed",
    [
        {1: "County Kerry, Ireland", 2: "County Kerry, Ireland"},  # burn text
    ],
)
def test_a_burn_text_change_still_stales_a_split_day(tmp_path, changed):
    from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

    stored = _audit_hash(ensure_unique_titles({1: _COUNTY, 2: _COUNTY}, lf_parts=_PARTS))
    new = ensure_unique_titles(dict(changed), lf_parts=_PARTS)
    with pytest.raises(OstAuditPendingError):
        _gate(tmp_path, _gate_state(new, ost=changed, stored=stored))


def test_a_changed_base_label_or_part_count_still_stales_the_day(tmp_path):
    from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

    base = ensure_unique_titles({1: _COUNTY, 2: _COUNTY}, lf_parts=_PARTS)
    stored = _audit_hash(base)
    relabelled = ensure_unique_titles({1: "Cork", 2: "Cork"}, lf_parts=_PARTS)
    with pytest.raises(OstAuditPendingError):  # same burn text, a different base label
        _gate(tmp_path, _gate_state(relabelled, stored=stored))
    three = ensure_unique_titles({1: _COUNTY, 2: _COUNTY}, lf_parts={1: (1, 3), 2: (2, 3)})
    with pytest.raises(OstAuditPendingError):  # the day was re-split into three
        _gate(tmp_path, _gate_state(three, stored=stored))


def test_an_unsplit_undated_lfs_hash_is_the_unnormalised_hash():
    from gomoveshift.pipeline.ost_audit_sidecar import compute_ost_audit_hash, format_ost_audit

    state = _gate_state({1: "Foo to Bar, NC"}, ost={1: "Foo to Bar, NC"})
    payload = format_ost_audit(_DAY, state, [1])
    assert compute_ost_audit_hash(payload) == compute_ost_audit_hash(payload, normalise=False)


# ── a split LF that exactly matches a live title ──

_LIVE_PART1 = {f"{_COUNTY} in 4K — Part 1 of 2"}


def test_a_split_lf_exact_live_duplicate_is_refused_like_an_unsplit_one():
    with pytest.raises(PhaseBCExitContractError):
        ensure_unique_titles(
            {1: _COUNTY, 2: _COUNTY}, published_titles=_LIVE_PART1, lf_parts=_PARTS
        )


def test_a_split_lf_exact_live_duplicate_is_dated_before_the_part_suffix():
    from gomoveshift.pipeline.longform_metadata import assemble_final_title
    from gomoveshift.pipeline.title_builder import append_full_drive_part_suffix

    got = ensure_unique_titles(
        {1: _COUNTY, 2: _COUNTY},
        published_titles=_LIVE_PART1,
        published_title_ids={f"{_COUNTY} in 4K — Part 1 of 2".lower(): "VIDX"},
        drive_project="ireland-2025-09-19-old-head",
        lf_dates={1: "2025-09-19", 2: "2025-09-19"},
        video_projects={"VIDX": "ireland-2025-09-17-baltimore"},
        lf_parts=_PARTS,
    )
    assert got[1] == f"{_COUNTY} (Sep 19, 2025) — Part 1 of 2"
    assert got[2] == f"{_COUNTY} — Part 2 of 2"
    shipped = append_full_drive_part_suffix(assemble_final_title(got[1]), 1, 2)
    assert shipped == f"{_COUNTY} (Sep 19, 2025) in 4K — Part 1 of 2"
    assert "(Part 2)" not in "".join(got.values())


# ── two split days under one label never get a garbled double part suffix ──


def test_two_split_days_with_one_undated_label_never_get_a_double_part_suffix():
    """Unique titles need a date; with none there is no honest title: refuse (fail closed)."""
    with pytest.raises(PhaseBCExitContractError):
        ensure_unique_titles(
            {1: _COUNTY, 2: _COUNTY, 3: _COUNTY, 4: _COUNTY},
            lf_parts={1: (1, 2), 2: (2, 2), 3: (1, 2), 4: (2, 2)},
        )


def test_two_split_days_with_one_label_are_dated_on_the_base_and_suffixed_once():
    got = ensure_unique_titles(
        {1: _COUNTY, 2: _COUNTY, 3: _COUNTY, 4: _COUNTY},
        lf_parts={1: (1, 2), 2: (2, 2), 3: (1, 2), 4: (2, 2)},
        lf_dates={1: "2025-09-19", 2: "2025-09-19", 3: "2025-09-20", 4: "2025-09-20"},
    )
    assert got == {
        1: f"{_COUNTY} (Sep 19, 2025) — Part 1 of 2",
        2: f"{_COUNTY} (Sep 19, 2025) — Part 2 of 2",
        3: f"{_COUNTY} (Sep 20, 2025) — Part 1 of 2",
        4: f"{_COUNTY} (Sep 20, 2025) — Part 2 of 2",
    }
    assert all(t.count("Part") == 1 for t in got.values())
    assert len(set(got.values())) == 4


def test_two_split_days_on_one_date_cannot_be_made_unique_and_are_refused():
    with pytest.raises(PhaseBCExitContractError):
        ensure_unique_titles(
            {1: _COUNTY, 2: _COUNTY, 3: _COUNTY, 4: _COUNTY},
            lf_parts={1: (1, 2), 2: (2, 2), 3: (1, 2), 4: (2, 2)},
            lf_dates={k: "2025-09-19" for k in (1, 2, 3, 4)},
        )


def test_a_split_day_beside_an_unsplit_one_with_the_same_label_is_unchanged():
    got = ensure_unique_titles(
        {1: _COUNTY, 2: _COUNTY, 3: _COUNTY}, lf_parts={1: (1, 2), 2: (2, 2)}
    )
    assert got == {
        1: f"{_COUNTY} — Part 1 of 2",
        2: f"{_COUNTY} — Part 2 of 2",
        3: _COUNTY,
    }


# ── the per-day gate accepts a hash stored before suffixes were normalised ──


def _per_day_state(ost, title):
    date = "2025-09-14"
    return {
        "per_lf_ost_text": {1: ost},
        "per_lf_titles": {1: title},
        "per_lf_recording_date": {1: date},
        "per_lf_locations": {1: "County Kerry"},
    }


def _per_day_run(tmp_path, state):
    from test_day_pipeline_ost_audit_gate import _run

    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "shorts_ost_audit_2025-09-14.confirmed").write_text("ok")
    return _run(tmp_path, state)


def _store_legacy(state):
    from gomoveshift.pipeline.ost_audit_sidecar import compute_ost_audit_hash, format_ost_audit

    payload = format_ost_audit("2025-09-14", state, [1])
    state["ost_audit_passed"] = {"2025-09-14": True}
    state["ost_audit_hash"] = {"2025-09-14": compute_ost_audit_hash(payload, normalise=False)}


def test_the_per_day_gate_passes_a_day_holding_an_old_raw_hash_with_unchanged_titles(tmp_path):
    state = _per_day_state("Ireland", "Ireland (Sep 14, 2025)")
    _store_legacy(state)
    result, apply_mock, _ = _per_day_run(tmp_path, state)
    assert result is not None
    assert apply_mock.call_count == 1


def test_the_per_day_gate_still_stops_a_real_burn_text_change(tmp_path):
    from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

    state = _per_day_state("Ireland", "Ireland (Sep 14, 2025)")
    _store_legacy(state)
    state["per_lf_ost_text"] = {1: "County Kerry, Ireland"}
    with pytest.raises(OstAuditPendingError):
        _per_day_run(tmp_path, state)


def test_the_per_day_stale_message_names_the_changed_burn_not_the_flag(tmp_path):
    from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

    state = _per_day_state("Ireland", "Ireland (Sep 14, 2025)")
    _store_legacy(state)
    state["per_lf_ost_text"] = {1: "County Kerry, Ireland"}
    with pytest.raises(OstAuditPendingError) as exc:
        _per_day_run(tmp_path, state)
    msg = str(exc.value)
    assert "changed since it was confirmed" in msg
    assert "re-run with --ost-audit-confirmed" not in msg


def test_the_per_day_unconfirmed_message_still_names_the_flag(tmp_path):
    from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

    with pytest.raises(OstAuditPendingError) as exc:
        _per_day_run(tmp_path, _per_day_state("Ireland", "Ireland (Sep 14, 2025)"))
    assert "re-run with --ost-audit-confirmed" in str(exc.value)


@pytest.mark.parametrize(
    "raw", ["2025-09-16", "2025-09-05T10:00:00", "", None, "garbage", "2025-13-40"]
)
def test_the_sidecars_drive_date_suffix_is_the_one_title_builder_dates_with(raw):
    """The sidecar (engine) may not import ``title_builder``, so it builds the suffix itself."""
    from gomoveshift.pipeline.ost_audit_sidecar import _drive_date_suffix
    from gomoveshift.pipeline.title_builder import _drive_dated

    assert _drive_date_suffix(raw) == (_drive_dated("", raw) or "")


# ── a pre-render-decision LF is checked as the title Phase D ships ──────────


def _decision_state(input_label, ost_text=_COUNTY):
    from gomoveshift.pipeline.title_decision import route_fingerprint

    state = make_state(
        per_lf_ost_text={1: ost_text},
        per_lf_recording_date={1: "2025-09-19"},
        source_dir="/footage/Ireland-Courtmacsherry",
    )
    record = {"segments": []}
    state.per_lf_route_record = {1: record}
    state.per_lf_pre_render_title_decisions = {
        1: {
            "input_label": input_label,
            "chosen_title": input_label,
            "ost_text": ost_text,
            "route_fingerprint": route_fingerprint(record),
        }
    }
    return state


def _live_under_an_unowned_video(monkeypatch, title_key):
    from gomoveshift.pipeline import longform_metadata as lm

    monkeypatch.setattr(lm, "live_catalog_title_ids", lambda: {title_key: "VIDX"})
    monkeypatch.setattr(lm, "ledger_video_projects", lambda: {})


def test_a_decision_lf_is_checked_as_the_dated_label_it_ships(tmp_path, monkeypatch):
    _live_under_an_unowned_video(monkeypatch, "county cork, ireland in 4k")
    dated = f"{_COUNTY} (Sep 19, 2025)"
    state = _decision_state(dated)
    _plans, gen = _split_run_with_state(
        tmp_path, state, {1: _COUNTY}, published={f"{_COUNTY} in 4K"}
    )
    assert _titles(gen) == [dated]


def test_a_decision_lf_whose_shipped_label_is_a_real_duplicate_still_refuses(tmp_path, monkeypatch):
    _live_under_an_unowned_video(monkeypatch, "county cork, ireland in 4k")
    state = _decision_state(_COUNTY)
    with pytest.raises(PhaseBCExitContractError):
        _split_run_with_state(tmp_path, state, {1: _COUNTY}, published={f"{_COUNTY} in 4K"})


def test_a_split_decision_lf_is_checked_without_its_part_suffix(tmp_path, monkeypatch):
    _live_under_an_unowned_video(monkeypatch, "county cork, ireland in 4k")
    label = f"{_COUNTY} (Sep 19, 2025) — Part 1 of 2"
    state = _decision_state(label)
    _plans, gen = _split_run_with_state(
        tmp_path, state, {1: _COUNTY}, {1: (1, 2)}, published={f"{_COUNTY} in 4K"}
    )
    assert _titles(gen) == [label]
