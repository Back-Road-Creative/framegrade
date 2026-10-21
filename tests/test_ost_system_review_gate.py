"""The opt-in system review that may stand in for a person at the
OST burn gates.

Phase BC's route-claim check writes a system record into state for each day:
the burn text it checked, each LF's verdict and any refused claim. With
``--ost-system-review`` on THIS run, a record that agrees on every LF, names no
refused claim and covers the exact current burn text passes the gate -- at
both the per-day gate (``run_day_pipeline``) and the E2 front-load
(``lf_gate_or_raise``). Anything else stays pending, exactly as without the
opt-in. Nothing on disk can satisfy it: the record is read from state only.

Synthetic places only ("Harrow", "Westmark", "Longwood, SC" as a refused
label); no production trace, title or coordinate appears here.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_day_pipeline_ost_audit_gate import _run, _state_with_ost_text
from test_retained_gps_claims import _BandGeocoder, _episode, _lf

from gomoveshift.pipeline import gps_enrichment as ge
from gomoveshift.pipeline import ost_system_review as osr
from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError, lf_gate_or_raise
from gomoveshift.pipeline.route_claims import RouteClaimRefusedError, route_claim_outcome

D = "2025-09-14"


def _lf_gate(tmp_path, state):
    lf_gate_or_raise(tmp_path / "out", D, [1], state, location="County Kerry")


def _day_gate(tmp_path, state):
    _run(tmp_path, state, D)


GATES = pytest.mark.parametrize("gate", [_day_gate, _lf_gate], ids=["per-day", "front-load"])


def _state(*, opt_in=True, text=None):
    state = _state_with_ost_text(D)
    if text is not None:
        state["per_lf_ost_text"] = {1: text}
    if opt_in:
        state["ost_system_review_this_run"] = True
    return state


def _record(state, *, date=D, refused=(), texts=None):
    """The record Phase BC writes, through the real writer."""
    texts = texts or dict(state["per_lf_ost_text"])
    outcomes = {
        idx: route_claim_outcome(
            idx,
            label,
            None,
            chosen=None if idx in refused else label,
            reason="claims more than the retained GPS proves" if idx in refused else "",
        )
        for idx, label in texts.items()
    }
    osr.record_system_reviews(state, {date: dict(texts)}, outcomes)
    return state["ost_system_review"][date]


def _reseal(state, mutate, date=D):
    record = json.loads(json.dumps(state["ost_system_review"][date]))
    mutate(record)
    state["ost_system_review"][date] = osr.seal(record)


def _no_markers(tmp_path):
    return not list((tmp_path / "out").glob("*.confirmed"))


# --- B1 / M-c: the opt-in defaults OFF -----------------------------------


@GATES
def test_b1_without_opt_in_an_agreeing_record_stays_pending(tmp_path, gate):
    state = _state(opt_in=False)
    _record(state)
    with pytest.raises(OstAuditPendingError):
        gate(tmp_path, state)
    assert (tmp_path / "out" / f"ost_audit_{D}.json").exists()


# --- B2: the system record passes, with no person-made signal -------------


@GATES
def test_b2_agreeing_record_on_the_exact_text_passes(tmp_path, gate, caplog):
    state = _state()
    record = _record(state)
    with caplog.at_level(logging.INFO):
        gate(tmp_path, state)
    assert state["ost_audit_passed"][D] is True
    assert "ost_audit_confirmed_this_run" not in state
    assert _no_markers(tmp_path)
    passed = [r.getMessage() for r in caplog.records if "OST audit gate passed" in r.getMessage()]
    assert passed and "system review" in passed[-1] and record["ost_hash"] in passed[-1]
    copy = tmp_path / "out" / f"ost_system_review_{D}.json"
    assert json.loads(copy.read_text())["ost_hash"] == record["ost_hash"]
    assert str(copy) in caplog.text


# --- B3 / M-a: one changed character -------------------------------------


@GATES
def test_b3_one_changed_character_stays_pending(tmp_path, gate):
    state = _state()
    _record(state)
    text = state["per_lf_ost_text"][1]
    state["per_lf_ost_text"] = {1: text[:-1] + ("x" if text[-1] != "x" else "y")}
    with pytest.raises(OstAuditPendingError, match="system review"):
        gate(tmp_path, state)


# --- B4: a re-reviewed label still reopens a confirmed day ----------------


@GATES
def test_b4_label_change_after_a_pass_is_stale_even_when_re_reviewed(tmp_path, gate):
    state = _state()
    _record(state)
    gate(tmp_path, state)
    state["per_lf_ost_text"] = {1: "Harrow to Westmark via R9, County Clare"}
    _record(state)  # route claims re-ran and agree on the new text
    with pytest.raises(OstAuditPendingError):
        gate(tmp_path, state)


# --- Only an agree verdict counts -------------------------------


@GATES
@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["reviews"]["1"].__setitem__("verdict", "disagree"),
        lambda r: r["reviews"]["1"].pop("verdict"),
        lambda r: r.__setitem__("reviews", "not-a-map"),
    ],
    ids=["disagree", "absent", "malformed"],
)
def test_b5_a_verdict_that_is_not_agree_stays_pending(tmp_path, gate, mutate):
    state = _state()
    _record(state)
    _reseal(state, mutate)
    with pytest.raises(OstAuditPendingError, match="system review"):
        gate(tmp_path, state)


@GATES
def test_b5_a_record_that_is_not_a_record_stays_pending(tmp_path, gate):
    state = _state()
    state["ost_system_review"] = {D: "agree"}
    with pytest.raises(OstAuditPendingError, match="system review"):
        gate(tmp_path, state)


# --- B6 / M-b: a refused route claim names itself -------------------------


@GATES
def test_b6_a_refused_route_claim_stays_pending_and_is_named(tmp_path, gate):
    state = _state(text="Longwood, SC")
    _record(state, refused={1})
    with pytest.raises(OstAuditPendingError) as caught:
        gate(tmp_path, state)
    msg = str(caught.value)
    assert "refused" in msg and "LF 1" in msg and "Longwood, SC" in msg


@GATES
def test_b6_a_refused_claim_blocks_even_with_an_agree_verdict(tmp_path, gate):
    state = _state(text="Longwood, SC")
    _record(state, refused={1})
    _reseal(state, lambda r: r["reviews"]["1"].__setitem__("verdict", "agree"))
    with pytest.raises(OstAuditPendingError, match="refused"):
        gate(tmp_path, state)


# --- B7: another day or another LF set ------------------------------------


@GATES
def test_b7_a_record_for_another_date_stays_pending(tmp_path, gate):
    state = _state()
    _record(state, date="2025-09-15")
    state["ost_system_review"][D] = state["ost_system_review"].pop("2025-09-15")
    with pytest.raises(OstAuditPendingError, match="system review"):
        gate(tmp_path, state)


@GATES
def test_b7_a_record_for_another_lf_set_stays_pending(tmp_path, gate):
    state = _state()
    texts = {1: state["per_lf_ost_text"][1], 2: "Harrow to Westmark via R2, County Kerry"}
    _record(state, texts=texts)
    with pytest.raises(OstAuditPendingError, match="system review"):
        gate(tmp_path, state)


# --- B8: nothing a person can drop in satisfies it ------------------------


@GATES
def test_b8_a_record_file_on_disk_is_never_read(tmp_path, gate):
    donor = _state()
    record = _record(donor)
    state = _state()
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ost_system_review_{D}.json").write_text(json.dumps(record))
    (out / f"ost_audit_{D}.system-reviewed").write_text(record["ost_hash"])
    with pytest.raises(OstAuditPendingError):
        gate(tmp_path, state)


@GATES
@pytest.mark.parametrize("field", ["writer", "record_version"])
def test_b8_a_record_without_system_provenance_stays_pending(tmp_path, gate, field):
    state = _state()
    _record(state)
    _reseal(state, lambda r: r.pop(field))
    with pytest.raises(OstAuditPendingError, match="system review"):
        gate(tmp_path, state)


@GATES
def test_b8_an_edited_record_fails_its_seal(tmp_path, gate):
    state = _state(text="Longwood, SC")
    _record(state, refused={1})
    record = state["ost_system_review"][D]
    record["reviews"]["1"].update(verdict="agree", refused=False)  # no reseal
    with pytest.raises(OstAuditPendingError, match="system review"):
        gate(tmp_path, state)


# --- B9 / B10: resume -----------------------------------------------------


@GATES
def test_b9_resume_with_unchanged_text_passes_on_the_stored_hash(tmp_path, gate, caplog):
    state = _state()
    _record(state)
    gate(tmp_path, state)
    state.pop("ost_system_review_this_run")  # a resume without the opt-in
    with caplog.at_level(logging.INFO):
        gate(tmp_path, state)
    assert "matching hash" in caplog.text
    assert _no_markers(tmp_path)


@GATES
def test_b10_text_change_on_resume_stays_pending(tmp_path, gate):
    state = _state()
    _record(state)
    gate(tmp_path, state)
    state["per_lf_ost_text"] = {1: "Harrow to Westmark via R9, County Clare"}
    with pytest.raises(OstAuditPendingError):
        gate(tmp_path, state)


# --- B14 / mutant k: an LF with no retained GPS gets no system review ----


def _two_lf_day(*, lf2_measured):
    """LF1 measured ("Florence, SC"); LF2 measured only when ``lf2_measured``,
    else it has no retained GPS, so route claims hand the writer no outcome."""
    state = _state()
    texts = {1: "Florence, SC", 2: "Longwood, SC"}
    for idx, text in texts.items():
        state["per_lf_ost_text"][idx] = text
        state["per_lf_titles"][idx] = f"Driving {text} (Day {D})"
        state["per_lf_recording_date"][idx] = D
        state["per_lf_locations"][idx] = text
    measured = (1, 2) if lf2_measured else (1,)
    outcomes = {
        idx: route_claim_outcome(idx, texts[idx], None, chosen=texts[idx], reason="")
        for idx in measured
    }
    osr.record_system_reviews(state, {D: texts}, outcomes)
    return state


def _two_lf_day_gate(tmp_path, state):
    _run(tmp_path, state, D, lf_indices=(1, 2))


def _two_lf_lf_gate(tmp_path, state):
    lf_gate_or_raise(tmp_path / "out", D, [1, 2], state, location="County Kerry")


TWO_LF_GATES = pytest.mark.parametrize(
    "gate", [_two_lf_day_gate, _two_lf_lf_gate], ids=["per-day", "front-load"]
)


@TWO_LF_GATES
def test_b14_a_day_with_a_gps_less_lf_fails_the_system_signal(tmp_path, gate):
    state = _two_lf_day(lf2_measured=False)
    with pytest.raises(OstAuditPendingError):
        gate(tmp_path, state)
    assert (state.get("ost_audit_passed") or {}).get(D) is not True
    assert _no_markers(tmp_path)
    passes, reason = osr.system_review_passes(state, D, {1: "Florence, SC", 2: "Longwood, SC"})
    assert not passes and "record covers LFs [1], gate has [1, 2]" in reason
    assert state["ost_system_review"][D]["lf_indices"] == [1]


@TWO_LF_GATES
def test_b14_the_same_day_with_both_lfs_measured_passes(tmp_path, gate):
    """Control: the two-LF day passes once LF2 is measured too."""
    state = _two_lf_day(lf2_measured=True)
    gate(tmp_path, state)
    assert state["ost_audit_passed"][D] is True
    assert _no_markers(tmp_path)


# --- the writer, through the real pre-burn entry point --------------------


def _enrich_into(state, lfs, labels):
    with (
        patch.object(ge, "_build_per_lf_gpx_path_map", return_value={}),
        patch.object(ge, "_build_per_lf_gpx_episode_map", return_value={}),
        patch.object(ge, "_journey_endpoint_loss_by_lf", return_value={}),
        patch.object(ge, "_build_shared_geocoder", return_value=_BandGeocoder()),
        patch.object(ge, "_partition_lf_plans_by_date", return_value={"2026-01-02": lfs}),
        patch.object(ge, "_assert_per_lf_locations_resolvable", return_value=None),
        patch.object(ge, "derive_per_lf_locations", return_value={}),
        patch.object(ge, "_resolve_lf_destination", return_value=""),
        patch.object(ge, "canonical_lf_ost_text", side_effect=lambda lf, **k: labels[lf.index]),
        patch.object(ge, "_differentiate_and_revalidate_siblings", return_value=None),
    ):
        return ge._enrich_per_lf_gps(
            state=state,
            real_plan=SimpleNamespace(regular_lfs=lfs),
            day_batches=[],
            episode_files=[],
            per_day_locations={},
            drive_wide_location="",
        )


def _gps_state(overrides=None):
    return SimpleNamespace(
        gps_per_episode=_episode(),
        gpx_files=[],
        episode_dates=[],
        per_lf_overrides=overrides or {},
    )


def test_route_claims_write_an_agreeing_record_for_the_burned_text():
    state = _gps_state()
    # The label falls back to the county before the burn; the record covers
    # the fallback text, which is what burns.
    _enrich_into(state, [_lf(1, (0.0, 280.0), (320.0, 580.0))], {1: "Harrow, Westmark"})
    record = state.ost_system_review["2026-01-02"]
    assert record["writer"] == osr.WRITER
    assert record["lf_indices"] == [1]
    assert record["reviews"]["1"]["ost"] == "Ashby County, Westmark"
    assert record["reviews"]["1"]["verdict"] == "agree"
    assert osr.seal(dict(record)) == record


def test_route_claims_record_a_refusal_before_raising():
    state = _gps_state(overrides={1: "Harrow, Westmark"})
    lfs = [_lf(1, (0.0, 280.0), (620.0, 900.0))]
    with pytest.raises(RouteClaimRefusedError):
        _enrich_into(state, lfs, {1: "Harrow, Westmark"})
    review = state.ost_system_review["2026-01-02"]["reviews"]["1"]
    assert review["refused"] is True and review["verdict"] == "refused"
    assert review["ost"] == "Harrow, Westmark"


def test_record_carries_names_and_verdicts_never_a_fix():
    state = _gps_state()
    _enrich_into(state, [_lf(1, (0.0, 280.0))], {1: "Harrow, Westmark"})
    blob = json.dumps(state.ost_system_review)
    assert '"lat"' not in blob and '"lon"' not in blob and "9.5" not in blob


# --- the opt-in is a supported, per-run flag ------------------------------


def test_opt_in_is_a_cli_flag_defaulting_off():
    from gomoveshift.cli.parser import _build_parser

    parser = _build_parser()
    base = ["drive", "/fake/input"]
    assert not getattr(parser.parse_args(base), "ost_system_review", None)
    assert parser.parse_args([*base, "--ost-system-review"]).ost_system_review is True


def test_opt_in_never_survives_a_resume():
    from gomoveshift.pipeline.drive_state_model import PipelineState

    state = PipelineState()
    state.ost_system_review_this_run = True
    assert "ost_system_review_this_run" not in state.to_dict()


# ── A saved opt-in never survives a resume ──────────────────────────────────
# ``_UNSERIALIZED_FIELDS`` only stops writes; ``_try_resume`` still reads a
# legacy or edited ``pipeline_state.json``. The post-resume reset in ``run()``
# puts both per-run OST signals back to THIS invocation's flags.

_OPT_INS = ["ost_audit_confirmed_this_run", "ost_system_review_this_run"]


@pytest.mark.parametrize("field", _OPT_INS)
def test_b13_a_saved_opt_in_is_cleared_after_resume(tmp_path, field):
    from test_flag_registry import _fresh_state, _make_orch, _resume

    from gomoveshift.pipeline.drive_orchestrator import reset_ost_opt_ins_after_resume

    orch = _make_orch()
    state = _fresh_state(orch, tmp_path)
    _resume(orch, state, tmp_path, {field: True})
    reset_ost_opt_ins_after_resume(state, ost_audit_confirmed=None, ost_system_review=None)
    assert not state.get(field)


@GATES
@pytest.mark.parametrize("field", _OPT_INS)
def test_b13_a_resume_without_the_flag_refuses_at_the_gate(tmp_path, gate, field):
    from test_flag_registry import _fresh_state, _make_orch, _resume

    from gomoveshift.pipeline.drive_orchestrator import reset_ost_opt_ins_after_resume

    orch = _make_orch()
    resumed = _fresh_state(orch, tmp_path)
    _resume(orch, resumed, tmp_path, {field: True})
    reset_ost_opt_ins_after_resume(resumed, ost_audit_confirmed=None, ost_system_review=None)
    state = _state(opt_in=False)
    _record(state)
    for f in _OPT_INS:
        state[f] = resumed.get(f)
    with pytest.raises(OstAuditPendingError):
        gate(tmp_path, state)
    _no_markers(tmp_path)


def test_b13_the_flags_of_this_run_still_win_after_resume(tmp_path):
    from test_flag_registry import _fresh_state, _make_orch, _resume

    from gomoveshift.pipeline.drive_orchestrator import reset_ost_opt_ins_after_resume

    orch = _make_orch()
    state = _fresh_state(orch, tmp_path, ost_system_review=True)
    assert state.ost_system_review_this_run is True
    _resume(orch, state, tmp_path, {})
    reset_ost_opt_ins_after_resume(state, ost_audit_confirmed=True, ost_system_review=True)
    assert state.ost_system_review_this_run is True
    assert state.ost_audit_confirmed_this_run is True


def test_b13_run_resets_the_opt_ins_after_the_resume_merge():
    """``run()`` itself calls the reset, after ``_try_resume`` -- placed before
    it, the merge would re-import the saved value."""
    import ast
    import inspect
    import textwrap

    from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

    tree = ast.parse(textwrap.dedent(inspect.getsource(DrivePipelineOrchestrator.run)))
    lines = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            lines.setdefault(name, []).append(node.lineno)
    assert "reset_ost_opt_ins_after_resume" in lines, "run() never resets the OST opt-ins"
    assert min(lines["reset_ost_opt_ins_after_resume"]) > min(lines["_try_resume"])
