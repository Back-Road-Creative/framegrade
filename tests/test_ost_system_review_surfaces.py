"""The opt-in system review at the Shorts and superlong burn gates.

The day pipeline measures every Short's text over its own clip's retained GPS
and every superlong part's title and lower-thirds over that part's retained
ranges. ``ost_system_review.record_surface_reviews`` keeps what each check
decided, per surface and day, in a sealed record. With ``--ost-system-review``
on THIS run, a record that covers exactly the gate's entries, agrees on each,
names no refused claim, holds no inherited (``lf_label``) Short text and hashes
the exact current burn passes ``shorts_gate_or_raise`` /
``superlong_gate_or_raise``. Anything else stays pending, as without the
opt-in. No ``.confirmed`` marker is read or written.

Synthetic places only ("Harrow", "Westmark", "Calder" and "Corvane" are
invented); no production trace, title, label or coordinate appears here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from _state_helpers import make_state
from test_short_route_claims import _fixes
from test_short_route_claims import _run as _run_shorts
from test_superlong_route_claims import _points
from test_superlong_route_claims import _run as _run_superlong

from gomoveshift.pipeline import ost_system_review as osr
from gomoveshift.pipeline.ost_audit_sidecar import (
    OstAuditPendingError,
    shorts_gate_or_raise,
    superlong_gate_or_raise,
)

D = "2026-01-01"


def _short(rank, text="Harrow", *, start=0.0, end=38.0, source="clip_gps"):
    return {
        "rank": rank,
        "parent_lf_index": 1,
        "ost_text": text,
        "start_sec": start,
        "end_sec": end,
        "regions": ["Westmark"],
        "median_speed_mps": 20.0,
        "place_source": source,
    }


def _part(idx, text="Harrow", *, count=2, lower_thirds=(("Harrow", 0.0),)):
    return {
        "part_index": idx,
        "part_count": count,
        "ost_text": text,
        "lower_thirds": [{"label": lbl, "time_sec": t} for lbl, t in lower_thirds],
    }


SURFACES = {
    "shorts": (shorts_gate_or_raise, lambda: [_short(1), _short(2, start=40.0, end=78.0)]),
    "superlong": (superlong_gate_or_raise, lambda: [_part(1), _part(2)]),
}
SURFACE = pytest.mark.parametrize("surface", sorted(SURFACES))
_ID = {"shorts": "rank", "superlong": "part_index"}


def _ids(surface, entries):
    return [e[_ID[surface]] for e in entries]


def _state(opt_in=True):
    state = make_state()
    if opt_in:
        state["ost_system_review_this_run"] = True
    return state


def _record(state, surface, entries, *, refused=(), missing=()):
    """The record the measuring loop writes, through the real writer."""
    outcomes = {
        i: {
            "refused": i in refused,
            "reason": "claims more than the retained GPS proves" if i in refused else "",
            "label_requested": e["ost_text"],
            "ost_text": e["ost_text"],
        }
        for i, e in zip(_ids(surface, entries), entries)
        if i not in missing
    }
    osr.record_surface_reviews(state, surface, D, entries, outcomes)
    return (state.get(osr.SURFACE_FIELDS[surface]) or {}).get(D)


def _gate(tmp_path, surface, entries, state):
    gate, _ = SURFACES[surface]
    return gate(tmp_path / "out", D, entries, state)


def _pending(tmp_path, surface, entries, state, match=None):
    with pytest.raises(OstAuditPendingError, match=match):
        _gate(tmp_path, surface, entries, state)


def _no_markers(tmp_path):
    return not list((tmp_path / "out").glob("*.confirmed"))


@pytest.fixture(autouse=True)
def _out(tmp_path):
    (tmp_path / "out").mkdir()


# --- the system record passes, with no person-made signal ---


@SURFACE
def test_g2b1_an_agreeing_record_on_the_exact_burn_passes(tmp_path, surface, caplog):
    entries = SURFACES[surface][1]()
    state = _state()
    record = _record(state, surface, entries)
    with caplog.at_level(logging.INFO):
        _gate(tmp_path, surface, entries, state)
    assert not state.get("ost_audit_confirmed_this_run")
    assert _no_markers(tmp_path)
    copy = tmp_path / "out" / f"{surface}_ost_system_review_{D}.json"
    assert json.loads(copy.read_text())["text_hash"] == record["text_hash"]
    assert f"system review record {record['text_hash']}" in caplog.text
    assert str(copy) in caplog.text


# --- the opt-in defaults OFF ----------------------------------------


@SURFACE
def test_g2b6_without_opt_in_an_agreeing_record_stays_pending(tmp_path, surface):
    entries = SURFACES[surface][1]()
    state = _state(opt_in=False)
    _record(state, surface, entries)
    assert osr.surface_review_passes(state, surface, D, entries) == (False, "")
    _pending(tmp_path, surface, entries, state)
    assert (tmp_path / "out" / f"{surface}_ost_audit_{D}.json").exists()


# --- the record is bound to its surface -----------------------------


@SURFACE
def test_g2b2_an_lf_record_with_the_same_text_does_not_pass(tmp_path, surface):
    entries = SURFACES[surface][1]()
    state = _state()
    osr.record_system_reviews(
        state,
        {D: {1: entries[0]["ost_text"], 2: entries[1]["ost_text"]}},
        {
            i: {"refused": False, "reason": "", "label_requested": e["ost_text"]}
            for i, e in ((1, entries[0]), (2, entries[1]))
        },
    )
    assert state["ost_system_review"][D]
    _pending(tmp_path, surface, entries, state, match="no .* system review record")
    state[osr.SURFACE_FIELDS[surface]] = {D: state["ost_system_review"][D]}
    _pending(tmp_path, surface, entries, state, match="system provenance")


def test_g2b2_one_surface_record_never_passes_the_other(tmp_path):
    shorts, parts = SURFACES["shorts"][1](), SURFACES["superlong"][1]()
    state = _state()
    _record(state, "superlong", parts)
    state["shorts_ost_system_review"] = state["superlong_ost_system_review"]
    _pending(tmp_path, "shorts", shorts, state, match="no shorts system provenance")
    state = _state()
    _record(state, "shorts", shorts)
    state["superlong_ost_system_review"] = state["shorts_ost_system_review"]
    _pending(tmp_path, "superlong", parts, state, match="no superlong system provenance")


_CHANGES = {
    "shorts": {
        "text": lambda e: e[1].update(ost_text="Harrow!"),
        "start_sec": lambda e: e[1].update(start_sec=41.0),
        "end_sec": lambda e: e[1].update(end_sec=79.0),
    },
    "superlong": {
        "text": lambda e: e[1].update(ost_text="Harrow!"),
        "part_count": lambda e: [p.update(part_count=3) for p in e],
        "lower_third_label": lambda e: e[1]["lower_thirds"][0].update(label="Westmark"),
        "lower_third_time": lambda e: e[1]["lower_thirds"][0].update(time_sec=5.0),
    },
}


@pytest.mark.parametrize(
    "surface,change", [(s, c) for s, cs in sorted(_CHANGES.items()) for c in sorted(cs)]
)
def test_g2b2_a_changed_identity_or_burn_text_stays_pending(tmp_path, surface, change):
    entries = SURFACES[surface][1]()
    state = _state()
    _record(state, surface, entries)
    _CHANGES[surface][change](entries)
    _pending(tmp_path, surface, entries, state, match="differs from the text")


# --- exact coverage -------------------------------------------------


@SURFACE
def test_g2b3_one_entry_more_in_the_gate_is_refused(tmp_path, surface):
    entries = SURFACES[surface][1]()
    state = _state()
    _record(state, surface, entries[:1])
    _pending(tmp_path, surface, entries, state, match=r"record covers \w+ \[1\], gate has \[1, 2\]")


@SURFACE
def test_g2b3_one_entry_fewer_in_the_gate_is_refused(tmp_path, surface):
    entries = SURFACES[surface][1]()
    state = _state()
    _record(state, surface, entries)
    _pending(
        tmp_path, surface, entries[:1], state, match=r"record covers \w+ \[1, 2\], gate has \[1\]"
    )


# --- every entry measured on its own GPS ----------------------------


@SURFACE
def test_g2b4_an_entry_with_no_measurement_gets_no_review(tmp_path, surface):
    entries = SURFACES[surface][1]()
    state = _state()
    record = _record(state, surface, entries, missing=(2,))
    assert record["entry_ids"] == [1]
    _pending(tmp_path, surface, entries, state)


@SURFACE
def test_g2b4_text_other_than_the_measured_text_gets_no_review(tmp_path, surface):
    entries = SURFACES[surface][1]()
    state = _state()
    outcomes = {
        i: {"refused": False, "reason": "", "label_requested": "x", "ost_text": "Harrow"}
        for i in (1, 2)
    }
    entries[1]["ost_text"] = "Westmark"
    osr.record_surface_reviews(state, surface, D, entries, outcomes)
    assert state[osr.SURFACE_FIELDS[surface]][D]["entry_ids"] == [1]
    _pending(tmp_path, surface, entries, state)


@SURFACE
def test_g2b4_a_day_with_no_measured_entry_has_no_record(surface):
    entries = SURFACES[surface][1]()
    state = _state()
    assert _record(state, surface, entries, missing=(1, 2)) is None


def test_g2b4_an_inherited_short_does_not_pass_and_is_named(tmp_path):
    entries = [
        _short(1, "Calder, Westmark"),
        _short(2, "Corvane, Westmark", start=40.0, end=78.0, source="lf_label"),
    ]
    state = _state()
    _record(state, "shorts", entries)
    with pytest.raises(OstAuditPendingError) as exc:
        _gate(tmp_path, "shorts", entries, state)
    msg = str(exc.value)
    assert "short 2 'Corvane, Westmark' text is inherited (lf_label)" in msg
    assert "short 1" not in msg
    assert _no_markers(tmp_path)


# --- a refused claim refuses the surface ----------------------------


@SURFACE
def test_g2b5_a_refused_entry_stays_pending_with_its_reason(tmp_path, surface):
    entries = SURFACES[surface][1]()
    state = _state()
    _record(state, surface, entries, refused=(2,))
    _pending(tmp_path, surface, entries, state, match="refused: claims more than the retained GPS")


@SURFACE
def test_g2b5_an_edited_record_fails_its_seal(tmp_path, surface):
    entries = SURFACES[surface][1]()
    state = _state()
    _record(state, surface, entries, refused=(2,))
    state[osr.SURFACE_FIELDS[surface]][D]["reviews"]["2"].update(refused=False, verdict="agree")
    _pending(tmp_path, surface, entries, state, match="seal does not match")


# --- a saved opt-in never survives a resume -------------------------


@SURFACE
def test_g2b7_a_resume_without_the_flag_is_pending(tmp_path, surface):
    from test_flag_registry import _fresh_state, _make_orch, _resume

    from gomoveshift.pipeline.drive_orchestrator import reset_ost_opt_ins_after_resume

    orch = _make_orch()
    resumed = _fresh_state(orch, tmp_path)
    _resume(orch, resumed, tmp_path, {"ost_system_review_this_run": True})
    reset_ost_opt_ins_after_resume(resumed, ost_audit_confirmed=None, ost_system_review=None)
    entries = SURFACES[surface][1]()
    state = _state(opt_in=False)
    _record(state, surface, entries)
    state["ost_system_review_this_run"] = resumed.get("ost_system_review_this_run")
    _pending(tmp_path, surface, entries, state)


# --- what a system pass persists ------------------------------------


def test_g2b8_shorts_resume_passes_on_the_stored_hash_and_reopens_on_change(tmp_path):
    entries = SURFACES["shorts"][1]()
    state = _state()
    _record(state, "shorts", entries)
    _gate(tmp_path, "shorts", entries, state)
    assert state["shorts_ost_audit_passed"][D] is True
    state["ost_system_review_this_run"] = None  # a later run without the flag
    _gate(tmp_path, "shorts", SURFACES["shorts"][1](), state)
    changed = SURFACES["shorts"][1]()
    changed[1]["ost_text"] = "Westmark"
    _pending(tmp_path, "shorts", changed, state)


def test_g2b8_superlong_keeps_no_pass_so_a_resume_without_the_flag_parks(tmp_path):
    # Stricter than Shorts: the superlong gate has never persisted a pass
    # (its resume fingerprint is D8c's), so the system pass adds none.
    entries = SURFACES["superlong"][1]()
    state = _state()
    _record(state, "superlong", entries)
    assert _gate(tmp_path, "superlong", entries, state) == entries
    state["ost_system_review_this_run"] = None
    _pending(tmp_path, "superlong", SURFACES["superlong"][1](), state)


# --- through the measuring loops, no marker, no confirmation --------


def test_g2b10_shorts_pass_on_the_record_their_own_loop_wrote(monkeypatch, tmp_path, caplog):
    state = _state()
    with caplog.at_level(logging.INFO):
        entries, rendered = _run_shorts(
            monkeypatch,
            tmp_path,
            _fixes(80.0),
            ranks=((1, 0.0, 38.0), (2, 40.0, 78.0)),
            state=state,
        )
    record = state["shorts_ost_system_review"][D]
    assert record["entry_ids"] == [1, 2]
    assert len(rendered) == 2
    assert f"system review record {record['text_hash']}" in caplog.text
    assert not list(tmp_path.glob("**/*.confirmed"))


def test_g2b10_a_gps_less_day_has_no_shorts_record_and_parks(monkeypatch, tmp_path):
    # No fixes at all: the Shorts inherit text (lf_label) and nothing measures them.
    state = _state()
    with pytest.raises(OstAuditPendingError, match="no shorts system review record"):
        _run_shorts(monkeypatch, tmp_path, [], state=state)
    assert not (state.get("shorts_ost_system_review") or {}).get(D)


def test_g2b10_without_the_flag_the_same_shorts_park(monkeypatch, tmp_path):
    with pytest.raises(OstAuditPendingError):
        _run_shorts(monkeypatch, tmp_path, _fixes(), state=_state(opt_in=False))


def test_g2b10_superlong_parts_pass_on_the_record_their_own_loop_wrote(tmp_path, caplog):
    state = _state()
    with caplog.at_level(logging.INFO):
        run = _run_superlong(tmp_path, _points(), state=state)
    assert run.error is None, run.error
    record = state["superlong_ost_system_review"][D]
    assert record["entry_ids"] == [1, 2]
    assert [b["location"] for b in run.burns] == ["Harrow", "Harrow"]
    assert f"system review record {record['text_hash']}" in caplog.text
    assert not list(tmp_path.glob("**/*.confirmed"))


def test_g2b10_without_the_flag_the_same_parts_park(tmp_path):
    run = _run_superlong(tmp_path, _points(), state=_state(opt_in=False))
    assert isinstance(run.error, OstAuditPendingError)
    assert run.burns == []


def test_g2b10_the_short_route_fn_is_the_real_measurement(monkeypatch, tmp_path):
    # A Short with GPS but no route check (short_route_fn=None) is unmeasured.
    import gomoveshift.pipeline.day_pipeline as dp

    state = _state()
    rendered = []
    monkeypatch.setattr(dp, "_render_short_ffmpeg", lambda **kw: rendered.append(kw) or True)
    monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda b: [])
    monkeypatch.setattr(dp, "assign_shorts_to_lfs_by_source_window", lambda **k: {1: range(1)})
    with pytest.raises(OstAuditPendingError):
        dp._render_day_shorts(
            day_dir=tmp_path,
            date=D,
            day_batch=SimpleNamespace(episodes=[]),
            short_ranges=[SimpleNamespace(rank=1, start_sec=0.0, end_sec=38.0)],
            allow_short_count=True,
            gps_points_by_date={D: _fixes()},
            lf_plans=[SimpleNamespace(index=1)],
            per_lf_ost_text={1: "Harrow"},
            per_lf_locations={1: "Ashby County"},
            location="Drive Wide",
            short_ost_text_fn=lambda pts, sr: "Harrow",
            short_route_fn=None,
            master_source_for_extract=Path("/dev/null"),
            state=state,
        )
    assert rendered == []
    assert not (state.get("shorts_ost_system_review") or {}).get(D)


def _full_day(tmp_path, state, *, short_text="Harrow"):
    """``run_day_pipeline`` with LF + Shorts on synthetic GPS, heavy stages mocked."""
    from unittest.mock import patch

    from test_day_pipeline import TestRunDayPipeline as _StageMocks
    from test_day_pipeline import _make_day_batch, _make_lf_plan, _make_scores, _make_seg
    from test_day_pipeline_shorts_ost_gate import DATE, _touch_concat, _touch_montage
    from test_retained_gps_claims import _BandGeocoder

    from gomoveshift.pipeline import day_pipeline
    from gomoveshift.pipeline.route_claims import make_short_route_fn

    output_base = tmp_path / "day-out"
    _, mocks = _StageMocks()._patch_all_stages(tmp_path)
    osr.record_system_reviews(
        state,
        {DATE: {1: "Harrow"}},
        {1: {"refused": False, "reason": "", "label_requested": "Harrow"}},
    )
    with (
        mocks["encode_day_master"],
        mocks["apply_ost"],
        mocks["_render_short_ffmpeg"] as render_short,
        patch.object(day_pipeline, "extract_and_concat_segments", side_effect=_touch_concat),
        patch.object(day_pipeline, "prepend_montage", side_effect=_touch_montage),
    ):
        result = day_pipeline.run_day_pipeline(
            day_batch=_make_day_batch(DATE, 3, per_ep_duration=1200.0),
            day_scores=_make_scores(3600, score=0.6),
            output_dir=output_base,
            location="Drive Wide",
            config=None,
            lf_plans=[_make_lf_plan(1, [_make_seg(0, 100, 200)], [])],
            per_lf_ost_text={1: "Harrow"},
            gps_points_by_date={DATE: _fixes(3600.0)},
            short_ost_text_fn=lambda pts, sr: short_text,
            short_route_fn=make_short_route_fn(_BandGeocoder()),
            state=state,
        )
    return DATE, result, render_short, output_base


def test_g2b10_a_full_day_passes_lf_and_shorts_on_system_records(tmp_path, caplog):
    state = _state()
    with caplog.at_level(logging.INFO):
        date, result, render_short, out = _full_day(tmp_path, state)
    assert result is not None and render_short.call_count == len(result.short_paths) > 0
    record = state["shorts_ost_system_review"][date]
    assert f"system review record {state['ost_system_review'][date]['ost_hash']}" in caplog.text
    assert f"system review record {record['text_hash']}" in caplog.text
    assert not state.get("ost_audit_confirmed_this_run")
    assert not list(out.glob("*.confirmed"))


def test_g2b10_an_unsupported_short_label_is_broadened_and_the_record_says_why(tmp_path):
    # "Corvane, Westmark" names a place no retained fix resolves to: the route check
    # broadens it, the burn carries the broader label, and the sealed record
    # binds the burned text with the reason, never the requested label.
    state = _state()
    date, result, render_short, out = _full_day(tmp_path, state, short_text="Corvane, Westmark")
    burned = [c.kwargs.get("location") for c in render_short.call_args_list]
    assert burned and set(burned) == {"Ashby County, Westmark"}
    reviews = state["shorts_ost_system_review"][date]["reviews"].values()
    for r in reviews:
        assert r["ost_text"] == "Ashby County, Westmark"
        assert r["label_requested"] == "Corvane, Westmark"
        assert "no retained GPS fix resolves to" in r["reason"]
        assert r["refused"] is False
