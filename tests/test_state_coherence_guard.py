"""Tests for the resume-time state coherence guard.

The guard at :func:`gomoveshift.pipeline.state_invariants.assert_resume_state_coherent`
runs ONCE at orchestrator startup, after state load (and after any auto-populate
hook), to catch the cross-field drift where ``phases_completed`` contained
``phase_bc_day_first`` while ``per_day_completed`` covered only 3 of 4
days. The orchestrator skipped Phase B+C on resume, then Phase D ran
on an incomplete batch.

These tests pin the contract so a future refactor can't silently relax
the guard back to a no-op (which is what the unguarded baseline already
behaves as — every test here has to ASSERT the raise to be meaningful).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from gomoveshift.pipeline.exceptions import StateCoherenceError
from gomoveshift.pipeline.state_invariants import assert_resume_state_coherent
from _state_helpers import make_state


# ── Happy path ───────────────────────────────────────────────────────


def test_phase_bc_complete_all_dates_covered_passes():
    """phases_completed has phase_bc_day_first; per_day_completed covers
    every date in episode_dates → no raise."""
    state = make_state(
        phases_completed=["phase_a", "phase_bc_day_first"],
        episode_dates=[
            "2026-02-20",
            "2026-02-21",
            "2026-02-22",
            "2026-02-24",
        ],
        per_day_completed={
            "2026-02-20": {"master_path": "/x/d1.mp4", "master_sha": "a"},
            "2026-02-21": {"master_path": "/x/d2.mp4", "master_sha": "b"},
            "2026-02-22": {"master_path": "/x/d3.mp4", "master_sha": "c"},
            "2026-02-24": {"master_path": "/x/d4.mp4", "master_sha": "d"},
        },
    )
    assert_resume_state_coherent(state)  # no raise


def test_phase_bc_complete_dedups_episode_dates():
    """Multiple episodes per day still resolve to one expected entry."""
    state = make_state(
        phases_completed=["phase_bc_day_first"],
        episode_dates=["2026-02-20", "2026-02-20", "2026-02-21"],
        per_day_completed={
            "2026-02-20": {"master_path": "/x/d1.mp4"},
            "2026-02-21": {"master_path": "/x/d2.mp4"},
        },
    )
    assert_resume_state_coherent(state)  # no raise


# ── Failure mode: drift ──────────────────────────────────────────────


def test_phase_bc_complete_missing_one_date_raises():
    """phases_completed claims phase_bc_day_first done; per_day_completed
    is short one date → StateCoherenceError naming the missing date."""
    state = make_state(
        phases_completed=["phase_a", "phase_bc_day_first"],
        episode_dates=[
            "2026-02-20",
            "2026-02-21",
            "2026-02-22",
            "2026-02-24",
        ],
        per_day_completed={
            "2026-02-20": {"master_path": "/x/d1.mp4"},
            "2026-02-21": {"master_path": "/x/d2.mp4"},
            "2026-02-22": {"master_path": "/x/d3.mp4"},
            # 2026-02-24 absent — the failure case.
        },
    )
    with pytest.raises(StateCoherenceError) as exc_info:
        assert_resume_state_coherent(state)
    msg = str(exc_info.value)
    assert "phase_bc_day_first" in msg, (
        "error message must name the spurious phase so operator knows what to remove"
    )
    assert "2026-02-24" in msg, "missing date must appear in the message"
    assert "remove" in msg.lower() or "wipe" in msg.lower(), (
        "operator hint (remove phase entry / wipe state) is mandatory"
    )


def test_phase_bc_complete_missing_multiple_dates_raises():
    """All missing dates surface in the message, not just the first."""
    state = make_state(
        phases_completed=["phase_bc_day_first"],
        episode_dates=["2026-02-20", "2026-02-21", "2026-02-22"],
        per_day_completed={
            "2026-02-20": {"master_path": "/x/d1.mp4"},
        },
    )
    with pytest.raises(StateCoherenceError) as exc_info:
        assert_resume_state_coherent(state)
    msg = str(exc_info.value)
    assert "2026-02-21" in msg
    assert "2026-02-22" in msg


def test_phase_bc_complete_falsy_per_day_entry_counts_as_missing():
    """A None / empty-dict entry under a date is NOT coverage — treat as missing.

    A prior bug wrote ``per_day_completed[date] = None`` when bookkeeping
    raced with a kill; that should fail the coherence check, not pass it.
    """
    state = make_state(
        phases_completed=["phase_bc_day_first"],
        episode_dates=["2026-02-20", "2026-02-21"],
        per_day_completed={
            "2026-02-20": {"master_path": "/x/d1.mp4"},
            "2026-02-21": None,  # falsy → counts as missing
        },
    )
    with pytest.raises(StateCoherenceError) as exc_info:
        assert_resume_state_coherent(state)
    assert "2026-02-21" in str(exc_info.value)


# ── No-op cases (must NOT false-positive) ────────────────────────────


def test_phase_bc_not_in_phases_completed_is_noop():
    """Pre-Phase-BC state with phase_a only → guard does nothing.

    Critical: the guard must not false-positive on legitimate early-state
    where Phase BC hasn't run yet. Otherwise every fresh run would crash
    at startup.
    """
    state = make_state(
        phases_completed=["phase_a"],
        episode_dates=["2026-02-20", "2026-02-21"],
        per_day_completed={},
    )
    assert_resume_state_coherent(state)  # no raise


def test_empty_phases_completed_is_noop():
    """Brand-new state (no phases done yet) → no-op."""
    state = make_state(
        phases_completed=[],
        episode_dates=["2026-02-20"],
        per_day_completed={},
    )
    assert_resume_state_coherent(state)  # no raise


def test_phases_completed_missing_is_noop():
    """State without phases_completed key (very early init) → no-op."""
    state: dict = {}
    assert_resume_state_coherent(state)  # no raise


def test_no_drive_plan_no_episode_dates_no_ingest_plan_is_noop(caplog):
    """phase_bc_day_first marked complete but no expected-date source
    available → DEBUG log + no raise.

    Contract: ``If drive_plan is missing entirely (e.g. very
    early state), skip the check (log DEBUG) — the check only applies
    when there's a plan to check against.``
    """
    import logging

    state = make_state(
        phases_completed=["phase_bc_day_first"],
        per_day_completed={},
    )
    with caplog.at_level(logging.DEBUG, logger="gomoveshift.pipeline.state_invariants"):
        assert_resume_state_coherent(state)  # no raise
    debug_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("coherence" in m.lower() or "expected-date" in m.lower() for m in debug_messages), (
        "DEBUG log must mention the skip + reason so operators tailing logs see it"
    )


# ── Alternate date sources ───────────────────────────────────────────


def test_dates_resolved_from_drive_plan_dict_with_dates_key():
    """A serialized drive_plan dict carrying a 'dates' key is honored."""
    state = {
        "phases_completed": ["phase_bc_day_first"],
        "drive_plan": {
            "dates": ["2026-02-20", "2026-02-21"],
            "regular_lfs": [],
        },
        "per_day_completed": {
            "2026-02-20": {"master_path": "/x/d1.mp4"},
            # 2026-02-21 missing → must raise
        },
    }
    with pytest.raises(StateCoherenceError) as exc_info:
        assert_resume_state_coherent(state)
    assert "2026-02-21" in str(exc_info.value)


def test_dates_resolved_from_ingest_plan_episode_dicts():
    """When episode_dates / drive_plan.dates are absent but ingest_plan
    has episodes with date strings, those drive the expected set."""

    class _FakeIngestPlan:
        def __init__(self, episodes):
            self.episodes = episodes

    class _FakeEpisode:
        def __init__(self, date):
            self.date = date
            self.creation_time = None

    state = make_state(
        phases_completed=["phase_bc_day_first"],
        ingest_plan=_FakeIngestPlan([_FakeEpisode("2026-02-20"), _FakeEpisode("2026-02-21")]),
        per_day_completed={
            "2026-02-20": {"master_path": "/x/d1.mp4"},
        },
    )
    with pytest.raises(StateCoherenceError) as exc_info:
        assert_resume_state_coherent(state)
    assert "2026-02-21" in str(exc_info.value)


def test_dates_resolved_from_ingest_plan_creation_time():
    """Ingest plan episodes without `.date` but with creation_time datetime
    still resolve via the same fallback group_episodes_by_day uses."""

    class _FakeIngestPlan:
        def __init__(self, episodes):
            self.episodes = episodes

    class _FakeEpisode:
        def __init__(self, ct):
            self.creation_time = ct
            self.date = None

    state = make_state(
        phases_completed=["phase_bc_day_first"],
        ingest_plan=_FakeIngestPlan(
            [
                _FakeEpisode(datetime(2026, 2, 20, 10, 0, 0)),
                _FakeEpisode(datetime(2026, 2, 21, 11, 0, 0)),
            ]
        ),
        per_day_completed={
            "2026-02-20": {"master_path": "/x/d1.mp4"},
            "2026-02-21": {"master_path": "/x/d2.mp4"},
        },
    )
    assert_resume_state_coherent(state)  # no raise — both covered


def test_episode_dates_takes_priority_over_ingest_plan():
    """When both are present, episode_dates wins (matches what Phase BC
    actually consumes for `per_day_completed` keys)."""

    class _FakeIngestPlan:
        def __init__(self, episodes):
            self.episodes = episodes

    class _FakeEpisode:
        def __init__(self, date):
            self.date = date
            self.creation_time = None

    state = make_state(
        phases_completed=["phase_bc_day_first"],
        episode_dates=["2026-02-20"],
        ingest_plan=_FakeIngestPlan([_FakeEpisode("2026-02-20"), _FakeEpisode("2099-12-31")]),
        per_day_completed={
            "2026-02-20": {"master_path": "/x/d1.mp4"},
        },
    )
    assert_resume_state_coherent(state)  # no raise — only 2026-02-20 expected


# ── Defensive type handling ──────────────────────────────────────────


def test_non_list_phases_completed_is_noop():
    """If phases_completed is the wrong type, defer to Rule 3 of the
    save-time invariant — coherence check should not double-report or
    crash on bad input shapes."""
    state = make_state(phases_completed="phase_bc_day_first")  # str, not list
    assert_resume_state_coherent(state)  # no raise


def test_non_dict_per_day_completed_is_noop():
    """Same defense for per_day_completed shape bugs — the type-check
    rule (Rule 1b) handles those at save time. Coherence is a higher-
    level concern that assumes shape-valid input."""
    state = make_state(
        phases_completed=["phase_bc_day_first"],
        episode_dates=["2026-02-20"],
        per_day_completed="wrong-type",
    )
    assert_resume_state_coherent(state)  # no raise


def test_state_coherence_error_subclasses_runtime_error():
    """StateCoherenceError is RuntimeError, NOT ValueError — distinguishes
    from StateInvariantError (ValueError) so callers trapping
    ``except ValueError`` don't silently swallow a coherence failure."""
    assert issubclass(StateCoherenceError, RuntimeError)
    assert not issubclass(StateCoherenceError, ValueError)
