"""Tests for pipeline state invariants (the write-time gate on saved state).

`assert_state_invariants` guards `save_drive_state` so that state with wrong
types/shapes (the int/str-key class of bug, among others) can never
land on disk. Each rule has a fail-case test that pins the error message
contract — operators must be able to grep for the prefix — plus a pass-case
test that exercises a well-formed state value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gomoveshift.pipeline._drive_state import save_drive_state
from gomoveshift.pipeline.exceptions import StateInvariantError
from gomoveshift.pipeline.state_invariants import (
    STATE_INVARIANT_ERROR_PREFIX,
    assert_state_invariants,
)
from _state_helpers import make_state


# ── Rule 1 — int-keyed dicts (per_lf_locations, per_lf_titles, shorts_per_lf,
#    social_posts_per_lf, metadata_per_lf, upload_manifests, youtube_uploads) ──
#
# per_day_completed is excluded from the int-key check — it's keyed by
# date-strings.


def test_int_key_dict_str_keyed_raises():
    """per_lf_locations with string keys raises with a grep-able prefix."""
    state = make_state(per_lf_locations={"1": "Beaufort County, SC"})
    with pytest.raises(StateInvariantError) as exc_info:
        assert_state_invariants(state)
    msg = str(exc_info.value)
    assert msg.startswith(STATE_INVARIANT_ERROR_PREFIX)
    assert "per_lf_locations" in msg
    assert "non-int key" in msg


def test_int_key_dict_int_keyed_passes():
    """per_lf_locations with int keys is accepted."""
    state = make_state(per_lf_locations={1: "Beaufort County, SC", 2: "Dare County, NC"})
    assert_state_invariants(state)  # no raise


def test_int_key_rule_covers_all_listed_dicts():
    """Every dict in the Layer-1 int-key list is checked, not just one."""
    int_keyed = (
        "per_lf_locations",
        "per_lf_titles",
        "shorts_per_lf",
        "social_posts_per_lf",
        "metadata_per_lf",
        "upload_manifests",
        "youtube_uploads",
    )
    for key in int_keyed:
        state = {key: {"0": "bad"}}
        with pytest.raises(StateInvariantError) as exc_info:
            assert_state_invariants(state)
        assert key in str(exc_info.value), f"{key} not called out in error"


def test_per_day_completed_str_keys_pass():
    """per_day_completed is date-string-keyed — must NOT raise under int-key rule."""
    state = make_state(
        per_day_completed={
            "2026-02-20": {"master_path": "/x/master.mp4", "master_sha": "abc"},
            "2026-02-21": {"master_path": "/y/master.mp4", "master_sha": "def"},
        },
    )
    assert_state_invariants(state)  # no raise


def test_per_day_completed_int_keys_raise():
    """per_day_completed with int keys is wrong — dates are strings."""
    state = make_state(per_day_completed={1: {"master_path": "/x"}})
    with pytest.raises(StateInvariantError) as exc_info:
        assert_state_invariants(state)
    msg = str(exc_info.value)
    assert msg.startswith(STATE_INVARIANT_ERROR_PREFIX)
    assert "per_day_completed" in msg


# ── Rule 2 — publish_day ∈ {mon..sun} AND (when set) publish_day's int
#    ∈ LONG_FORM_DAYS from schedule_constants ──


def test_publish_day_unknown_raises():
    """publish_day='xyz' is rejected with the canonical error prefix."""
    state = make_state(publish_day="xyz")
    with pytest.raises(StateInvariantError) as exc_info:
        assert_state_invariants(state)
    msg = str(exc_info.value)
    assert msg.startswith(STATE_INVARIANT_ERROR_PREFIX)
    assert "publish_day" in msg
    assert "xyz" in msg


def test_publish_day_valid_day_not_in_long_form_days_raises(monkeypatch):
    """publish_day='tue' with LONG_FORM_DAYS={Wed=2} is rejected."""
    from gomoveshift.pipeline import schedule_constants

    monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", {2})
    state = make_state(publish_day="tue")
    with pytest.raises(StateInvariantError) as exc_info:
        assert_state_invariants(state)
    msg = str(exc_info.value)
    assert msg.startswith(STATE_INVARIANT_ERROR_PREFIX)
    assert "publish_day" in msg
    assert "LONG_FORM_DAYS" in msg


def test_publish_day_in_long_form_days_passes(monkeypatch):
    """publish_day='wed' with LONG_FORM_DAYS={2} passes."""
    from gomoveshift.pipeline import schedule_constants

    monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", {2})
    state = make_state(publish_day="wed")
    assert_state_invariants(state)  # no raise


def test_publish_day_empty_passes():
    """publish_day unset/empty passes — orchestrator init can have blank."""
    state = make_state(publish_day="")
    assert_state_invariants(state)  # no raise


# ── Rule 2b — lf_publish_date weekday ∈ LONG_FORM_DAYS (and == publish_day) ──


def _lf_days(monkeypatch, days):
    from gomoveshift.pipeline import schedule_constants

    monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", frozenset(days))


def test_lf_publish_date_off_publish_day_raises(monkeypatch):
    _lf_days(monkeypatch, {4})
    state = make_state(publish_day="fri", lf_publish_date="2026-10-06")
    with pytest.raises(StateInvariantError) as exc_info:
        assert_state_invariants(state)
    msg = str(exc_info.value)
    assert msg.startswith(STATE_INVARIANT_ERROR_PREFIX)
    for needle in ("2026-10-06", "weekday=1", "--lf-publish-date", "gomoveshift schedule"):
        assert needle in msg


def test_lf_publish_date_on_publish_day_passes(monkeypatch):
    _lf_days(monkeypatch, {4})
    assert_state_invariants(make_state(publish_day="fri", lf_publish_date="2026-10-09"))


def test_lf_publish_date_off_long_form_days_without_publish_day_raises(monkeypatch):
    _lf_days(monkeypatch, {4})
    with pytest.raises(StateInvariantError, match="2026-10-06"):
        assert_state_invariants(make_state(publish_day=None, lf_publish_date="2026-10-06"))


def test_lf_publish_date_malformed_raises(monkeypatch):
    _lf_days(monkeypatch, {4})
    with pytest.raises(StateInvariantError, match="10/06/2026"):
        assert_state_invariants(make_state(publish_day="fri", lf_publish_date="10/06/2026"))


@pytest.mark.parametrize("value", ["2026-10-9", "2026-1-09", " 2026-10-09", "2026-10-09 "])
def test_lf_publish_date_non_canonical_form_raises(monkeypatch, value):
    """strptime takes an unpadded date; the downstream isoformat parsers do not."""
    _lf_days(monkeypatch, {4})
    with pytest.raises(StateInvariantError, match="not a YYYY-MM-DD date"):
        assert_state_invariants(make_state(publish_day="fri", lf_publish_date=value))


@pytest.mark.parametrize("value", [None, ""])
def test_lf_publish_date_unset_passes(monkeypatch, value):
    _lf_days(monkeypatch, {4})
    assert_state_invariants(make_state(publish_day="fri", lf_publish_date=value))


def test_save_drive_state_refuses_off_day_anchor(tmp_path: Path, monkeypatch):
    _lf_days(monkeypatch, {4})
    state = make_state(publish_day="fri", lf_publish_date="2026-10-06")
    path = tmp_path / "s.json"
    with pytest.raises(StateInvariantError):
        save_drive_state(path, state)
    assert not path.exists()


# ── Rule 3 — phases_completed ⊆ known phases ──


def test_phases_completed_unknown_phase_raises():
    """phases_completed with an unknown phase name raises."""
    state = make_state(phases_completed=["phase_a", "phase_mystery"])
    with pytest.raises(StateInvariantError) as exc_info:
        assert_state_invariants(state)
    msg = str(exc_info.value)
    assert msg.startswith(STATE_INVARIANT_ERROR_PREFIX)
    assert "phases_completed" in msg
    assert "phase_mystery" in msg


def test_phases_completed_known_phases_pass():
    """All four known phases pass (with downstream cross-field state satisfied)."""
    state = make_state(
        phases_completed=[
            "phase_a",
            "phase_b_preview",
            "phase_bc_day_first",
            "phase_d",
        ],
        regular_lf_paths=["/x/lf_01.mp4"],
        shorts_per_lf={1: ["s1.mp4"]},
    )
    assert_state_invariants(state)  # no raise


def test_phases_completed_empty_list_passes():
    """Empty phases_completed (fresh run) passes."""
    state = make_state(phases_completed=[])
    assert_state_invariants(state)


def test_phases_completed_none_passes():
    """phases_completed=None (default) passes — unset means not-yet-populated."""
    state = make_state(phases_completed=None)
    assert_state_invariants(state)


# ── Rule 4 — Shape: episode_files, episode_dates, file_durations,
#    file_boundaries same length (only when all present) ──


def test_shape_mismatch_episode_fields_raises():
    """Four episode_* arrays disagree on length → raise."""
    state = make_state(
        episode_files=["/a.mp4", "/b.mp4"],
        episode_dates=["2026-02-20"],
        file_durations=[60.0, 120.0],
        file_boundaries=[(0.0, 60.0), (60.0, 180.0)],
    )
    with pytest.raises(StateInvariantError) as exc_info:
        assert_state_invariants(state)
    msg = str(exc_info.value)
    assert msg.startswith(STATE_INVARIANT_ERROR_PREFIX)
    assert "episode_" in msg
    assert "length" in msg


def test_shape_match_passes():
    """All four episode_* arrays same length → pass."""
    state = make_state(
        episode_files=["/a.mp4", "/b.mp4"],
        episode_dates=["2026-02-20", "2026-02-21"],
        file_durations=[60.0, 120.0],
        file_boundaries=[(0.0, 60.0), (60.0, 180.0)],
    )
    assert_state_invariants(state)  # no raise


def test_shape_partial_skipped():
    """If not all four arrays are present, shape check is skipped."""
    state = make_state(
        episode_files=["/a.mp4", "/b.mp4"],
    )
    assert_state_invariants(state)  # no raise


# ── Rule 5 — Cross-field: phase_bc in phases_completed → regular_lf_paths
#    non-empty AND shorts_per_lf populated (relaxed: shorts_per_lf may be {}
#    if every LF legitimately rendered zero shorts). ──


def test_cross_field_phase_bc_complete_zero_lfs_passes():
    """phase_bc_day_first complete with regular_lf_paths=[] must pass.

    Zero-qualifying-footage drives
    legitimately produce ``regular_lf_paths=[]`` via the early-exit path in
    ``PhaseBCDayFirst.run`` (it sets ``state.regular_lf_paths = []``
    when there is no usable ingest plan). The prior Rule 5 fired on every
    save, making those drives unresumable. The new rule requires the field
    to be a ``list`` (not ``None``), not that it be non-empty.
    """
    state = make_state(
        phases_completed=["phase_a", "phase_bc_day_first"],
        regular_lf_paths=[],
        shorts_per_lf={},
    )
    assert_state_invariants(state)  # no raise


def test_cross_field_phase_bc_complete_none_lfs_raises():
    """phase_bc_day_first complete but regular_lf_paths is None → raise.

    Rule 5 still rejects ``None`` because that indicates the field was
    never populated — the phase either didn't reach setdefault() or was
    reset incorrectly. Only a ``list`` (possibly empty) is valid.
    """
    state = make_state(
        phases_completed=["phase_bc_day_first"],
        regular_lf_paths=None,
        shorts_per_lf={1: ["s1.mp4"]},
    )
    with pytest.raises(StateInvariantError) as exc_info:
        assert_state_invariants(state)
    assert "regular_lf_paths" in str(exc_info.value)


def test_cross_field_phase_bc_complete_non_list_lfs_raises():
    """phase_bc_day_first complete with regular_lf_paths=<non-list> → raise.

    A string, dict, or other non-list type is a type bug — the field must
    be a list even when empty.
    """
    state = make_state(
        phases_completed=["phase_bc_day_first"],
        regular_lf_paths="oops-not-a-list",
        shorts_per_lf={},
    )
    with pytest.raises(StateInvariantError) as exc_info:
        assert_state_invariants(state)
    assert "regular_lf_paths" in str(exc_info.value)


def test_cross_field_phase_bc_save_zero_lf_does_not_raise(tmp_path):
    """Integration: save_drive_state accepts a zero-LF checkpoint for a
    completed phase_bc_day_first.

    Concrete bite: a
    zero-qualifying-footage drive would raise StateInvariantError on every
    save_drive_state call mid-run. With the fix, a valid zero-LF state
    lands on disk and the pipeline can resume from checkpoint.
    """
    state_file = tmp_path / "pipeline_state.json"
    zero_lf_state = make_state(
        phases_completed=["phase_a", "phase_bc_day_first"],
        regular_lf_paths=[],
        shorts_per_lf={},
        per_day_locations={},
    )
    save_drive_state(state_file, zero_lf_state)
    assert state_file.exists(), (
        "save_drive_state must write zero-LF checkpoint to disk; a valid "
        "zero-qualifying-footage drive is a legitimate outcome of the "
        "phase_bc_day_first early-exit path."
    )
    loaded = json.loads(state_file.read_text())
    assert loaded["regular_lf_paths"] == []
    assert "phase_bc_day_first" in loaded["phases_completed"]


def test_cross_field_phase_bc_complete_with_lfs_passes():
    """phase_bc complete, LFs + shorts present → pass."""
    state = make_state(
        phases_completed=["phase_bc_day_first"],
        regular_lf_paths=["/x/lf_01.mp4", "/x/lf_02.mp4"],
        shorts_per_lf={1: ["s1.mp4"], 2: ["s2.mp4"]},
    )
    assert_state_invariants(state)  # no raise


def test_cross_field_phase_bc_with_lfs_empty_shorts_passes():
    """phase_bc complete with LFs and shorts_per_lf={} → pass (relaxed rule).

    The strict rule says shorts_per_lf must be populated. In practice it's
    legal for every LF in a short-light day to produce zero renderable shorts
    (e.g. all segments below `min_shorts_duration`). We relax to: an empty
    dict is acceptable as long as regular_lf_paths is non-empty — operators
    inspect logs / per-LF output to confirm the zero-shorts reality.
    """
    state = make_state(
        phases_completed=["phase_bc_day_first"],
        regular_lf_paths=["/x/lf_01.mp4"],
        shorts_per_lf={},
    )
    assert_state_invariants(state)  # no raise


def test_cross_field_phase_bc_not_complete_no_checks():
    """Without phase_bc_day_first in phases_completed, cross-field rule is skipped."""
    state = make_state(
        phases_completed=["phase_a"],
        regular_lf_paths=None,
    )
    assert_state_invariants(state)  # no raise


# ── Integration — save_drive_state calls assert_state_invariants before
#    touching disk. Invalid state → raise + no file written. ──


def test_save_drive_state_refuses_invalid_state(tmp_path: Path):
    """save_drive_state raises before writing when invariants fail."""
    state_file = tmp_path / "pipeline_state.json"
    bad_state = make_state(per_lf_locations={"1": "Bad County, XY"})  # str key → fail
    with pytest.raises(StateInvariantError) as exc_info:
        save_drive_state(state_file, bad_state)
    assert str(exc_info.value).startswith(STATE_INVARIANT_ERROR_PREFIX)
    # File MUST NOT exist — the last-valid state on disk is preserved.
    assert not state_file.exists(), "save_drive_state wrote the file despite failing invariants"
    # Nor should there be a lingering .tmp sibling from the atomic-write path.
    assert list(tmp_path.glob("*.tmp")) == [], "atomic-write tempfile leaked when invariants failed"


def test_save_drive_state_accepts_valid_state(tmp_path: Path):
    """Well-formed state writes through without raising."""
    state_file = tmp_path / "pipeline_state.json"
    good_state = make_state(
        per_lf_locations={1: "Beaufort County, SC"},
        phases_completed=["phase_a"],
        publish_day="wed",
    )
    save_drive_state(state_file, good_state)
    assert state_file.exists()
    with open(state_file) as f:
        loaded = json.load(f)
    assert loaded["per_lf_locations"] == {"1": "Beaufort County, SC"}


def test_state_invariant_error_subclasses_value_error():
    """StateInvariantError is a ValueError subclass for easy catching."""
    assert issubclass(StateInvariantError, ValueError)


# ─────────────────────────────────────────────────────────────────────────
# --skip-invariants break-glass.
#
# A latent invariant bug could make every save_drive_state call raise
# mid-run, killing an 8-hour encode with no recovery short of --fresh.
# The break-glass flag lets an operator bypass the gate to rescue prior
# work while the underlying bug is fixed.
#
# Semantics:
#   * Default is OFF — never set in normal production.
#   * When set, save_drive_state logs WARNING and skips assert_state_invariants.
#   * The .tmp + os.replace atomic-write path is preserved either way.
# ─────────────────────────────────────────────────────────────────────────


def test_save_drive_state_skip_invariants_bypasses_gate(tmp_path: Path, caplog):
    """With skip_invariants=True, save_drive_state writes state that would
    otherwise fail the invariant check. A WARNING is logged so operators
    see the bypass in the log file."""
    import logging

    state_file = tmp_path / "pipeline_state.json"
    # This state would normally fail Rule 1 (per_lf_locations has str keys).
    bad_state = make_state(per_lf_locations={"1": "Bad County, XY"})

    with caplog.at_level(logging.WARNING):
        save_drive_state(state_file, bad_state, skip_invariants=True)

    assert state_file.exists(), (
        "skip_invariants=True must bypass the gate and write state to disk. "
        "Without this break-glass, a latent invariant bug kills "
        "an 8-hour encode with no recovery short of --fresh."
    )
    # A WARNING log line mentions the bypass so operators see it in the
    # pipeline log even when no human is watching stdout.
    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "skip_invariants" in messages or "invariants" in messages.lower(), (
        "skip_invariants bypass must emit a WARNING log line"
    )


def test_save_drive_state_skip_invariants_false_still_checks(tmp_path: Path):
    """skip_invariants=False (the default) preserves the existing gate —
    bad state still raises."""
    state_file = tmp_path / "pipeline_state.json"
    bad_state = make_state(per_lf_locations={"1": "Bad County, XY"})

    with pytest.raises(StateInvariantError):
        save_drive_state(state_file, bad_state, skip_invariants=False)

    assert not state_file.exists()


def test_save_drive_state_default_kwargs_check_invariants(tmp_path: Path):
    """Default behaviour (no kwarg passed) keeps the check — backward compat."""
    state_file = tmp_path / "pipeline_state.json"
    bad_state = make_state(per_lf_locations={"1": "Bad County, XY"})

    with pytest.raises(StateInvariantError):
        save_drive_state(state_file, bad_state)  # no skip_invariants kwarg


def test_drive_cli_exposes_skip_invariants_flag():
    """``gomoveshift drive --skip-invariants`` must be a recognized CLI flag.

    This proves the argparse wiring exists end-to-end; without it, operators
    cannot reach the break-glass path at all.
    """
    from gomoveshift.cli.parser import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["drive", "/tmp/src", "--skip-invariants"])
    assert getattr(args, "skip_invariants", False) is True, (
        "--skip-invariants must be a recognized drive subcommand flag (break-glass)."
    )


def test_drive_cli_skip_invariants_default_false():
    """The flag defaults to False — never set without an explicit CLI arg."""
    from gomoveshift.cli.parser import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["drive", "/tmp/src"])
    assert getattr(args, "skip_invariants", False) is False


# ─────────────────────────────────────────────────────────────────────────
# per_lf_gpx_path with explicit None values must pass
# the write-time int-key invariant. None is a legitimate value for GPS-less
# LFs under the robust-by-construction contract.
# ─────────────────────────────────────────────────────────────────────────


def test_per_lf_gpx_path_with_none_values_passes_int_key_check():
    """per_lf_gpx_path={1: None, 2: '/path/to.gpx'} is a valid shape — pass."""
    state = make_state(per_lf_gpx_path={1: None, 2: "/abs/day_2026-02-21/track.gpx"})
    assert_state_invariants(state)  # no raise


def test_per_lf_gpx_path_str_key_still_raises():
    """Str-keyed per_lf_gpx_path still trips Rule 1 (canonical-key invariant)."""
    state = make_state(per_lf_gpx_path={"1": None})
    with pytest.raises(StateInvariantError) as exc_info:
        assert_state_invariants(state)
    assert "per_lf_gpx_path" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────
# A corrupt/unreadable CONFIRMED ost_audit sidecar must NOT be skipped.
# "Can't verify the confirmed hash" is treated as drift requiring
# re-confirmation, never a silent vacuous pass at resume. The legitimately
# absent (never-confirmed) sidecar is still a clean pass.
# ─────────────────────────────────────────────────────────────────────────

_OST_AUDIT_DATE = "2026-02-21"


def _make_ost_audit_payload(date: str = _OST_AUDIT_DATE) -> dict:
    """Minimal audit payload matching what compute_ost_audit_hash hashes."""
    return {
        "date": date,
        "lfs": [{"lf_index": 1, "ost_text": "Beaufort via US-21, SC"}],
    }


def _write_confirmed_ost_audit(tmp_path: Path, payload: dict):
    """Write a confirmed sidecar to disk and return a state dict whose
    ost_audit_passed/ost_audit_hash mark it as confirmed."""
    from gomoveshift.pipeline.ost_audit_sidecar import compute_ost_audit_hash

    date = payload["date"]
    sidecar = tmp_path / f"ost_audit_{date}.json"
    sidecar.write_text(json.dumps(payload))
    state = {
        "output_base": str(tmp_path),
        "ost_audit_passed": {date: True},
        "ost_audit_hash": {date: compute_ost_audit_hash(payload)},
    }
    return sidecar, state


def test_confirmed_ost_audit_sidecar_intact_passes(tmp_path):
    """A confirmed sidecar that still hashes to the stored value passes."""
    from gomoveshift.pipeline.state_invariants import _check_ost_audit_hash_drift

    _, state = _write_confirmed_ost_audit(tmp_path, _make_ost_audit_payload())
    _check_ost_audit_hash_drift(state)  # no raise


def test_confirmed_ost_audit_sidecar_corrupt_raises(tmp_path):
    """A confirmed sidecar that is unparseable JSON cannot be hash-verified
    and MUST raise — not silently skip the re-confirmation gate."""
    from gomoveshift.pipeline.state_invariants import _check_ost_audit_hash_drift

    sidecar, state = _write_confirmed_ost_audit(tmp_path, _make_ost_audit_payload())
    sidecar.write_text("{ this is not valid json")

    with pytest.raises(StateInvariantError) as exc_info:
        _check_ost_audit_hash_drift(state)
    msg = str(exc_info.value)
    assert msg.startswith(STATE_INVARIANT_ERROR_PREFIX)
    assert _OST_AUDIT_DATE in msg


def test_absent_ost_audit_sidecar_still_passes(tmp_path):
    """A confirmed marker with no sidecar on disk is the never-written /
    legacy-confirmation case — unchanged clean pass (not corruption)."""
    from gomoveshift.pipeline.state_invariants import _check_ost_audit_hash_drift

    state = {
        "output_base": str(tmp_path),
        "ost_audit_passed": {_OST_AUDIT_DATE: True},
        "ost_audit_hash": {_OST_AUDIT_DATE: "deadbeef"},
    }
    _check_ost_audit_hash_drift(state)  # no raise
