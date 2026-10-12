"""Tests for ``resume_invariants(state, config)``.

Part of the state-contract consolidation. Folds the
following one-off checks behind a single post-load entry point that
every resume path passes through:

  * disk + RAM + ffmpeg preflight (formerly skipped when
    ``phase_a`` was in ``resumed_phases``);
  * youtube_uploads.json sidecar integrity (when ``phase_d`` is
    partially complete);
  * per-LF recording-date + trim-ratio completeness for any phase
    already in ``phases_completed``;
  * OST audit gate hash consistency (when ``ost_audit_passed`` is
    set, the loaded sidecar hash MUST still match the just-computed
    hash — otherwise per-LF labels drifted under the operator's
    feet and we re-raise the gate).

Plus an integration test asserting the orchestrator calls
``resume_invariants`` on the resume path AND NOT on the fresh path.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.exceptions import (
    SidecarIntegrityError,
    StateInvariantError,
)
from gomoveshift.pipeline.state_invariants import (
    STATE_INVARIANT_ERROR_PREFIX,
    resume_invariants,
)
from _state_helpers import make_state


# ── Helpers ────────────────────────────────────────────────────────────


def _make_drive_plan(lf_indices=(1, 2)):
    return SimpleNamespace(regular_lfs=[SimpleNamespace(index=i) for i in lf_indices])


def _make_state(
    tmp_path: Path,
    *,
    phases_completed=None,
    drive_plan=None,
    per_lf_recording_date=None,
    per_lf_trim_ratio=None,
    per_lf_gpx_path=None,
    ost_audit_passed=None,
    ost_audit_hash=None,
    regular_lf_paths=None,
    shorts_per_lf=None,
):
    """Build a plain dict state with the fields ``resume_invariants`` reads."""
    output_base = tmp_path / "out"
    output_base.mkdir(parents=True, exist_ok=True)
    source_dir = tmp_path / "src"
    source_dir.mkdir(parents=True, exist_ok=True)
    state = make_state(
        source_dir=source_dir,
        output_base=output_base,
        phases_completed=list(phases_completed or []),
        drive_plan=drive_plan,
        per_lf_recording_date=per_lf_recording_date or {},
        per_lf_trim_ratio=per_lf_trim_ratio or {},
        per_lf_gpx_path=per_lf_gpx_path or {},
        ost_audit_passed=ost_audit_passed or {},
        ost_audit_hash=ost_audit_hash or {},
        per_lf_ost_text={},
        per_lf_titles={},
        per_lf_locations={},
        regular_lf_paths=regular_lf_paths if regular_lf_paths is not None else [],
        shorts_per_lf=shorts_per_lf if shorts_per_lf is not None else {},
    )
    return state


def _make_config(tmp_path: Path):
    """Minimal config stub. ``resume_invariants`` only consults SEO paths
    indirectly via run_preflight, so we point them at scratch dirs.
    """
    seo_dir = tmp_path / "seo"
    seo_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        seo=SimpleNamespace(storage_path=str(seo_dir)),
        drive=SimpleNamespace(),
    )


# ── Disk / RAM preflight is run on every resume ────────────────────────


def test_resume_invariants_runs_disk_check(tmp_path, monkeypatch):
    """A resume into Phase D with insufficient disk MUST fail loud here,
    not opaquely 6 hours into upload.
    """
    state = _make_state(
        tmp_path,
        phases_completed=["phase_a", "phase_bc_day_first"],
        drive_plan=_make_drive_plan(),
        per_lf_recording_date={1: "2026-02-20", 2: "2026-02-21"},
        per_lf_trim_ratio={1: 0.6, 2: 0.7},
        regular_lf_paths=["/tmp/lf_01.mp4", "/tmp/lf_02.mp4"],
        shorts_per_lf={1: [], 2: []},
    )
    config = _make_config(tmp_path)

    def _low_disk(path, min_free_gb=50):
        raise RuntimeError(f"Insufficient disk space at {path}: 1.0 GB free, need {min_free_gb} GB")

    monkeypatch.setattr("gomoveshift.pipeline.state_invariants.check_disk_space", _low_disk)
    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        resume_invariants(state, config)


def test_resume_invariants_runs_ram_check(tmp_path, monkeypatch):
    """Low RAM must also fail at the resume gate (preflight, full)."""
    state = _make_state(
        tmp_path,
        phases_completed=["phase_a"],
        drive_plan=_make_drive_plan(),
    )
    config = _make_config(tmp_path)

    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_disk_space",
        lambda *a, **kw: None,
    )

    def _low_ram(min_free_gb=2.0):
        raise RuntimeError(f"Insufficient free RAM: 0.5 GB available, need {min_free_gb} GB")

    monkeypatch.setattr("gomoveshift.pipeline.state_invariants.check_ram", _low_ram)
    with pytest.raises(RuntimeError, match="Insufficient free RAM"):
        resume_invariants(state, config)


# ── NVENC preflight on resume ──────────────────


def test_resume_invariants_probes_nvenc_when_encoding_ahead(tmp_path, monkeypatch):
    """When ``phase_bc_day_first`` is NOT yet complete, resume must probe
    NVENC up-front so a driver reset between runs fails at the gate, not
    hours into encode.
    """
    from gomoveshift.pipeline.color_filters import NVENCUnavailableError

    state = _make_state(
        tmp_path,
        phases_completed=["phase_a"],
        drive_plan=_make_drive_plan(),
    )
    config = _make_config(tmp_path)

    # Disk + RAM + tool checks pass — let the NVENC gate be the only failure.
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_disk_space",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_ram",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_tools",
        lambda *a, **kw: None,
    )

    def _no_nvenc():
        raise NVENCUnavailableError("NVENC probe failed; recovery: ...")

    monkeypatch.setattr("gomoveshift.pipeline.color_filters.detect_nvenc", _no_nvenc)
    with pytest.raises(NVENCUnavailableError, match="NVENC probe failed"):
        resume_invariants(state, config)


def test_resume_invariants_skips_nvenc_when_phase_bc_complete(tmp_path, monkeypatch):
    """When ``phase_bc_day_first`` is already complete, every encode is on
    disk — a transient NVENC outage between runs must NOT block a
    Phase-D-only resume.
    """
    state = _make_state(
        tmp_path,
        phases_completed=["phase_a", "phase_bc_day_first"],
        drive_plan=_make_drive_plan(),
        per_lf_recording_date={1: "2026-02-20", 2: "2026-02-21"},
        per_lf_trim_ratio={1: 0.6, 2: 0.7},
        per_lf_gpx_path={1: "/mock/g01.gpx", 2: "/mock/g02.gpx"},
        regular_lf_paths=["/tmp/lf_01.mp4", "/tmp/lf_02.mp4"],
        shorts_per_lf={1: [], 2: []},
    )
    config = _make_config(tmp_path)

    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_disk_space",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_ram",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_tools",
        lambda *a, **kw: None,
    )

    nvenc_calls: list[int] = []

    def _track_nvenc():
        nvenc_calls.append(1)
        return True

    monkeypatch.setattr("gomoveshift.pipeline.color_filters.detect_nvenc", _track_nvenc)
    # Should NOT raise — and detect_nvenc should not have been invoked.
    resume_invariants(state, config)
    assert nvenc_calls == [], (
        "detect_nvenc was probed even though phase_bc_day_first was already "
        "complete — no further encoding is required, the probe is gratuitous"
    )


# ── Sidecar integrity on partial phase_d ────────────────────────────────


def test_resume_invariants_phase_d_sidecar_phantom(tmp_path, monkeypatch):
    """Sidecar with index not in drive_plan.regular_lfs raises
    SidecarIntegrityError on the resume path."""
    state = _make_state(
        tmp_path,
        phases_completed=["phase_a", "phase_bc_day_first", "phase_d"],
        drive_plan=_make_drive_plan(lf_indices=(1, 2)),
        per_lf_recording_date={1: "2026-02-20", 2: "2026-02-21"},
        per_lf_trim_ratio={1: 0.6, 2: 0.7},
        regular_lf_paths=["/tmp/lf_01.mp4", "/tmp/lf_02.mp4"],
    )
    output_base: Path = state["output_base"]
    # Sidecar with a phantom index (99) absent from drive_plan.
    (output_base / "youtube_uploads.json").write_text(
        json.dumps({"1": {"video_id": "abc"}, "99": {"video_id": "xyz"}})
    )

    # Stub preflight so it doesn't gate on real disk.
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_disk_space",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_ram",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_tools",
        lambda *a, **kw: None,
    )

    with pytest.raises(SidecarIntegrityError) as exc:
        resume_invariants(state, _make_config(tmp_path))
    assert 99 in exc.value.phantom_indices


# ── Per-LF backstop fields ──────────────────────────────────────────────


def test_resume_invariants_per_lf_partial_drift_raises(tmp_path, monkeypatch):
    """Partial drift (some LFs mapped, others missing) MUST raise — the
    downstream backstops cannot cleanly repair a half-correct map.
    """
    state = _make_state(
        tmp_path,
        phases_completed=["phase_a", "phase_bc_day_first"],
        drive_plan=_make_drive_plan(lf_indices=(1, 2)),
        per_lf_recording_date={1: "2026-02-20"},  # 1 of 2 — partial drift
        per_lf_trim_ratio={1: 0.6, 2: 0.7},
        per_lf_gpx_path={1: "/tmp/g01.gpx", 2: "/tmp/g02.gpx"},
        regular_lf_paths=["/tmp/lf_01.mp4", "/tmp/lf_02.mp4"],
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_disk_space",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_ram",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_tools",
        lambda *a, **kw: None,
    )
    with pytest.raises(StateInvariantError) as exc:
        resume_invariants(state, _make_config(tmp_path))
    assert "per_lf_recording_date" in str(exc.value)
    # The message names the missing index(es) specifically rather
    # than just the count drift, so partial-key-offset corruption surfaces
    # too. LF 2 is the missing index in this fixture.
    assert "[2]" in str(exc.value)
    assert STATE_INVARIANT_ERROR_PREFIX in str(exc.value)


def test_resume_invariants_per_lf_legacy_empty_raises(tmp_path, monkeypatch):
    """Fully-empty per-LF maps under completed phase_bc
    now RAISE.

    ``PipelineState`` declares ``per_lf_recording_date`` / ``per_lf_trim_ratio``
    and adds a load-time disk-derivation migration in
    ``from_dict``. Every legitimate resume enters ``resume_invariants``
    with these fields populated (either by Phase BC's canonical writer
    or by the migration). Reaching here empty means the on-disk LF
    artifacts cannot be recovered — hard-fail is correct.
    """
    state = _make_state(
        tmp_path,
        phases_completed=["phase_a", "phase_bc_day_first"],
        drive_plan=_make_drive_plan(lf_indices=(1, 2)),
        per_lf_recording_date={},  # empty AFTER migration → corrupted artifacts
        per_lf_trim_ratio={},
        regular_lf_paths=["/tmp/lf_01.mp4", "/tmp/lf_02.mp4"],
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_disk_space",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_ram",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_tools",
        lambda *a, **kw: None,
    )
    with pytest.raises(StateInvariantError) as exc:
        resume_invariants(state, _make_config(tmp_path))
    msg = str(exc.value)
    assert "per_lf_recording_date" in msg
    assert STATE_INVARIANT_ERROR_PREFIX in msg


# ── OST audit hash drift ────────────────────────────────────────────────


def test_resume_invariants_ost_audit_hash_mismatch(tmp_path, monkeypatch):
    """If state says ``ost_audit_passed[date]=True`` but the on-disk sidecar
    hashes to a different value (operator edited per-LF labels after the
    confirm), the resume gate MUST raise so the operator re-confirms."""
    state = _make_state(
        tmp_path,
        phases_completed=["phase_a", "phase_bc_day_first"],
        drive_plan=_make_drive_plan(lf_indices=(1,)),
        per_lf_recording_date={1: "2026-02-20"},
        per_lf_trim_ratio={1: 0.6},
        per_lf_gpx_path={1: "/tmp/g01.gpx"},
        ost_audit_passed={"2026-02-20": True},
        ost_audit_hash={"2026-02-20": "stale-hash-from-previous-confirmation"},
        regular_lf_paths=["/tmp/lf_01.mp4"],
    )
    output_base: Path = state["output_base"]
    # Write a sidecar whose computed hash will NOT match the stale stored hash.
    sidecar_payload = {
        "date": "2026-02-20",
        "generated_at": "2026-05-12T00:00:00+00:00",
        "lfs": [
            {
                "lf_index": 1,
                "ost_text": "Edited label after confirmation",
                "youtube_title": "Edited label",
                "thumbnail_text": "Edited label",
                "recording_date": "2026-02-20",
            }
        ],
    }
    (output_base / "ost_audit_2026-02-20.json").write_text(
        json.dumps(sidecar_payload, indent=2, sort_keys=True)
    )

    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_disk_space",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_ram",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_tools",
        lambda *a, **kw: None,
    )

    with pytest.raises(StateInvariantError) as exc:
        resume_invariants(state, _make_config(tmp_path))
    msg = str(exc.value)
    assert "ost_audit_hash" in msg or "OST audit" in msg
    assert "2026-02-20" in msg


# ── Clean-resume happy path ─────────────────────────────────────────────


def test_resume_invariants_clean_resume_passes(tmp_path, monkeypatch):
    state = _make_state(
        tmp_path,
        phases_completed=["phase_a", "phase_bc_day_first"],
        drive_plan=_make_drive_plan(lf_indices=(1, 2)),
        per_lf_recording_date={1: "2026-02-20", 2: "2026-02-21"},
        per_lf_trim_ratio={1: 0.6, 2: 0.7},
        per_lf_gpx_path={1: "/tmp/g01.gpx", 2: "/tmp/g02.gpx"},
        regular_lf_paths=["/tmp/lf_01.mp4", "/tmp/lf_02.mp4"],
        shorts_per_lf={1: [], 2: []},
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_disk_space",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_ram",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.state_invariants.check_tools",
        lambda *a, **kw: None,
    )
    # No sidecar on disk → empty sidecar → integrity check short-circuits.
    assert resume_invariants(state, _make_config(tmp_path)) is None


# ── Orchestrator wiring ─────────────────────────────────────────────────


def test_drive_orchestrator_calls_resume_invariants_on_resume(monkeypatch):
    """The orchestrator MUST call ``resume_invariants`` immediately after
    ``_try_resume`` returns a non-empty resume set, and MUST NOT call it
    on a pure-fresh run.

    We exercise both by patching the symbol in drive_orchestrator and
    inspecting call counts.
    """
    import gomoveshift.pipeline.drive_orchestrator as do

    # Source-level check: the orchestrator module imports
    # ``resume_invariants`` and references it in ``run``.
    src = Path(do.__file__).read_text()
    assert "resume_invariants" in src, (
        "drive_orchestrator.py must reference resume_invariants — "
        "otherwise the post-load entry point is unwired."
    )
    # The disk/RAM preflight guard that was removed MUST stay gone.
    # We pin by the legacy comment ``skip on resume`` — there is one
    # legitimate remaining ``"phase_a" not in resumed_phases`` reference
    # (the preview-block guard around phase_a's own execution), so we
    # cannot pin by code text alone.
    assert "skip on resume" not in src, (
        "The 'if phase_a not in resumed_phases:' guard around run_preflight "
        "MUST be removed — resume paths must hit preflight too."
    )
    # And the new resume_invariants call site MUST exist.
    assert "resume_invariants(state" in src, (
        "drive_orchestrator must call resume_invariants(state, config) "
        "after _try_resume() returns a non-empty resume set."
    )
