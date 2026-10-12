"""Regression tests for dataclass re-hydration in ``PipelineState.from_dict``.

Covers the plan fields ``early_plan``,
``drive_plan``, and ``ingest_plan``, which are serialized via their own
``to_dict()`` / ``dataclasses.asdict()`` path but were historically
reloaded as raw ``dict`` objects. Downstream callers reach for attribute
access (``drive_plan.regular_lfs``) and hit ``AttributeError`` on any
resume path that skips Phase A/BC and restores the field straight from
the checkpoint.

These tests pin the expected behavior: after ``from_dict(to_dict(state))``
each of the three fields must be an instance of its declared dataclass
with at least one attribute round-tripping correctly.
"""

from pathlib import Path

from gomoveshift.models import DrivePlan, RegularLFPlan
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.ingest.gopro_parser import IngestPlan


def _make_drive_plan() -> DrivePlan:
    """Minimal valid DrivePlan — one empty LF, all numeric fields set."""
    lf = RegularLFPlan(
        index=1,
        segments=[],
        montage_peaks=[],
        shorts_target=3,
        region="ireland",
        total_duration=600.0,
        undersized=False,
    )
    return DrivePlan(
        source_files=[Path("/tmp/src/ep_000.mp4")],
        processed_files=[Path("/tmp/out/ep_000_FINAL.mp4")],
        regular_lfs=[lf],
        score_threshold=0.42,
        total_source_duration=1800.0,
        total_highlight_duration=600.0,
        best_moments_filtered=False,
    )


def _make_ingest_plan() -> IngestPlan:
    """Minimal valid IngestPlan — source_dir only, empty nested lists."""
    return IngestPlan(source_dir=Path("/tmp/src"))


def test_drive_plan_rehydrates_to_dataclass():
    """``drive_plan`` must load back as a DrivePlan, not a plain dict."""
    state = PipelineState(
        source_dir=Path("/tmp/src"),
        output_base=Path("/tmp/out"),
        drive_plan=_make_drive_plan(),
    )
    restored = PipelineState.from_dict(state.to_dict())
    assert isinstance(restored.drive_plan, DrivePlan), (
        f"expected DrivePlan, got {type(restored.drive_plan).__name__}"
    )
    # At least one attribute round-trips — regression pin for the
    # original crash site (downstream code reads ``drive_plan.regular_lfs``).
    assert len(restored.drive_plan.regular_lfs) == 1
    assert restored.drive_plan.regular_lfs[0].index == 1
    assert restored.drive_plan.score_threshold == 0.42


def test_ingest_plan_rehydrates_to_dataclass():
    """``ingest_plan`` must load back as an IngestPlan, not a plain dict."""
    state = PipelineState(
        source_dir=Path("/tmp/src"),
        output_base=Path("/tmp/out"),
        ingest_plan=_make_ingest_plan(),
    )
    restored = PipelineState.from_dict(state.to_dict())
    assert isinstance(restored.ingest_plan, IngestPlan), (
        f"expected IngestPlan, got {type(restored.ingest_plan).__name__}"
    )
    assert restored.ingest_plan.source_dir == Path("/tmp/src")


def test_rehydration_handles_none_fields():
    """None-valued dataclass fields stay None across the round trip."""
    state = PipelineState(
        source_dir=Path("/tmp/src"),
        output_base=Path("/tmp/out"),
    )
    restored = PipelineState.from_dict(state.to_dict())
    assert restored.drive_plan is None
    assert restored.ingest_plan is None


def test_rehydration_idempotent_for_already_hydrated_instance():
    """Passing an already-hydrated dataclass through ``from_dict`` is a no-op.

    ``PipelineState.from_dict`` is sometimes fed a dict where values are
    already live instances (e.g. ``update()`` from another ``PipelineState``
    serialized via ``to_dict`` but one side skipped serialization). The
    re-hydration branch must tolerate this.
    """
    plan = _make_drive_plan()
    state_dict = {"drive_plan": plan, "region": "ireland"}
    restored = PipelineState.from_dict(state_dict)
    assert restored.drive_plan is plan


# ── per_lf_gpx_path durable resume invariant ──
#
# Legacy state files (phase_bc_day_first complete but no per_lf_gpx_path
# key) MUST migrate at ``from_dict`` time so the resume-time invariant
# ``state_invariants._check_per_lf_completeness`` accepts the loaded
# state. The migration walks each LF's ``day_<DATE>/`` directory and
# probes for a sibling ``.gpx`` file. GPS-less LFs are filled with
# explicit ``None``.


def _make_legacy_state_dict(tmp_path: Path, *, lf_indices=(1,), make_gpx_for=()):
    """Construct a legacy phase_bc-complete state dict + on-disk LF layout.

    Returns ``(state_dict, regular_lf_paths)``. ``make_gpx_for`` is a
    set of LF indices that should have a sibling ``.gpx`` in their
    day dir; the rest are GPS-less.
    """
    regular_lf_paths: list[str] = []
    skeleton_lfs = []
    for lf_idx in lf_indices:
        day_date = f"2026-02-{20 + lf_idx:02d}"
        day_dir = tmp_path / f"day_{day_date}"
        regular_lfs_dir = day_dir / "regular_lfs"
        regular_lfs_dir.mkdir(parents=True, exist_ok=True)
        lf_path = regular_lfs_dir / f"lf_{lf_idx:02d}.mp4"
        lf_path.write_bytes(b"")  # zero-byte placeholder; migration only probes path layout
        if lf_idx in make_gpx_for:
            (day_dir / f"gopro_{lf_idx:02d}.gpx").write_text("<gpx></gpx>")
        regular_lf_paths.append(str(lf_path))
        skeleton_lfs.append(
            RegularLFPlan(
                index=lf_idx,
                segments=[],
                montage_peaks=[],
                shorts_target=3,
                region="ireland",
                total_duration=600.0,
                undersized=False,
            )
        )
    drive_plan = DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=skeleton_lfs,
        score_threshold=0.42,
        total_source_duration=1800.0,
        total_highlight_duration=600.0,
        best_moments_filtered=False,
    )
    # Serialize via to_dict() then strip per_lf_gpx_path to simulate
    # a legacy file written before the field was emitted.
    seed = PipelineState(
        source_dir=tmp_path / "src",
        output_base=tmp_path / "out",
        drive_plan=drive_plan,
        phases_completed=["phase_a", "phase_bc_day_first"],
        regular_lf_paths=regular_lf_paths,
        shorts_per_lf={i: [] for i in lf_indices},
    )
    state_dict = seed.to_dict()
    state_dict.pop("per_lf_gpx_path", None)
    return state_dict, regular_lf_paths


def test_legacy_state_single_gps_less_lf_migrates_to_none(tmp_path, caplog):
    """Legacy state with 1 LF + no .gpx sibling → per_lf_gpx_path = {1: None}."""
    import logging

    state_dict, _ = _make_legacy_state_dict(tmp_path, lf_indices=(1,), make_gpx_for=())
    with caplog.at_level(logging.WARNING):
        restored = PipelineState.from_dict(state_dict)
    assert restored.per_lf_gpx_path == {1: None}
    # WARNING line logged so operators see the migration.
    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "per_lf_gpx_path migration" in messages
    # Invariant accepts the migrated state.
    from gomoveshift.pipeline.state_invariants import assert_state_invariants

    assert_state_invariants(restored)  # no raise


def test_legacy_state_mixed_gps_and_gps_less_lfs(tmp_path):
    """Legacy state with 2 LFs (one GPS, one GPS-less) → discover + None mix."""
    state_dict, _ = _make_legacy_state_dict(
        tmp_path,
        lf_indices=(1, 2),
        make_gpx_for=(1,),
    )
    restored = PipelineState.from_dict(state_dict)
    assert set(restored.per_lf_gpx_path.keys()) == {1, 2}
    assert restored.per_lf_gpx_path[1] is not None
    assert restored.per_lf_gpx_path[1].endswith("gopro_01.gpx")
    assert restored.per_lf_gpx_path[2] is None


def test_fresh_full_drive_gps_less_no_migration_needed(tmp_path):
    """Fresh phase_bc full_drive GPS-less: per_lf_gpx_path already
    populated with explicit None — invariant passes, no migration log."""
    state_dict, _ = _make_legacy_state_dict(tmp_path, lf_indices=(1,), make_gpx_for=())
    state_dict["per_lf_gpx_path"] = {1: None}  # already populated; no migration
    restored = PipelineState.from_dict(state_dict)
    assert restored.per_lf_gpx_path == {1: None}
    from gomoveshift.pipeline.state_invariants import assert_state_invariants

    assert_state_invariants(restored)  # no raise


def test_legacy_state_missing_lf_directory_fills_none(tmp_path):
    """Legacy state referencing a deleted LF directory → migration fills None."""
    state_dict, _ = _make_legacy_state_dict(tmp_path, lf_indices=(1,), make_gpx_for=())
    # Overwrite path to a non-existent dir; the migration must fill None.
    state_dict["regular_lf_paths"] = [
        str(tmp_path / "deleted_run" / "day_2026-02-21" / "regular_lfs" / "lf_01.mp4")
    ]
    state_dict.pop("per_lf_gpx_path", None)
    restored = PipelineState.from_dict(state_dict)
    assert restored.per_lf_gpx_path == {1: None}
