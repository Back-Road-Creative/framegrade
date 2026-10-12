"""Phase D per-LF Drive upload failure propagation + single-shape prior-manifest contract.

Two Phase D bugs are closed here:

  Bug A — ``prior_manifest.files`` AttributeError on dict-shaped manifests.
          The state-load path coerces ``upload_manifests`` entries to
          ``UploadManifest`` dataclass instances, so every value is a single
          shape and no dict-tolerance helper is needed.

  Bug B — ``except Exception`` swallow in the per-LF upload loop. The
          previous behaviour logged a WARN, left ``upload_manifests[lf.index]``
          unpopulated, and let the loop continue — causing the consolidated
          Metricool CSV to ship with empty ``Picture Url 1`` cells. The
          broad swallow is gone: non-``_RefreshError`` failures propagate
          immediately with an LF-index ``logger.error`` for operator
          diagnosis. ``_RefreshError`` still re-raises as the existing
          operator-actionable ``RuntimeError`` (regression test included).
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.models import DrivePlan, LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_phases import phase_d as phase_d_mod
from gomoveshift.pipeline.drive_phases.phase_d import (
    _upload_all_lfs_shorts_to_drive,
)
from gomoveshift.pipeline.uploader import UploadManifest

from _state_helpers import make_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_lf(index: int) -> RegularLFPlan:
    seg = LFSegment(
        source_file_index=index,
        source_path=Path(f"/tmp/ep_{index:03d}_FINAL.mp4"),
        start_time=0.0,
        end_time=600.0,
        original_start_time=0.0,
        original_end_time=600.0,
        composite_score=0.7,
        gps_center_lat=0.0,
        gps_center_lon=0.0,
        terrain_type="countryside",
        contiguous_with_previous=False,
    )
    return RegularLFPlan(
        index=index,
        segments=[seg],
        montage_peaks=[seg],
        shorts_target=2,
        region="usa",
        total_duration=600.0,
    )


def _build_state_and_plan(tmp_path: Path, num_lfs: int = 2):
    """Build the minimum state + drive_plan needed for the per-LF upload loop.

    Two LFs each with two day-first shorts under a per-day shorts dir.
    """
    output_base = tmp_path / "output"
    output_base.mkdir()
    stages_dir = tmp_path / "cache" / "_stages"
    day_shorts_dir = stages_dir / "day_2026-04-29" / "shorts"
    day_shorts_dir.mkdir(parents=True)

    shorts_per_lf: dict[int, list[Path]] = {}
    short_counter = 0
    regular_lfs = []
    for i in range(num_lfs):
        regular_lfs.append(_make_lf(i + 1))
        per_lf_shorts: list[Path] = []
        for _j in range(2):
            short_counter += 1
            sp = day_shorts_dir / f"short_{short_counter:02d}.mp4"
            sp.write_bytes(b"y" * 64)
            per_lf_shorts.append(sp)
        shorts_per_lf[i + 1] = per_lf_shorts

    drive_plan = DrivePlan(
        source_files=[],
        processed_files=[],
        regular_lfs=regular_lfs,
        score_threshold=0.5,
        total_source_duration=600.0 * num_lfs,
        total_highlight_duration=600.0 * num_lfs,
    )

    state = make_state(
        shorts_per_lf=shorts_per_lf,
    )
    return state, drive_plan, output_base


# ---------------------------------------------------------------------------
# Bug B — propagation on per-LF upload failure
# ---------------------------------------------------------------------------


def test_per_lf_upload_failure_propagates(tmp_path, caplog):
    """Generic failure on the 2nd LF must propagate; LF index in the error log."""
    state, drive_plan, output_base = _build_state_and_plan(tmp_path, num_lfs=2)

    call_count = {"n": 0}

    def fake_inner_upload(*, uploader, lf, project_name, shorts_paths, output_base, use_dayfirst):
        call_count["n"] += 1
        if lf.index == 2:
            raise RuntimeError("simulated drive failure")
        return UploadManifest(
            project_name=project_name,
            folder_id="folder_lf1",
            files={
                Path(p).name: f"https://drive.google.com/file/d/x{i}"
                for i, p in enumerate(shorts_paths)
            },
            completed=True,
        )

    with (
        patch.object(phase_d_mod, "_upload_lf_shorts_to_drive", side_effect=fake_inner_upload),
        caplog.at_level(logging.ERROR),
    ):
        with pytest.raises(RuntimeError, match="simulated drive failure"):
            _upload_all_lfs_shorts_to_drive(
                drive_plan=drive_plan,
                state=state,
                project_base="proj",
                output_base=output_base,
                uploader_factory=lambda: MagicMock(),
                skip_drive_upload=False,
            )

    # LF 1 ran (success), LF 2 ran (failure) — then loop aborted.
    assert call_count["n"] == 2

    # ERROR log line includes the failing LF index.
    error_msgs = [rec.message for rec in caplog.records if rec.levelno == logging.ERROR]
    assert any("LF 2" in m for m in error_msgs), (
        f"expected ERROR log naming LF 2, got: {error_msgs!r}"
    )


def test_per_lf_upload_failure_no_csv_written(tmp_path):
    """When the upload loop aborts, ``_emit_consolidated_shorts_csv`` is never called.

    The bug this guards against: the former broad ``except Exception`` swallow
    let the loop continue with an empty ``upload_manifests[lf.index]``,
    propagated through to CSV generation, and shipped a Metricool CSV with
    empty ``Picture Url 1`` cells. The fix is that the upload loop now raises
    before CSV generation can occur.
    """
    state, drive_plan, output_base = _build_state_and_plan(tmp_path, num_lfs=2)

    def always_fails(*args, **kwargs):
        raise RuntimeError("first LF blows up")

    csv_spy = MagicMock()
    with (
        patch.object(phase_d_mod, "_upload_lf_shorts_to_drive", side_effect=always_fails),
        patch.object(phase_d_mod, "_emit_consolidated_shorts_csv", csv_spy),
    ):
        with pytest.raises(RuntimeError, match="first LF blows up"):
            _upload_all_lfs_shorts_to_drive(
                drive_plan=drive_plan,
                state=state,
                project_base="proj",
                output_base=output_base,
                uploader_factory=lambda: MagicMock(),
                skip_drive_upload=False,
            )

    csv_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Bug A — dataclass-shaped prior manifest tolerated by skip-when-complete
# predicate. Load-time rehydration via
# ``_coerce_dict_value_dataclasses`` guarantees every ``upload_manifests``
# value is an ``UploadManifest`` instance, so the legacy dict-shape
# tolerance shim (``_manifest_files``) has been retired. The contract is
# now single-shape; this test exercises the resume-skip predicate against
# the canonical shape.
# ---------------------------------------------------------------------------


def test_dataclass_prior_manifest_in_resume_path(tmp_path, caplog):
    """A complete UploadManifest prior skips re-upload."""
    state, drive_plan, output_base = _build_state_and_plan(tmp_path, num_lfs=2)

    # Pre-populate state["upload_manifests"] with an UploadManifest prior for
    # LF 1 whose file count matches the expected day-first shorts count (2).
    # The upload loop should treat this as complete and skip re-upload. LF 2 has no
    # prior and must run.
    state["upload_manifests"] = {
        1: UploadManifest(
            project_name="proj_lf01",
            folder_id="folder_lf01",
            files={
                "short_01.mp4": "https://drive.google.com/file/d/a/view",
                "short_02.mp4": "https://drive.google.com/file/d/b/view",
            },
            completed=True,
        ),
    }

    inner_calls: list[int] = []

    def fake_inner_upload(*, uploader, lf, project_name, shorts_paths, output_base, use_dayfirst):
        inner_calls.append(lf.index)
        return UploadManifest(
            project_name=project_name,
            folder_id=f"folder_lf{lf.index:02d}",
            files={
                Path(p).name: f"https://drive.google.com/file/d/x{i}"
                for i, p in enumerate(shorts_paths)
            },
            completed=True,
        )

    with (
        patch.object(phase_d_mod, "_upload_lf_shorts_to_drive", side_effect=fake_inner_upload),
        caplog.at_level(logging.INFO),
    ):
        manifests, _uploader = _upload_all_lfs_shorts_to_drive(
            drive_plan=drive_plan,
            state=state,
            project_base="proj",
            output_base=output_base,
            uploader_factory=lambda: MagicMock(),
            skip_drive_upload=False,
        )

    # LF 1 (complete) skipped; LF 2 ran.
    assert inner_calls == [2], (
        f"expected complete prior to skip LF 1, got inner calls: {inner_calls}"
    )

    # Both manifests are dataclass-shaped (single-shape contract).
    assert 1 in manifests and 2 in manifests
    assert isinstance(manifests[1], UploadManifest)
    assert isinstance(manifests[2], UploadManifest)

    # The skip-info log mentions LF 1 specifically.
    info_msgs = [rec.message for rec in caplog.records if rec.levelno == logging.INFO]
    assert any("LF 1" in m and "skipping re-upload" in m for m in info_msgs), (
        f"expected skip log for LF 1, got: {info_msgs!r}"
    )


# ---------------------------------------------------------------------------
# Regression — narrow ``_RefreshError`` path is unchanged
# ---------------------------------------------------------------------------


def test_refresh_error_path_unchanged(tmp_path):
    """``_RefreshError`` still surfaces as the operator-actionable ``RuntimeError``."""
    state, drive_plan, output_base = _build_state_and_plan(tmp_path, num_lfs=2)

    # Build a stand-in RefreshError class that matches whatever phase_d
    # currently considers _RefreshError (real google.auth class if installed,
    # otherwise the type(None) sentinel). To make the narrow except actually
    # match, monkey-patch the module-level _RefreshError to a custom class
    # we raise. The test exercises the BRANCH, not the specific google class.
    class _FakeRefreshError(Exception):
        pass

    def fake_inner_upload(*args, **kwargs):
        raise _FakeRefreshError("token expired")

    with (
        patch.object(phase_d_mod, "_RefreshError", _FakeRefreshError),
        patch.object(phase_d_mod, "_upload_lf_shorts_to_drive", side_effect=fake_inner_upload),
    ):
        with pytest.raises(
            RuntimeError, match=r"Drive auth failed.*refresh the Drive token \(GMS_DRIVE_TOKEN_ENC"
        ):
            _upload_all_lfs_shorts_to_drive(
                drive_plan=drive_plan,
                state=state,
                project_base="proj",
                output_base=output_base,
                uploader_factory=lambda: MagicMock(),
                skip_drive_upload=False,
            )
