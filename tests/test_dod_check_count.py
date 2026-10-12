"""Asserts the canonical DoD check list shape after retiring a check.

``check_metricool_text_fidelity`` was deleted from
``tools/validate_pipeline_output.py``. This test pins the post-deletion
shape so a future re-introduction (accidental copy-paste, revert) is
caught by CI rather than shipping a permanent no-op check again.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import validate_pipeline_output as v  # noqa: E402


def test_metricool_text_fidelity_function_is_gone() -> None:
    assert not hasattr(v, "check_metricool_text_fidelity"), (
        "check_metricool_text_fidelity was deleted. "
        "If you need to re-add a per-LF text-fidelity check, give it a new "
        "name and a real implementation — do not restore the no-op stub."
    )


def test_post_upload_mode_excludes_retired_check(tmp_path: Path) -> None:
    project_dir = tmp_path / "tp"
    project_dir.mkdir()
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    report = v.run_validation(
        project="tp",
        output_base=tmp_path,
        snapshot_dir=snapshot,
        state_dir=state_dir,
        run_start=datetime.now(tz=timezone.utc),
        mode=v.Mode.POST_UPLOAD,
    )
    names = {c.name for c in report.checks}
    assert "metricool_text_fidelity" not in names


def test_drive_link_checks_are_gone() -> None:
    """The checks shorts_drive_links_complete and
    metricool_picture_url_fidelity, and their private helpers, are
    deleted along with their POST_UPLOAD/SKIP_UPLOAD_AUDIT registry rows.
    """
    for name in (
        "check_shorts_drive_links_complete",
        "check_metricool_picture_url_fidelity",
        "_PICTURE_URL_VALID_HOSTS",
        "_picture_url_host_ok",
        "_head_drive_url",
    ):
        assert not hasattr(v, name), f"{name} should have been deleted"


def test_post_upload_check_count_is_15(tmp_path: Path) -> None:
    # The check LIST length is fixed regardless of whether individual
    # checks pass or fail against this minimal (empty) tree.
    project_dir = tmp_path / "tp"
    project_dir.mkdir()
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    report = v.run_validation(
        project="tp",
        output_base=tmp_path,
        snapshot_dir=snapshot,
        state_dir=state_dir,
        run_start=datetime(2026, 9, 29, tzinfo=timezone.utc),
        mode=v.Mode.POST_UPLOAD,
    )
    assert len(report.checks) == 15


def test_skip_upload_mode_excludes_retired_check(tmp_path: Path) -> None:
    project_dir = tmp_path / "tp"
    project_dir.mkdir()
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    report = v.run_validation(
        project="tp",
        output_base=tmp_path,
        snapshot_dir=snapshot,
        state_dir=state_dir,
        run_start=datetime.now(tz=timezone.utc),
        mode=v.Mode.SKIP_UPLOAD_AUDIT,
    )
    names = {c.name for c in report.checks}
    assert "metricool_text_fidelity" not in names


def test_skip_upload_check_count_is_12(tmp_path: Path) -> None:
    project_dir = tmp_path / "tp"
    project_dir.mkdir()
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    report = v.run_validation(
        project="tp",
        output_base=tmp_path,
        snapshot_dir=snapshot,
        state_dir=state_dir,
        run_start=datetime(2026, 9, 29, tzinfo=timezone.utc),
        mode=v.Mode.SKIP_UPLOAD_AUDIT,
    )
    assert len(report.checks) == 12


def test_csv_fresh_and_format_checks_are_gone() -> None:
    """The checks metricool_fresh and
    metricool_csv_format, and their private helpers, are deleted along
    with their POST_UPLOAD/SKIP_UPLOAD_AUDIT registry rows.
    """
    for name in (
        "check_metricool_fresh",
        "check_metricool_csv_format",
        "_read_csv_text_field",
        "_all_csv_paths",
    ):
        assert not hasattr(v, name), f"{name} should have been deleted"


def test_validator_never_opens_shorts_csv() -> None:
    src = (REPO_ROOT / "tools" / "validate_pipeline_output.py").read_text(encoding="utf-8")
    assert "_shorts_metricool" not in src


def test_validator_has_no_metricool_references() -> None:
    proc = subprocess.run(
        ["/bin/grep", "-il", "metricool", str(REPO_ROOT / "tools" / "validate_pipeline_output.py")],
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "", f"metricool references remain: {proc.stdout}"
