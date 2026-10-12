"""Tests for the refactored ``run_validation`` callable and ``ValidationReport``.

The body of ``tools/validate_pipeline_output.py::main`` is a pure
``run_validation(...)`` callable so orchestrators can call it in-process.
These tests pin that API.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import validate_pipeline_output as v  # noqa: E402


def _load_sibling_fixtures():
    """Load _build_valid_output + PROJECT from the sibling test module.

    ``tests/`` is not a Python package so direct ``from tests.xxx import`` is
    unavailable; use importlib.util instead to keep the existing fixture
    scaffolding as the single source of truth.
    """
    sibling = Path(__file__).resolve().parent / "test_validate_pipeline_output.py"
    spec = importlib.util.spec_from_file_location("_gms_validator_fixtures", sibling)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROJECT, mod._build_valid_output


PROJECT, _build_valid_output = _load_sibling_fixtures()


# ---------------------------------------------------------------------------
# ValidationReport invariants
# ---------------------------------------------------------------------------


def _canned_probe_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid invoking real ffprobe so tests don't depend on ffmpeg binaries."""
    canned = {
        "format": {"duration": "30.0"},
        "streams": [
            {"codec_type": "video", "codec_name": "hevc", "duration": "30.0"},
            {"codec_type": "audio", "codec_name": "aac", "duration": "30.0"},
        ],
    }
    monkeypatch.setattr(v, "ffprobe", lambda _path: canned)


def test_run_validation_callable_on_passing_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fully-populated output tree returns 13/13 PASS via run_validation."""
    _canned_probe_patch(monkeypatch)
    paths = _build_valid_output(tmp_path, lf_count=2)
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
    )
    assert isinstance(report, v.ValidationReport)
    assert report.passed_count == 15
    assert report.failed_count == 0
    assert len(report.checks) == 15


def test_run_validation_tolerates_output_base_as_project_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``--output-dir`` is the project dir itself (``output_base.name ==
    project``), run_validation must resolve ``project_dir = output_base`` —
    mirroring ``drive_orchestrator._resolve_paths`` already_nested — instead of
    double-nesting into ``output_base/<project>/`` where nothing exists. Without
    this the DoD validator hard-fails a run whose artifacts phase_d wrote flat
    to ``output_base``."""
    _canned_probe_patch(monkeypatch)
    paths = _build_valid_output(tmp_path, lf_count=2)
    # Pass the project dir itself as output_base (nested convention).
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["project_dir"],  # output_base.name == PROJECT
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
    )
    assert report.failed_count == 0, [c for c in report.checks if not c.passed]
    assert report.passed_count == 15


def test_run_validation_callable_on_missing_lf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing an LF causes check 01 (lf_count) to fail."""
    _canned_probe_patch(monkeypatch)
    paths = _build_valid_output(tmp_path, lf_count=2)
    # Nuke all LFs so count drops to 0 (< expected_lf_min=1)
    for lf in paths["lfs"]:
        lf.unlink()
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
    )
    assert report.failed_count >= 1
    check_01 = next(c for c in report.checks if c.number == 1)
    assert check_01.passed is False


def test_main_cli_still_works(tmp_path: Path) -> None:
    """Invoking the CLI as before returns exit 0 on a passing fixture."""
    paths = _build_valid_output(tmp_path, lf_count=1)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "validate_pipeline_output.py"),
            "--project",
            PROJECT,
            "--output-base",
            str(paths["output_base"]),
            "--state-dir",
            str(paths["state_dir"]),
            "--snapshot-dir",
            str(paths["snapshot_dir"]),
            "--run-start",
            paths["run_start"].isoformat(),
            "--min-lf-duration",
            "25",
            "--max-lf-duration",
            "35",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Summary:" in proc.stdout


def test_format_table_stable_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """format_table emits header, per-check rows (with PASS/FAIL), and summary."""
    _canned_probe_patch(monkeypatch)
    paths = _build_valid_output(tmp_path, lf_count=1)
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
    )
    table = report.format_table()
    assert table.startswith(f"== DoD Validation: {PROJECT}")
    # Each surviving check row (1-13, minus #7 retired) shows a
    # PASS or FAIL tag + a zero-padded number.
    for i in range(1, 14):
        if i == 7:
            continue
        assert f" {i:02d} " in table
    assert f"Summary: {report.passed_count}/15" in table


def test_run_validation_does_not_sys_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pure callable contract: run_validation never calls sys.exit."""
    _canned_probe_patch(monkeypatch)
    paths = _build_valid_output(tmp_path, lf_count=1)

    called = {"exit": False}

    def _fake_exit(_code=0):  # pragma: no cover — guard that shouldn't fire
        called["exit"] = True
        raise AssertionError("sys.exit was called inside run_validation")

    monkeypatch.setattr(sys, "exit", _fake_exit)

    v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
    )
    assert called["exit"] is False


def test_run_validation_accepts_iso_string_run_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_start accepts both datetime and ISO-8601 str."""
    _canned_probe_patch(monkeypatch)
    paths = _build_valid_output(tmp_path, lf_count=1)
    # Pass run_start as ISO string rather than datetime
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"].isoformat(),
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
    )
    assert report.passed_count == 15


def test_validation_report_passed_failed_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """passed_count + failed_count == total checks, always."""
    _canned_probe_patch(monkeypatch)
    paths = _build_valid_output(tmp_path, lf_count=1)
    # Ghost metadata -> one failure (per-LF Metricool CSVs are retired;
    # ghost detection now fires on per-LF youtube.json sidecars).
    ghost = paths["project_dir"] / f"{PROJECT}_lf99_youtube.json"
    ghost.write_text(json.dumps({"title": "ghost", "description": ""}))
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
    )
    assert report.passed_count + report.failed_count == len(report.checks)
    assert report.failed_count >= 1


def test_validation_report_youtube_uploads_sha_kwarg_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """youtube_uploads_sha kwarg is accepted (soft-skip wiring); does not crash."""
    _canned_probe_patch(monkeypatch)
    paths = _build_valid_output(tmp_path, lf_count=1)
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        youtube_uploads_sha=None,
    )
    assert report.passed_count == 15
