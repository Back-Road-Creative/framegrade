"""Tests for the ``drive`` subcommand's flag plumbing.

Regression guard: per-invocation operator flags (``--force-sidecar-reset``,
``--skip-upload``, ``--youtube-upload``) must flow from the CLI parser
→ ``cmd_drive`` → ``DrivePipelineOrchestrator.run(...)``. When a new flag
is added, both the parser and the handler need a corresponding update,
or the flag is silently dropped — the exact failure mode that masked the
youtube_uploads.json phantom-sidecar bug.

``--force-sidecar-reset`` is the sidecar escape hatch — this module pins the flag
plumbing end to end.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

from gomoveshift.cli import _build_parser
from gomoveshift.cli.handlers.drive import cmd_drive
from gomoveshift.config import get_config
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator


class TestDriveCliForceSidecarResetFlag:
    """--force-sidecar-reset must parse, propagate, and default to False."""

    def test_parser_recognizes_flag(self, tmp_path):
        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--force-sidecar-reset",
            ]
        )
        assert getattr(args, "force_sidecar_reset") is True

    def test_parser_default_is_none(self, tmp_path):
        """Argparse-sentinel pattern — the
        parser defaults to ``None`` (not supplied) instead of ``False``
        so the orchestrator's _try_resume path can distinguish
        "operator did not pass the flag" from "operator explicitly
        said no". See gomoveshift/cli/parser.py comment near
        --force-sidecar-reset.
        """
        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
            ]
        )
        assert getattr(args, "force_sidecar_reset") is None

    def test_orchestrator_run_accepts_force_sidecar_reset(self):
        """The orchestrator signature must accept force_sidecar_reset
        so cmd_drive can forward it. A signature mismatch is exactly
        the class of bug that once masked skip_d_copy.

        The sentinel default is ``None`` — see
        parser.py for the argparse-sentinel rationale.
        """
        params = inspect.signature(DrivePipelineOrchestrator.run).parameters
        assert "force_sidecar_reset" in params, (
            "force_sidecar_reset missing from DrivePipelineOrchestrator.run; "
            "cmd_drive would silently drop the CLI flag."
        )
        assert params["force_sidecar_reset"].default is None

    def test_cmd_drive_passes_force_sidecar_reset_to_orchestrator(self, tmp_path, monkeypatch):
        """End-to-end plumbing: cmd_drive must forward args.force_sidecar_reset
        into DrivePipelineOrchestrator.run(...)."""
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        parser = _build_parser()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--no-tmux",
                "--lf-day",
                "tue",
                "--force-sidecar-reset",
            ]
        )
        # Short-circuit tmux escalation regardless of env.
        monkeypatch.setattr("gomoveshift.cli.handlers.drive.is_in_tmux", lambda: True)
        captured_kwargs: dict = {}

        class FakeResult:
            success = True
            drive_plan = None
            errors: list = []

        def fake_run(self, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeResult()

        with patch.object(DrivePipelineOrchestrator, "run", fake_run):
            rc = cmd_drive(args, get_config())
        assert rc == 0
        assert captured_kwargs.get("force_sidecar_reset") is True, (
            f"cmd_drive failed to forward force_sidecar_reset; "
            f"captured kwargs: {sorted(captured_kwargs)}"
        )

    def test_cmd_drive_default_force_sidecar_reset_none(self, tmp_path, monkeypatch):
        """When the flag is absent, cmd_drive must forward the sentinel
        ``None`` (not ``False``). The
        sentinel-vs-explicit distinction lets the orchestrator's
        _try_resume preserve a saved True across invocations that
        didn't re-pass the flag.
        """
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        parser = _build_parser()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--no-tmux",
                "--lf-day",
                "tue",
            ]
        )
        monkeypatch.setattr("gomoveshift.cli.handlers.drive.is_in_tmux", lambda: True)
        captured_kwargs: dict = {}

        class FakeResult:
            success = True
            drive_plan = None
            errors: list = []

        def fake_run(self, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeResult()

        with patch.object(DrivePipelineOrchestrator, "run", fake_run):
            rc = cmd_drive(args, get_config())
        assert rc == 0
        assert captured_kwargs.get("force_sidecar_reset") is None


# ---------------------------------------------------------------------------
# --dry-run / --force flags
# ---------------------------------------------------------------------------
class TestDriveCliPreflightFlags:
    """--dry-run / --force must parse, propagate, and default to False.

    Adds the same signature-level, parser-level and plumbing coverage used
    by the force-sidecar-reset block above. Without these we silently drop
    the operator's intent.
    """

    def test_parser_recognizes_dry_run(self, tmp_path):
        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--dry-run",
            ]
        )
        assert getattr(args, "dry_run") is True

    def test_parser_recognizes_force(self, tmp_path):
        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--force",
            ]
        )
        assert getattr(args, "force") is True

    def test_parser_defaults_both_false(self, tmp_path):
        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
            ]
        )
        assert getattr(args, "dry_run") is False
        assert getattr(args, "force") is False

    def test_orchestrator_run_accepts_dry_run_and_force(self):
        params = inspect.signature(DrivePipelineOrchestrator.run).parameters
        assert "dry_run" in params, (
            "dry_run missing from DrivePipelineOrchestrator.run — "
            "cmd_drive would silently drop --dry-run."
        )
        assert params["dry_run"].default is False
        assert "force" in params, (
            "force missing from DrivePipelineOrchestrator.run — "
            "cmd_drive would silently drop --force."
        )
        assert params["force"].default is False

    def test_cmd_drive_forwards_dry_run_and_force(self, tmp_path, monkeypatch):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        parser = _build_parser()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--no-tmux",
                "--lf-day",
                "tue",
                "--dry-run",
                "--force",
            ]
        )
        monkeypatch.setattr("gomoveshift.cli.handlers.drive.is_in_tmux", lambda: True)
        captured_kwargs: dict = {}

        class FakeResult:
            success = True
            drive_plan = None
            errors: list = []

        def fake_run(self, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeResult()

        with patch.object(DrivePipelineOrchestrator, "run", fake_run):
            rc = cmd_drive(args, get_config())
        assert rc == 0
        assert captured_kwargs.get("dry_run") is True
        assert captured_kwargs.get("force") is True

    def test_cmd_drive_defaults_dry_run_and_force_false(self, tmp_path, monkeypatch):
        """Defaults propagate as False, not None — so the orchestrator's
        default doesn't win and hide operator-absent intent."""
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        parser = _build_parser()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--no-tmux",
                "--lf-day",
                "tue",
            ]
        )
        monkeypatch.setattr("gomoveshift.cli.handlers.drive.is_in_tmux", lambda: True)
        captured_kwargs: dict = {}

        class FakeResult:
            success = True
            drive_plan = None
            errors: list = []

        def fake_run(self, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeResult()

        with patch.object(DrivePipelineOrchestrator, "run", fake_run):
            rc = cmd_drive(args, get_config())
        assert rc == 0
        assert captured_kwargs.get("dry_run") is False
        assert captured_kwargs.get("force") is False


# ---------------------------------------------------------------------------
# --per-lf-location and --list-lfs
# ---------------------------------------------------------------------------
class TestDriveCliPerLfLocation:
    """--per-lf-location KEY=VALUE (repeatable) + --list-lfs."""

    def test_parser_recognizes_single_per_lf_location(self, tmp_path):
        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--per-lf-location",
                "1=Alpha to Beta via Route 9, Example County",
            ]
        )
        assert args.per_lf_locations == [(1, "Alpha to Beta via Route 9, Example County")]

    def test_parser_recognizes_repeated_per_lf_location(self, tmp_path):
        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--per-lf-location",
                "1=Route A",
                "--per-lf-location",
                "2=Route B",
            ]
        )
        assert args.per_lf_locations == [(1, "Route A"), (2, "Route B")]

    def test_parser_rejects_malformed_kv_at_argparse_time(self, tmp_path):
        import pytest

        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "drive",
                    str(src),
                    "--region",
                    "usa",
                    "--output-dir",
                    str(out),
                    "--per-lf-location",
                    "no-equals-sign",
                ]
            )

    def test_parser_rejects_non_integer_index(self, tmp_path):
        import pytest

        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "drive",
                    str(src),
                    "--region",
                    "usa",
                    "--output-dir",
                    str(out),
                    "--per-lf-location",
                    "abc=Some Route",
                ]
            )

    def test_parser_rejects_empty_label(self, tmp_path):
        import pytest

        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "drive",
                    str(src),
                    "--region",
                    "usa",
                    "--output-dir",
                    str(out),
                    "--per-lf-location",
                    "1=",
                ]
            )

    def test_validator_rejects_when_all_overrides_equal_location(self, tmp_path):
        """Cross-flag rule: every override label == --location is the
        ambiguous case the design explicitly rejects (would silently
        re-introduce the deleted drive-wide fallback)."""
        import argparse
        import pytest

        from gomoveshift.cli.parser import _validate_per_lf_location_args

        ns = argparse.Namespace(
            location="Exampleland",
            per_lf_locations=[(1, "Exampleland"), (2, "Exampleland")],
        )
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _validate_per_lf_location_args(ns)
        assert "more specific" in str(exc.value)

    def test_validator_accepts_mixed_overrides_and_location(self, tmp_path):
        """One override equals --location, another doesn't → allowed
        (only ALL == location is rejected by rule 1).

        Drive-wide must have >= 2 comma components
        and every per-LF label must end with drive-wide at the comma
        boundary, so this test uses a properly-formed two-component
        drive-wide with consistent per-LF tails."""
        import argparse

        from gomoveshift.cli.parser import _validate_per_lf_location_args

        ns = argparse.Namespace(
            location="Example County, Exampleland",
            per_lf_locations=[
                (1, "Example County, Exampleland"),
                (2, "Alpha to Beta, Example County, Exampleland"),
            ],
        )
        _validate_per_lf_location_args(ns)  # no raise

    def test_parser_recognizes_list_lfs(self, tmp_path):
        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--list-lfs",
            ]
        )
        assert args.list_lfs is True

    def test_parser_default_list_lfs_false(self, tmp_path):
        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        args = parser.parse_args(["drive", str(src), "--region", "usa", "--output-dir", str(out)])
        assert args.list_lfs is False

    def test_orchestrator_run_accepts_per_lf_overrides_and_list_lfs(self):
        params = inspect.signature(DrivePipelineOrchestrator.run).parameters
        assert "per_lf_overrides" in params
        assert "list_lfs" in params

    def test_cmd_drive_forwards_per_lf_overrides_to_orchestrator(self, tmp_path, monkeypatch):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        parser = _build_parser()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--no-tmux",
                "--lf-day",
                "tue",
                "--per-lf-location",
                "1=Route A",
                "--per-lf-location",
                "2=Route B",
            ]
        )
        monkeypatch.setattr("gomoveshift.cli.handlers.drive.is_in_tmux", lambda: True)
        captured_kwargs: dict = {}

        class FakeResult:
            success = True
            drive_plan = None
            errors: list = []

        def fake_run(self, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeResult()

        with patch.object(DrivePipelineOrchestrator, "run", fake_run):
            rc = cmd_drive(args, get_config())
        assert rc == 0
        assert captured_kwargs.get("per_lf_overrides") == {1: "Route A", 2: "Route B"}

    def test_cmd_drive_returns_nonzero_when_all_overrides_equal_location(
        self, tmp_path, monkeypatch
    ):
        """Cross-flag validation runs before any tmux / orchestrator
        call — invalid invocations exit fast with a non-zero code."""
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        parser = _build_parser()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--no-tmux",
                "--lf-day",
                "tue",
                "--location",
                "Exampleland",
                "--per-lf-location",
                "1=Exampleland",
                "--per-lf-location",
                "2=Exampleland",
            ]
        )
        monkeypatch.setattr("gomoveshift.cli.handlers.drive.is_in_tmux", lambda: True)
        # Orchestrator.run must NOT be called — the validator should
        # bail before we get there.
        called: dict = {"yes": False}

        def _run(self, **kwargs):
            called["yes"] = True
            raise AssertionError("orchestrator.run should not be called")

        with patch.object(DrivePipelineOrchestrator, "run", _run):
            rc = cmd_drive(args, get_config())
        assert rc != 0
        assert called["yes"] is False


class TestDriveCliJournalHookOnPostUploadFailure:
    """PipelineValidationError / PhaseQualityError fire AFTER Phase D's
    irreversible uploads. The journal auto-publish hook (journey merge /
    deferral sidecar / video_routing append) must still run before the error
    propagates — the uploads already happened and the merge is recoverable, so
    a silent skip contradicts the 'warns loudly, never a silent skip' contract.
    The original error must NOT be swallowed: the exit code still signals it.
    """

    def _parse(self, tmp_path):
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        parser = _build_parser()
        return parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "usa",
                "--output-dir",
                str(out),
                "--no-tmux",
                "--lf-day",
                "tue",
            ]
        )

    def test_phase_quality_error_still_runs_journal_hook(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline.exceptions import PhaseQualityError

        args = self._parse(tmp_path)
        monkeypatch.setattr("gomoveshift.cli.handlers.drive.is_in_tmux", lambda: True)
        hook_calls: list = []
        monkeypatch.setattr(
            "gomoveshift.cli.handlers.drive._maybe_auto_publish_journal",
            lambda *a, **k: hook_calls.append(a),
        )

        def fake_run(self, **kwargs):
            raise PhaseQualityError("title collision")

        with patch.object(DrivePipelineOrchestrator, "run", fake_run):
            rc = cmd_drive(args, get_config())
        assert rc == 4, "the Phase-D quality-gate exit code must still propagate"
        assert len(hook_calls) == 1, "journal hook must run despite the post-upload failure"

    def test_pipeline_validation_error_still_runs_journal_hook(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline.exceptions import PipelineValidationError

        args = self._parse(tmp_path)
        monkeypatch.setattr("gomoveshift.cli.handlers.drive.is_in_tmux", lambda: True)
        hook_calls: list = []
        monkeypatch.setattr(
            "gomoveshift.cli.handlers.drive._maybe_auto_publish_journal",
            lambda *a, **k: hook_calls.append(a),
        )

        def fake_run(self, **kwargs):
            raise PipelineValidationError("DoD validation failed")

        with patch.object(DrivePipelineOrchestrator, "run", fake_run):
            rc = cmd_drive(args, get_config())
        assert rc == 1, "the DoD-validation exit code must still propagate"
        assert len(hook_calls) == 1, "journal hook must run despite the post-upload failure"


class TestDriveCliDriveWideLocationSpecificity:
    """--location must be a comma-suffix of every --per-lf-location
    label AND have >= 2 comma components when overrides are set.

    Catches the case where the operator passed
    --per-lf-location '1=Example Pass, Example County, Exampleland' but
    --location 'Exampleland' — drive-wide silently surfaced "Exampleland" in
    Gate A and on the published page's fallback paths."""

    def test_rejects_single_component_drive_wide_when_per_lf_set(self):
        import argparse

        import pytest

        from gomoveshift.cli.parser import _validate_per_lf_location_args

        ns = argparse.Namespace(
            location="Exampleland",
            per_lf_locations=[(1, "Alpha to Beta Pass, Example County, Exampleland")],
        )
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _validate_per_lf_location_args(ns)
        msg = str(exc.value)
        assert ">= 2 components" in msg
        assert "Example County, Exampleland" in msg  # suggestion derived from per-LF

    def test_rejects_drive_wide_that_is_not_comma_suffix_of_per_lf(self):
        import argparse

        import pytest

        from gomoveshift.cli.parser import _validate_per_lf_location_args

        # "Beta Pass" lives inside the first comma-component of per-LF
        # ("Alpha to Beta Pass"), so "Beta Pass, Example County, Exampleland" is
        # not a comma-suffix at the comma boundary.
        ns = argparse.Namespace(
            location="Beta Pass, Example County, Exampleland",
            per_lf_locations=[(1, "Alpha to Beta Pass, Example County, Exampleland")],
        )
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _validate_per_lf_location_args(ns)
        msg = str(exc.value)
        assert "comma-suffix" in msg

    def test_accepts_two_component_comma_suffix(self):
        import argparse

        from gomoveshift.cli.parser import _validate_per_lf_location_args

        ns = argparse.Namespace(
            location="Example County, Exampleland",
            per_lf_locations=[(1, "Alpha to Beta Pass, Example County, Exampleland")],
        )
        _validate_per_lf_location_args(ns)  # no raise

    def test_accepts_full_comma_suffix(self):
        import argparse

        from gomoveshift.cli.parser import _validate_per_lf_location_args

        ns = argparse.Namespace(
            location="Beta Pass, Example County, Exampleland",
            per_lf_locations=[(1, "Beta Pass, Example County, Exampleland")],
        )
        # This hits the all-equal rule first — that's a different test
        # class. Here we use a label that's STRICTLY a superset.
        ns.per_lf_locations = [(1, "Detour, Beta Pass, Example County, Exampleland")]
        _validate_per_lf_location_args(ns)  # no raise

    def test_suggestion_takes_common_suffix_across_multiple_lfs(self):
        import argparse

        import pytest

        from gomoveshift.cli.parser import _validate_per_lf_location_args

        ns = argparse.Namespace(
            location="Exampleland",
            per_lf_locations=[
                (1, "Alpha to Delta, Example County, Exampleland"),
                (2, "Delta to Gamma, Example County, Exampleland"),
            ],
        )
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _validate_per_lf_location_args(ns)
        assert "Example County, Exampleland" in str(exc.value)

    def test_no_validation_when_per_lf_not_set(self):
        import argparse

        from gomoveshift.cli.parser import _validate_per_lf_location_args

        # Drive-wide alone (no per-LF overrides) — single-component is
        # legal because GPS-bearing drives may legitimately use a
        # country-level drive-wide.
        ns = argparse.Namespace(location="Exampleland", per_lf_locations=None)
        _validate_per_lf_location_args(ns)  # no raise

    def test_no_validation_when_drive_wide_empty(self):
        import argparse

        from gomoveshift.cli.parser import _validate_per_lf_location_args

        # PhaseBC coverage gate catches missing labels later. The
        # parser only validates the cross-flag relationship when both
        # are present.
        ns = argparse.Namespace(
            location="",
            per_lf_locations=[(1, "Alpha to Beta Pass, Example County, Exampleland")],
        )
        _validate_per_lf_location_args(ns)  # no raise

    def test_end_to_end_handler_rejects_ireland_with_per_lf(self, tmp_path, monkeypatch):
        """A single-component --location with a per-LF override must fail at cmd_drive entry, before
        any orchestrator work, just like the all-equal case."""
        parser = _build_parser()
        src = tmp_path / "source"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        args = parser.parse_args(
            [
                "drive",
                str(src),
                "--region",
                "ireland",
                "--output-dir",
                str(out),
                "--lf-day",
                "sun",
                "--location",
                "Exampleland",
                "--per-lf-location",
                "1=Alpha to Beta Pass, Example County, Exampleland",
            ]
        )
        monkeypatch.setattr("gomoveshift.cli.handlers.drive.is_in_tmux", lambda: True)
        called: dict = {"yes": False}

        def _run(self, **kwargs):
            called["yes"] = True
            raise AssertionError("orchestrator.run should not be called")

        with patch.object(DrivePipelineOrchestrator, "run", _run):
            rc = cmd_drive(args, get_config())
        assert rc != 0
        assert called["yes"] is False
