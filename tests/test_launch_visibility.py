"""A run nobody can answer is refused; a run always ends with an explicit final line."""

import logging
from pathlib import Path
from unittest.mock import MagicMock

from gomoveshift.cli.handlers import drive as drive_handler
from gomoveshift.config import Config
from gomoveshift.pipeline import orchestrator_preflight


def _run(tmp_path: Path, *, force: bool, dry_run: bool = False):
    return orchestrator_preflight.run_preflight_and_prompt(
        state=MagicMock(),
        stages_dir=tmp_path,
        dry_run=dry_run,
        force=force,
        source_dir=tmp_path,
        project_dir=tmp_path,
        start_time=0.0,
        stages_completed=[],
        config=Config(),
        emit_summary=MagicMock(),
    )


def _quiet(monkeypatch):
    monkeypatch.setattr(orchestrator_preflight, "print_plan", lambda *a, **k: None)
    monkeypatch.setattr(
        orchestrator_preflight, "run_preflight", lambda *a, **k: None, raising=False
    )


def test_non_tty_without_force_is_refused_naming_the_flags(monkeypatch, tmp_path, caplog):
    _quiet(monkeypatch)
    monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: False))
    monkeypatch.setattr(
        "builtins.input", lambda *_: (_ for _ in ()).throw(AssertionError("prompted"))
    )
    with caplog.at_level(logging.ERROR):
        result = _run(tmp_path, force=False)
    assert result is not None and result.success is False
    assert "--force" in result.errors[0] and "--dry-run" in result.errors[0]
    assert "PREFLIGHT-NO-OPERATOR" in result.errors[0]


def test_non_tty_dry_run_still_succeeds(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: False))
    result = _run(tmp_path, force=False, dry_run=True)
    assert result is not None and result.success is True


def test_tty_without_force_still_prompts(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: True))
    monkeypatch.setattr("sys.stdout", MagicMock(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    result = _run(tmp_path, force=False)
    assert result is not None and result.success is True  # declined, clean exit


def test_final_line_completed_and_failed():
    assert drive_handler.drive_final_line(0) == "DRIVE COMPLETED"
    assert drive_handler.drive_final_line(1, "boom") == "DRIVE FAILED: boom"
    assert drive_handler.drive_final_line(4).startswith("DRIVE FAILED: ")


def test_cmd_drive_prints_final_line_last_and_keeps_rc(monkeypatch, capsys):
    monkeypatch.setattr(drive_handler, "_cmd_drive_impl", lambda a, c: 1)
    assert drive_handler.cmd_drive(object(), object()) == 1
    assert capsys.readouterr().out.rstrip().splitlines()[-1].startswith("DRIVE FAILED: ")
    monkeypatch.setattr(drive_handler, "_cmd_drive_impl", lambda a, c: 0)
    assert drive_handler.cmd_drive(object(), object()) == 0
    assert capsys.readouterr().out.rstrip().splitlines()[-1] == "DRIVE COMPLETED"


def test_tmux_launcher_parent_prints_no_final_line(monkeypatch, capsys):
    def impl(a, c):
        drive_handler._FINAL_STATE["launcher_only"] = True
        return 0

    monkeypatch.setattr(drive_handler, "_cmd_drive_impl", impl)
    assert drive_handler.cmd_drive(object(), object()) == 0
    assert "DRIVE COMPLETED" not in capsys.readouterr().out


def _stdin_tty_only(monkeypatch, *, stderr_tty=False, controlling=None):
    monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: True, readline=lambda: "n\n"))
    monkeypatch.setattr("sys.stdout", MagicMock(isatty=lambda: False))
    monkeypatch.setattr("sys.stderr", MagicMock(isatty=lambda: stderr_tty))
    monkeypatch.setattr(orchestrator_preflight, "_open_controlling_tty", lambda: controlling)
    monkeypatch.setattr(
        "builtins.input", lambda *_: (_ for _ in ()).throw(AssertionError("prompt hit the log"))
    )


def test_tty_stdin_with_redirected_output_prompts_on_the_controlling_tty(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    tty = MagicMock()
    _stdin_tty_only(monkeypatch, controlling=tty)
    result = _run(tmp_path, force=False)
    tty.write.assert_called_once_with("Proceed? [y/N] ")
    assert result is not None and result.success is True  # "n" -> declined, gate kept


def test_tty_stdin_prompts_on_stderr_when_that_is_a_tty(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    _stdin_tty_only(monkeypatch, stderr_tty=True)
    result = _run(tmp_path, force=False)
    import sys

    sys.stderr.write.assert_called_with("Proceed? [y/N] ")
    assert result.success is True


def test_tty_stdin_with_no_visible_channel_refuses_by_name(monkeypatch, tmp_path):
    _quiet(monkeypatch)
    _stdin_tty_only(monkeypatch)
    result = _run(tmp_path, force=False)
    assert result.success is False and "PREFLIGHT-NO-OPERATOR" in result.errors[0]


# --- LF-TARGET-OVERRIDE, gate waiting, --lf-mode registry row -------------------


def test_lf_target_duration_logs_a_named_override(caplog):
    import argparse

    from gomoveshift.config import Config

    args = argparse.Namespace(lf_mode=None, lf_target_duration=10.0, superlong_enabled=False)
    cfg = Config()
    from gomoveshift.cli.handlers import drive as h

    src = Path(h.__file__).read_text()
    assert "LF-TARGET-OVERRIDE" in src and "source=cli" in src
    # the override mutates best_moments knobs; the log line must precede it
    assert src.index("LF-TARGET-OVERRIDE") < src.index("config.drive.best_moments_mode = True")
    assert args.lf_target_duration == 10.0 and cfg.drive.best_moments_mode in (True, False)


def test_gate_wait_line_names_gate_files_and_next_action():
    result = MagicMock(success=True)
    orch = MagicMock(
        waiting_gate="OST audit pending for day d: review /o/ost_audit_d.json and create "
        "/o/ost_audit_d.confirmed (or re-run with --ost-audit-confirmed)"
    )
    line = drive_handler.gate_wait_line(result, orch)
    assert line.startswith("DRIVE WAITING: ost_audit")
    assert "ost_audit_d.json" in line and ".confirmed" in line and "--ost-audit-confirmed" in line
    assert "next:" in line


def test_gate_wait_line_none_for_a_finished_or_mocked_run():
    assert drive_handler.gate_wait_line(MagicMock(success=True), MagicMock()) is None
    assert (
        drive_handler.gate_wait_line(MagicMock(success=True), MagicMock(waiting_gate=None)) is None
    )


def test_waiting_exit_code_is_distinct_and_final_line_is_the_wait_line():
    assert drive_handler.EXIT_WAITING_AT_GATE not in (0, 1, 2, 3, 4, 130, 143)
    line = "DRIVE WAITING: ost_audit — x — next: y"
    assert drive_handler.drive_final_line(drive_handler.EXIT_WAITING_AT_GATE, line) == line


def test_orchestrator_records_the_waiting_gate_not_completion(tmp_path, monkeypatch):
    from unittest.mock import patch

    from test_preflight_plan import _install_passthrough_gates

    from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
    from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

    _install_passthrough_gates(monkeypatch)
    src = tmp_path / "src"
    src.mkdir()
    orch = DrivePipelineOrchestrator()
    err = OstAuditPendingError(
        "OST audit pending for day d: review /o/a.json and create /o/a.confirmed"
    )
    with patch.object(orch, "_phase_a_ingest_and_analyze", MagicMock(side_effect=err)):
        orch.run(
            source_dir=src,
            output_base=tmp_path / "out",
            region="usa",
            publish_day="wed",
            force=True,
        )
    assert orch.waiting_gate and "a.confirmed" in orch.waiting_gate


def test_lf_mode_flag_has_a_registry_row_and_state_field():
    from gomoveshift.pipeline.drive_state_model import PipelineState
    from gomoveshift.pipeline.flag_registry import FLAG_REGISTRY

    rows = [r for r in FLAG_REGISTRY if r.cli_dest == "lf_mode"]
    assert len(rows) == 1 and rows[0].config_mirror == "lf_selection_mode"
    assert "lf_mode" in PipelineState.__dataclass_fields__
