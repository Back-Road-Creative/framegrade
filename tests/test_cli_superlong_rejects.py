"""``--superlong`` mutual-exclusion rejects, same style as
``--lf-target-duration`` (test_cli_lf_target_duration.py):

  1. Reject ``--superlong`` with ``--lf-mode full_drive``.
  2. Reject ``--superlong`` with ``--lf-target-duration``.
  3. ``--superlong`` alone still works (no reject).

``--superlong`` is registry-backed (flag_registry.py); these checks are
validation-only cross-flag rejects, not flag merging.
"""

from __future__ import annotations

from gomoveshift.config import Config
from test_cli_lf_target_duration import _drive_args, _run_cmd_drive


class TestSuperlongRejects:
    def test_full_drive_mode_rejected(self, tmp_path, capsys):
        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            lf_mode="full_drive",
            superlong_enabled=True,
        )
        rc = _run_cmd_drive(args, cfg)
        assert rc == 2
        err = capsys.readouterr().err
        assert "incompatible" in err.lower()
        assert "full_drive" in err

    def test_lf_target_duration_rejected(self, tmp_path, capsys):
        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            lf_target_duration=15.0,
            superlong_enabled=True,
        )
        rc = _run_cmd_drive(args, cfg)
        assert rc == 2
        err = capsys.readouterr().err
        assert "incompatible" in err.lower()
        assert "lf-target-duration" in err

    def test_superlong_alone_still_works(self, tmp_path, capsys):
        cfg = Config()
        args = _drive_args(
            directory=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            superlong_enabled=True,
        )
        rc = _run_cmd_drive(args, cfg)
        assert rc == 0
        assert capsys.readouterr().err == ""
