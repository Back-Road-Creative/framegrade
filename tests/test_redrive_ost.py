"""Tests for the redrive_ost CLI — re-derive OST text without rerunning ingest."""

import json
from pathlib import Path


def _minimal_state(gpx_files: list[str]) -> dict:
    return {
        "source_dir": "/tmp/fake_source",
        "location": "OLD DRIVE OST",
        "episode_locations": ["OLD EP OST"] * len(gpx_files),
        "gpx_files": gpx_files,
        # A real known phase name — redrive now round-trips through
        # load_drive_state/save_drive_state, which filters out unknown
        # phase names (see state_invariants Rule 3 / _KNOWN_PHASES).
        "phases_completed": ["phase_a"],
    }


def _write_state(tmp_path: Path, state: dict) -> Path:
    state_path = tmp_path / "pipeline_state.json"
    state_path.write_text(json.dumps(state, indent=2))
    return state_path


class TestRedriveOST:
    def test_dry_run_does_not_modify_file(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import redrive_ost

        monkeypatch.setattr(
            redrive_ost,
            "derive_episode_ost_text",
            lambda gpx, **kw: f"NEW {Path(gpx).stem}",
        )
        monkeypatch.setattr(
            redrive_ost,
            "derive_drive_ost_text",
            lambda *a, **kw: "NEW DRIVE OST",
        )

        original = _minimal_state(["/fake/ep1.gpx", "/fake/ep2.gpx"])
        state_path = _write_state(tmp_path, original)

        result = redrive_ost.redrive_state(state_path, dry_run=True)

        # File on disk is unchanged
        on_disk = json.loads(state_path.read_text())
        assert on_disk["location"] == "OLD DRIVE OST"
        assert on_disk["episode_locations"] == ["OLD EP OST", "OLD EP OST"]

        # Result carries the proposed changes
        assert result.after_drive == "NEW DRIVE OST"
        assert result.after_episode_locations == ["NEW ep1", "NEW ep2"]
        assert result.backup_path is None

    def test_write_creates_backup_and_updates_state(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import redrive_ost

        monkeypatch.setattr(
            redrive_ost,
            "derive_episode_ost_text",
            lambda gpx, **kw: f"NEW {Path(gpx).stem}",
        )
        monkeypatch.setattr(
            redrive_ost,
            "derive_drive_ost_text",
            lambda *a, **kw: "NEW DRIVE OST",
        )

        original = _minimal_state(["/fake/ep1.gpx", "/fake/ep2.gpx"])
        state_path = _write_state(tmp_path, original)

        result = redrive_ost.redrive_state(state_path, dry_run=False)

        # File on disk is updated
        on_disk = json.loads(state_path.read_text())
        assert on_disk["location"] == "NEW DRIVE OST"
        assert on_disk["episode_locations"] == ["NEW ep1", "NEW ep2"]

        # Other keys preserved
        assert on_disk["source_dir"] == "/tmp/fake_source"
        assert on_disk["phases_completed"] == ["phase_a"]

        # Backup path was created and contains original
        assert result.backup_path is not None
        assert result.backup_path.exists()
        backup = json.loads(result.backup_path.read_text())
        assert backup["location"] == "OLD DRIVE OST"
        assert backup["episode_locations"] == ["OLD EP OST", "OLD EP OST"]

    def test_write_failure_leaves_state_file_intact(self, tmp_path, monkeypatch):
        """Redrive must route the write through the atomic writer so a
        crash partway through (here: os.replace raising, as it would on a
        full disk) leaves the pre-existing state file intact and parseable
        rather than truncated. A bare ``write_text`` has no such guard."""
        from gomoveshift.pipeline import _atomic_write, redrive_ost

        monkeypatch.setattr(
            redrive_ost,
            "derive_episode_ost_text",
            lambda gpx, **kw: f"NEW {Path(gpx).stem}",
        )
        monkeypatch.setattr(
            redrive_ost,
            "derive_drive_ost_text",
            lambda *a, **kw: "NEW DRIVE OST",
        )

        def _raise_disk_full(*args, **kwargs):
            raise OSError("[Errno 28] No space left on device")

        monkeypatch.setattr(_atomic_write.os, "replace", _raise_disk_full)

        original = _minimal_state(["/fake/ep1.gpx", "/fake/ep2.gpx"])
        state_path = _write_state(tmp_path, original)

        import pytest

        with pytest.raises(OSError):
            redrive_ost.redrive_state(state_path, dry_run=False)

        # Pre-existing state file survives the failed write untouched —
        # NOT truncated, NOT replaced with the half-written new content.
        on_disk = json.loads(state_path.read_text())
        assert on_disk["location"] == "OLD DRIVE OST"
        assert on_disk["episode_locations"] == ["OLD EP OST", "OLD EP OST"]

    def test_missing_state_file_raises(self, tmp_path):
        from gomoveshift.pipeline import redrive_ost

        missing = tmp_path / "does_not_exist.json"
        import pytest

        with pytest.raises(FileNotFoundError):
            redrive_ost.redrive_state(missing, dry_run=True)

    def test_empty_gpx_list_still_runs(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import redrive_ost

        monkeypatch.setattr(
            redrive_ost,
            "derive_episode_ost_text",
            lambda gpx, **kw: "",
        )
        monkeypatch.setattr(
            redrive_ost,
            "derive_drive_ost_text",
            lambda *a, **kw: "",
        )

        original = _minimal_state([])
        state_path = _write_state(tmp_path, original)

        result = redrive_ost.redrive_state(state_path, dry_run=True)
        assert result.after_episode_locations == []
        assert result.after_drive == ""

    def test_validator_drops_bad_output(self, tmp_path, monkeypatch):
        """OST output that fails the validator is blanked (consistent with
        orchestrator behavior)."""
        from gomoveshift.pipeline import redrive_ost

        # Return a known-bad string
        monkeypatch.setattr(
            redrive_ost,
            "derive_episode_ost_text",
            lambda gpx, **kw: "Oceanmarsh Subdivision",  # HOA pattern — invalid
        )
        monkeypatch.setattr(
            redrive_ost,
            "derive_drive_ost_text",
            lambda *a, **kw: "None",  # broken literal — invalid
        )

        original = _minimal_state(["/fake/ep1.gpx"])
        state_path = _write_state(tmp_path, original)

        result = redrive_ost.redrive_state(state_path, dry_run=True)
        assert result.after_episode_locations == [""]
        assert result.after_drive == ""

    def test_override_file_applied_after_derivation(self, tmp_path, monkeypatch):
        """ost_overrides.json next to the state file pins exact OST text,
        trumping the algorithmic derivation AND the validator."""
        from gomoveshift.pipeline import redrive_ost

        # Distinct per-episode values so duplicate differentiation is a no-op
        monkeypatch.setattr(
            redrive_ost,
            "derive_episode_ost_text",
            lambda gpx, **kw: f"Auto {Path(gpx).stem}",
        )
        monkeypatch.setattr(
            redrive_ost,
            "derive_drive_ost_text",
            lambda *a, **kw: "Auto Drive",
        )

        original = _minimal_state(["/fake/ep1.gpx", "/fake/ep2.gpx"])
        state_path = _write_state(tmp_path, original)

        # Drop an override file in the same dir
        (tmp_path / "ost_overrides.json").write_text(
            json.dumps(
                {
                    "episode_1": "Hand-picked Ep 2",
                    "drive": "Hand-picked Drive",
                }
            )
        )

        result = redrive_ost.redrive_state(state_path, dry_run=True)
        assert result.after_episode_locations[1] == "Hand-picked Ep 2"
        assert result.after_drive == "Hand-picked Drive"
        # Non-overridden episode keeps the algorithmic value
        assert result.after_episode_locations[0] == "Auto ep1"

    def test_backup_succeeds_when_copystat_unavailable(self, tmp_path, monkeypatch):
        """Regression test for the /mnt/d NTFS bug.

        When the pipeline state lives on an NTFS mount (/mnt/d under WSL),
        BOTH shutil.copystat() and shutil.copymode() raise PermissionError
        trying to preserve file mtime/mode. shutil.copy2() calls copystat
        and shutil.copy() calls copymode — both propagate the error and
        abort redrive before the state file is rewritten.

        shutil.copyfile() is the content-only copy with no metadata
        preservation, safe on NTFS. redrive must use it.

        This test simulates the NTFS failure by patching BOTH copystat
        and copymode to always raise, then asserts the redrive still
        writes the state file and creates a backup.
        """
        import shutil as _shutil

        from gomoveshift.pipeline import redrive_ost

        monkeypatch.setattr(
            redrive_ost,
            "derive_episode_ost_text",
            lambda gpx, **kw: f"NEW {Path(gpx).stem}",
        )
        monkeypatch.setattr(
            redrive_ost,
            "derive_drive_ost_text",
            lambda *a, **kw: "NEW DRIVE OST",
        )

        def _raise_ntfs_perm(*args, **kwargs):
            raise PermissionError("[Errno 1] Operation not permitted")

        # NTFS breaks both of these; copyfile is the only safe path.
        monkeypatch.setattr(_shutil, "copystat", _raise_ntfs_perm)
        monkeypatch.setattr(_shutil, "copymode", _raise_ntfs_perm)

        original = _minimal_state(["/fake/ep1.gpx", "/fake/ep2.gpx"])
        state_path = _write_state(tmp_path, original)

        result = redrive_ost.redrive_state(state_path, dry_run=False)

        # State file was rewritten despite the NTFS metadata failures
        on_disk = json.loads(state_path.read_text())
        assert on_disk["location"] == "NEW DRIVE OST"
        assert on_disk["episode_locations"] == ["NEW ep1", "NEW ep2"]

        # Backup was created and contains the original state
        assert result.backup_path is not None
        assert result.backup_path.exists()
        backup = json.loads(result.backup_path.read_text())
        assert backup["location"] == "OLD DRIVE OST"


class TestRedriveLoggingSetup:
    """The ``__main__``/``main`` entrypoint routes through the shared
    ``configure_logging`` instead of a bare ``logging.basicConfig``, so
    errors render with the standard handlers, not plain INFO lines."""

    def test_main_configures_logging(self, tmp_path, monkeypatch):
        from gomoveshift.cli import logging_setup
        from gomoveshift.pipeline import redrive_ost

        calls: list[dict] = []
        monkeypatch.setattr(
            logging_setup,
            "configure_logging",
            lambda **kw: calls.append(kw),
        )
        # Stop after logging is configured — a missing state file exits 2.
        state_path = tmp_path / "does_not_exist.json"

        rc = redrive_ost.main([str(state_path)])

        assert rc == 2  # FileNotFoundError path
        assert calls, "main() must call configure_logging"
        assert calls[0] == {"verbose": False}
