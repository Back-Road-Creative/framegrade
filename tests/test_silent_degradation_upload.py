"""Silent-degradation regression tests for upload / credential / resume paths.

Guards log-only fixes for silent degradation on the upload, credential and
resume paths.

Each test forces a fallback path and asserts that
  (a) the fallback behavior ran and
  (b) a WARNING was emitted naming the preferred path / what was lost,
so an operator watching the console can see the degradation.

Covered cases:
  * uploader.py pre-flight credential refresh silent swallow
  * _yt_creds.py SOPS attempt failures logged at debug (now WARNING)
  * drive_orchestrator._try_resume silently dropped CLI overrides
    (shorts_target / shorts_gap) when CLI flag not re-passed on resume
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


UPLOADER_LOGGER = "gomoveshift.pipeline.uploader"
YT_CREDS_LOGGER = "gomoveshift.pipeline._yt_creds"
ORCH_LOGGER = "gomoveshift.pipeline.drive_orchestrator"


# ───────────────────────────────────────────────────────────── uploader.py


class TestPreflightCredentialRefreshWarns:
    """uploader.upload_shorts_from_dir pre-flight refresh must WARN on failure."""

    def test_refresh_exception_logged_as_warning(self, tmp_path, caplog):
        """Refresh raising Exception must emit a WARNING naming the fallback.

        Previously ``except Exception: pass`` hid network errors completely;
        parallel upload threads would then race on their own refreshes
        (potential heap corruption under httplib2).
        """
        from gomoveshift.pipeline.uploader import PipelineUploader

        # Build an uploader without triggering credential loading.
        uploader = PipelineUploader.__new__(PipelineUploader)
        uploader._creds = MagicMock()
        uploader._creds.refresh_token = "rt-123"  # noqa: S105 — test fixture, not a real secret

        def boom_refresh(_request):
            raise RuntimeError("network down")

        uploader._creds.refresh.side_effect = boom_refresh

        # Minimal shorts dir with one file so upload_shorts_from_dir proceeds
        # past the "no files" guard before we short-circuit.
        shorts_dir = tmp_path / "shorts"
        shorts_dir.mkdir()
        (shorts_dir / "short_0001.mp4").write_bytes(b"fake")

        # Stub everything after the pre-flight refresh so the test is
        # isolated to the refresh-swallow block.
        uploader._get_client = MagicMock(return_value=MagicMock())
        uploader.drive_folder = "root"
        uploader._get_or_create_folder_path = MagicMock(return_value="folder-id")
        uploader._build_drive_service = MagicMock(return_value=MagicMock())
        uploader._upload_file = MagicMock(return_value="https://drive/link")

        with caplog.at_level(logging.WARNING, logger=UPLOADER_LOGGER):
            uploader.upload_shorts_from_dir(
                shorts_dir=str(shorts_dir),
                project_name="test",
                make_shareable=False,
                max_workers=1,
            )

        # Fallback ran — refresh raised but the method continued.
        uploader._creds.refresh.assert_called_once()
        # WARNING surfaced the degradation.
        messages = [r.getMessage() for r in caplog.records if r.name == UPLOADER_LOGGER]
        assert any(
            "pre-flight credential refresh failed" in m and "per-thread refresh" in m
            for m in messages
        ), f"expected pre-flight refresh failure warning, got: {messages!r}"


# ───────────────────────────────────────────────────────────── _yt_creds.py


class TestSopsAttemptFailureWarns:
    """_yt_creds.load_youtube_credentials must WARN on SOPS attempt failures."""

    def test_sops_nonzero_exit_warns(self, tmp_path, monkeypatch, caplog):
        """Non-zero return from sops (stderr populated) must be a WARNING."""
        from gomoveshift.pipeline import _yt_creds

        # Lay down a fake encrypted-token file so the SOPS branch executes.
        # The file must live at creds_dir.parent / "youtube_upload_token.json.enc".
        creds_dir = tmp_path / "credentials" / "gomoveshift"
        creds_dir.mkdir(parents=True)
        sops_file = tmp_path / "credentials" / "youtube_upload_token.json.enc"
        sops_file.write_bytes(b"fake-encrypted")

        call_count = {"n": 0}

        def fake_run(cmd, *_a, **_kw):
            call_count["n"] += 1
            r = MagicMock()
            r.returncode = 1
            r.stdout = ""
            r.stderr = "decrypt failed: no MAC"
            return r

        monkeypatch.setattr(_yt_creds.subprocess, "run", fake_run)

        with caplog.at_level(logging.WARNING, logger=YT_CREDS_LOGGER):
            with pytest.raises(FileNotFoundError):
                _yt_creds.load_youtube_credentials(credentials_dir=creds_dir)

        # Both SOPS attempts were made and the fallback ran
        # (FileNotFoundError after BrandCredentials also fails).
        assert call_count["n"] >= 1
        messages = [r.getMessage() for r in caplog.records if r.name == YT_CREDS_LOGGER]
        # Per-attempt warning plus the final "All SOPS decrypt attempts failed"
        # warning should both be present.
        assert any("SOPS attempt failed" in m and "trying next source" in m for m in messages), (
            f"expected per-attempt SOPS warning, got: {messages!r}"
        )
        assert any("All SOPS decrypt attempts failed" in m for m in messages), (
            f"expected final SOPS aggregate warning, got: {messages!r}"
        )

    def test_sops_filenotfound_warns(self, tmp_path, monkeypatch, caplog):
        """Missing sops binary (FileNotFoundError) must also WARN."""
        from gomoveshift.pipeline import _yt_creds

        creds_dir = tmp_path / "credentials" / "gomoveshift"
        creds_dir.mkdir(parents=True)
        sops_file = tmp_path / "credentials" / "youtube_upload_token.json.enc"
        sops_file.write_bytes(b"fake-encrypted")

        def fake_run(cmd, *_a, **_kw):
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(_yt_creds.subprocess, "run", fake_run)

        with caplog.at_level(logging.WARNING, logger=YT_CREDS_LOGGER):
            with pytest.raises(FileNotFoundError):
                _yt_creds.load_youtube_credentials(credentials_dir=creds_dir)

        messages = [r.getMessage() for r in caplog.records if r.name == YT_CREDS_LOGGER]
        assert any(
            "binary" in m and "not found" in m and "trying next source" in m for m in messages
        ), f"expected missing-binary warning, got: {messages!r}"


# ─────────────────────────────────────────────────── drive_orchestrator.py


class TestResumeShortsOverride:
    """_try_resume persists shorts_target / shorts_gap on state so they
    survive resume without re-passing the CLI flag. The "override loss"
    WARNING fires only when a fresh CLI flag conflicts with a persisted
    state value (stale CLI arg), not on the happy path where state wins
    silently.
    """

    def _build_state_file(self, state_file: Path, saved: dict) -> None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(saved))

    def test_shorts_target_persisted_override_survives_without_cli_flag(self, tmp_path, caplog):
        """Happy path: state has shorts_target=5, CLI flag absent on
        resume. Override must be restored onto state AND re-applied to
        config.drive for in-process planners. No warning fires — silent
        state-wins is the new expected behavior."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        self._build_state_file(
            state_file,
            {
                "phases_completed": ["phase_b_preview"],
                "shorts_target": 5,
            },
        )

        state = PipelineState(source_dir=Path("/tmp/src"), output_base=tmp_path)
        # Do NOT set state["shorts_target"] — simulates missing CLI flag.
        stages_completed: list[str] = []

        with caplog.at_level(logging.WARNING, logger=ORCH_LOGGER):
            orch._try_resume(
                state,
                state_file,
                tmp_path,
                stages_completed,
                fresh=False,
            )

        # State now carries the persisted override.
        assert state.get("shorts_target") == 5, (
            "Persisted shorts_target did not survive resume onto state."
        )
        # config.drive.* mirrors state (backward-compat for in-process readers).
        assert orch.config.drive.shorts_target_count == 5, (
            "Persisted shorts_target was not re-applied "
            "to config.drive.shorts_target_count on resume."
        )
        # No warning on the happy path — state wins silently.
        messages = [r.getMessage() for r in caplog.records if r.name == ORCH_LOGGER]
        assert not any("Resume CLI override" in m for m in messages), (
            f"No warning expected on state-wins happy path; got: {messages!r}"
        )

    def test_shorts_gap_persisted_override_survives_without_cli_flag(self, tmp_path, caplog):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        self._build_state_file(
            state_file,
            {
                "phases_completed": ["phase_b_preview"],
                "shorts_gap": 45,
            },
        )

        state = PipelineState(source_dir=Path("/tmp/src"), output_base=tmp_path)
        stages_completed: list[str] = []

        with caplog.at_level(logging.WARNING, logger=ORCH_LOGGER):
            orch._try_resume(
                state,
                state_file,
                tmp_path,
                stages_completed,
                fresh=False,
            )

        assert state.get("shorts_gap") == 45
        assert orch.config.drive.shorts_min_gap_sec == 45, (
            "Persisted shorts_gap was not re-applied to config.drive.shorts_min_gap_sec on resume."
        )
        messages = [r.getMessage() for r in caplog.records if r.name == ORCH_LOGGER]
        assert not any("Resume CLI override" in m for m in messages)

    def test_conflict_warning_fires_when_fresh_cli_disagrees_with_state(self, tmp_path, caplog):
        """Inverted warning: fire only when BOTH a persisted
        state value AND a fresh CLI flag are present AND they differ
        (stale CLI arg vs. authoritative state). State wins — warning
        tells the operator how to force the CLI value if desired."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        self._build_state_file(
            state_file,
            {
                "phases_completed": ["phase_b_preview"],
                "shorts_target": 5,
            },
        )

        state = PipelineState(source_dir=Path("/tmp/src"), output_base=tmp_path)
        # Simulate CLI --shorts-target 7 on this invocation that disagrees
        # with the persisted state value of 5.
        state["shorts_target"] = 7
        stages_completed: list[str] = []

        with caplog.at_level(logging.WARNING, logger=ORCH_LOGGER):
            orch._try_resume(
                state,
                state_file,
                tmp_path,
                stages_completed,
                fresh=False,
            )

        # State wins the conflict.
        assert state.get("shorts_target") == 5, (
            "The state-wins rule must preserve the persisted value "
            "even when a conflicting CLI flag is passed on resume."
        )
        # Warning fired so the operator sees why the CLI arg was ignored.
        messages = [r.getMessage() for r in caplog.records if r.name == ORCH_LOGGER]
        assert any(
            "Resume CLI override conflict" in m and "shorts_target" in m and "state wins" in m
            for m in messages
        ), f"expected conflict warning, got: {messages!r}"

    def test_no_warning_when_cli_matches_state(self, tmp_path, caplog):
        """When CLI flag is re-passed with the same value as state, no
        warning — values agree, nothing to flag."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())

        state_file = tmp_path / "pipeline_state.json"
        self._build_state_file(
            state_file,
            {
                "phases_completed": ["phase_b_preview"],
                "shorts_target": 5,
            },
        )

        state = PipelineState(source_dir=Path("/tmp/src"), output_base=tmp_path)
        state["shorts_target"] = 5  # Same as persisted.
        stages_completed: list[str] = []

        with caplog.at_level(logging.WARNING, logger=ORCH_LOGGER):
            orch._try_resume(
                state,
                state_file,
                tmp_path,
                stages_completed,
                fresh=False,
            )

        messages = [r.getMessage() for r in caplog.records if r.name == ORCH_LOGGER]
        assert not any("Resume CLI override" in m for m in messages), (
            f"No warning expected when CLI matches state; got: {messages!r}"
        )


# ─────────────────────────────────────────────── resume sentinel CLI-wins

# Per-invocation gate-bypass + cleanup-opt-out flags that must follow
# CLI-wins-on-resume semantics,
# exactly like skip_upload / youtube_upload / allow_seo_degraded. Each is an
# argparse-sentinel: PipelineState default is None so the schedule_overrides
# allowlist in _try_resume can tell "operator passed it" (True) from "absent"
# (None). The whole family lives behind this one parametrized guard so the
# next forgotten flag fails loudly.
RESUME_SENTINEL_FLAGS = [
    "ost_audit_confirmed",
    "gate_cd_confirmed",
    "force_sidecar_reset",
    "keep_intermediates",
    "keep_masters",
    "allow_short_count",
    # no_youtube_upload is ALWAYS_CLI (see flag_registry row):
    # paired with youtube_upload, both re-decided every invocation so a saved
    # True is clearable by omitting the flag. Covered by
    # tests/test_flag_registry.py::TestAlwaysCliUploadFlags, not this
    # sentinel contract ("saved True survives CLI-absent" is intentionally
    # FALSE for it now).
]


class TestResumeSentinelCliWins:
    """Resume must honor explicit CLI gate/cleanup flags over saved state.

    Two directions per flag:
      * CLI True beats a saved False — the headline bug: the flag was
        silently discarded on resume because it was absent from the
        ``schedule_overrides`` allowlist in ``_try_resume``.
      * A saved True survives a CLI-absent resume — guards the inverse
        trap: if the flag's ``PipelineState`` default were a concrete
        ``False`` instead of ``None``, the allowlist would capture that
        ``False`` from the fresh state and re-stomp the saved ``True``.
    """

    def _build_state_file(self, state_file: Path, saved: dict) -> None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(saved))

    @pytest.mark.parametrize("flag", RESUME_SENTINEL_FLAGS)
    def test_cli_true_beats_saved_false_on_resume(self, flag, tmp_path):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())
        state_file = tmp_path / "pipeline_state.json"
        self._build_state_file(
            state_file,
            {"phases_completed": ["phase_b_preview"], flag: False},
        )

        state = PipelineState(source_dir=Path("/tmp/src"), output_base=tmp_path)
        state[flag] = True  # operator passed --<flag> THIS invocation
        stages_completed: list[str] = []

        orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)

        assert state.get(flag) is True, (
            f"{flag}: explicit CLI True was stomped by the saved False on "
            f"resume — flag missing from the schedule_overrides allowlist."
        )

    @pytest.mark.parametrize("flag", RESUME_SENTINEL_FLAGS)
    def test_saved_true_survives_cli_absent_on_resume(self, flag, tmp_path):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
        from gomoveshift.pipeline.drive_state_model import PipelineState

        orch = DrivePipelineOrchestrator(config=Config())
        state_file = tmp_path / "pipeline_state.json"
        self._build_state_file(
            state_file,
            {"phases_completed": ["phase_b_preview"], flag: True},
        )

        # Flag NOT set on the fresh state — simulates the operator NOT
        # re-passing the flag on resume. Requires the PipelineState default
        # to be None so the allowlist skips it (no captured False to stomp
        # the saved True with).
        state = PipelineState(source_dir=Path("/tmp/src"), output_base=tmp_path)
        stages_completed: list[str] = []

        orch._try_resume(state, state_file, tmp_path, stages_completed, fresh=False)

        assert state.get(flag) is True, (
            f"{flag}: saved True did not survive a CLI-absent resume — the "
            f"PipelineState default is likely False (not None), so the "
            f"allowlist captured the fresh-state default and re-stomped it."
        )
